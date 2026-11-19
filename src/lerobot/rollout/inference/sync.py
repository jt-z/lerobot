# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)


# TODO(Steven): support relative-action policies.  The per-tick flow refreshes
# ``RelativeActionsProcessorStep._last_state`` every call, so cached chunk
# actions popped on later ticks get reanchored to the *current* robot state and
# absolute targets drift through the chunk.  Relative-action policies are
# rejected at context-build time today; RTC postprocesses the whole chunk and
# is unaffected.
#
# Candidate fix: drive the policy via ``predict_action_chunk`` and serve a
# local FIFO of postprocessed actions.  Eliminates drift by construction and
# saves per-tick pre/post work, but bypasses ``select_action`` — needs
# fallbacks for SAC (raises), ACT temporal ensembling (ensembler lives in
# ``select_action``), and Diffusion-family (obs-history queues populated as a
# side effect of ``select_action``).


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        seam_max_step_deg: float = 3.0,
        seam_jump_threshold_deg: float = 12.0,
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        # Seam smoothing: chunking policies execute their chunk open-loop, so the
        # action re-planned after a chunk is used up can differ from the previous
        # command by tens of degrees (tracking error accumulated over the chunk).
        # A step larger than ``seam_jump_threshold_deg`` is spread over as many
        # ticks as needed to keep every joint at or below ``seam_max_step_deg``
        # per tick — the tick count is derived from the jump, not configured.
        self._seam_max_step = float(seam_max_step_deg)
        self._seam_threshold = float(seam_jump_threshold_deg)
        self._seam_active = False
        self._last_dispatched: torch.Tensor | None = None
        # The gripper is a near-binary actuator and closes/opens at full speed; rate
        # limiting it would only delay grasps by seconds.  Smooth arm joints only.
        self._seam_mask = torch.tensor(
            ["gripper" not in k.lower() for k in ordered_action_keys], dtype=torch.bool
        )
        if self._seam_max_step <= 0:
            logger.info("Seam smoothing disabled (seam_max_step_deg <= 0)")
        else:
            logger.info(
                "Seam smoothing enabled: arm steps > %.1f deg spread at <= %.2f deg per tick (gripper unaffected)",
                self._seam_threshold,
                self._seam_max_step,
            )
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        # A reset starts a new command stream: the first action is not a seam.
        self._seam_active = False
        self._last_dispatched = None
        # The policy was just reset, so a pending task change has nothing stale to flush.
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        task, task_changed = self._take_task()
        with torch.inference_mode(), autocast_ctx:
            if task_changed:
                # Chunking policies queue actions computed under the previous instruction,
                # so drop them and let the new one take effect on this tick.  Narrower
                # than ``policy.reset``: observation history and other episode state stay.
                logger.info("Task changed to '%s' — dropping precomputed actions", task)
                self._policy.drop_queued_actions()
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        # ``task`` is the pre-inference snapshot: a /subtask landing mid-inference must
        # not relabel this action.
        self._set_dispatched_task(task)
        return self._apply_seam_blend(torch.tensor([action_dict[k] for k in self._ordered_action_keys]))

    def _apply_seam_blend(self, action: torch.Tensor) -> torch.Tensor:
        """Rate-limit a chunk-seam step so no joint moves faster than the cap.

        A chunking policy re-plans only when its queue runs dry, and the new plan is
        anchored to the measured state while the previous command came from a
        prediction made a whole chunk ago.  The difference is a step at the seam.
        Once a step above ``seam_jump_threshold_deg`` is seen, the command follows
        the new chunk at no more than ``seam_max_step_deg`` per tick until it has
        caught up, so how many ticks the transition takes follows the jump size
        (44° at 3°/tick → 15 ticks) instead of being a fixed number.  Motion below
        the threshold is passed through untouched, and the gripper is never limited.
        Disabled when ``seam_max_step_deg <= 0``.
        """
        if self._seam_max_step <= 0:
            self._last_dispatched = action
            return action

        last = self._last_dispatched
        if last is None:
            self._last_dispatched = action
            return action

        delta = action - last
        if not self._seam_active:
            jump = float(delta[self._seam_mask].abs().max()) if self._seam_mask.any() else 0.0
            if jump <= self._seam_threshold:
                self._last_dispatched = action
                return action
            self._seam_active = True
            logger.debug(
                "Chunk seam (%.1f deg) — rate-limited to %.2f deg/tick", jump, self._seam_max_step
            )

        limited = delta.clamp(-self._seam_max_step, self._seam_max_step)
        out = last + torch.where(self._seam_mask, limited, delta)
        remaining = float((action - out)[self._seam_mask].abs().max()) if self._seam_mask.any() else 0.0
        if remaining <= 1e-4:
            self._seam_active = False
        self._last_dispatched = out
        return out

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when the policy has a text head."""
        return self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """Inference runs inline on the control thread, so queries are served there too."""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on the current observation."""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        # Live task, read without consuming the task-changed edge (the action path needs it).
        task = self.task
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(obs_frame, self._device, task, self._robot_type)
            observation = self._mark_query(observation, query)
            # Reusing the action path's preprocessor is safe only while its steps are
            # stateless per call; the one that is not (an enabled RelativeActionsProcessorStep)
            # is rejected for this backend at context-build time.
            observation = self._preprocessor(observation)
            # No str() coercion: _service_query validates the return value.
            return self._policy.generate_text(observation)

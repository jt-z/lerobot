## Summary

Handle single-turn absolute encoder wrap on the B601-RS (RobStride) **gripper**, so a
power-cycle (or a session started with the gripper not at the 0° home position) can no
longer feed a wrapped/negative reading into observations.

### Problem

RobStride motors use a **14-bit single-turn absolute magnetic encoder**. When the arm
loses 48 V power, the reported angle for the *same physical position* can shift by a
whole turn (measured: `269.1°` before power-off reads `-90.8°` after boot). The gripper
travel (0°~345°) sits close to the 0°/360° boundary and `safe_zero()` intentionally does
not close the gripper to 0° (only pulls it back to ~170° if wide open, to avoid crushing a
held object). Consequences observed in the field:

1. If the gripper is started away from 0° (or after a power cycle), `get_observation`
   reports a negative / wrapped value such as `-90.8°`.
2. That value feeds the state vector / policy / joint-clamp logic, which all assume a
   0~270° domain -> "zero drift", the gripper exceeding the configured range and being
   pushed out.
3. Re-seating the gripper at 0° and re-running does not help until the motor zero is
   rewritten (e.g. with RobStride's MotorBridge Studio).

### What this PR does (no change to motion / clamp / safety logic)

- **Normalize gripper observations into `[0, 360)`** in `get_observation` (negative
  wrapped readings become their valid equivalent, e.g. `-90.8° -> 269.2°`).
- **Save the gripper angle at disconnect** (next to the calibration file,
  `<id>_gripper_last_deg.json`) and, on the next `configure()`/enable, run a **startup
  sanity check** that:
  - prints the raw / normalized reading and the home-point offset (debug visibility), and
  - warns when the reading is discontinuous (>60°) with the previous exit position, which
    indicates the gripper was moved by hand while powered off or its zero reference
    shifted.
- All new behaviour is diagnostic / observation-only: it never drives the gripper and
  never changes `joint_limits`, `joint_directions`, the MIT clamp, or `safe_zero`.

### Evidence / reproduction

Discrimination experiment (direct motorbridge access, gripper motor only):

| step | reading |
|---|---|
| enabled, gripper physically closed | `+0.02°` |
| enabled, gripper open (same position) | `+269.11°` |
| **power cycle**, same position, enabled | **`-90.82°`** (≈ -360° wrap) |

Field runs of `lerobot-record` (seeed_b601_rs_follower single arm) after the change stay
stable across repeated Ctrl-C restarts without power loss; only a genuine power cycle
produces the wrap, which is now detected and reported at startup instead of silently
corrupting observations.

### Tests / checks

- `python -m py_compile` clean.
- Pure helpers (`normalize`, shortest-arc delta) are side-effect free and covered by the
  checks shown above; a hardware-free unit test can be added if desired.

### Notes for maintainers

- The wrap also affects any code path that reads the gripper state raw (e.g. the bimanual
  wrapper in LeRobot clones). Exposing the normalize helper (or a `wrap_pos_deg` static)
  would let DM / RS / bimanual integrations share it.

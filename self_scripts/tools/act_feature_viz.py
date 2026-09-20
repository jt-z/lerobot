#!/usr/bin/env python
"""lerobot-rollout 的 ACT 内部可视化包装器（运行时打补丁，不改 lerobot / ACT 源码）。

用法与 lerobot-rollout 完全一致，只是把命令换成：
    python act_feature_viz.py --strategy.type=base --policy.path=... ...

可视化内容（全部来自 decoder 的注意力权重，不改前向计算）：
    主叠图（每 tick）act_viz/attn/<cam>：
      最后一层 decoder 的 cross-attention = 「action chunk 的 100 个动作 query → 902 个 token」。
      902 = 1(latent) + 1(robot_state) + 3×300(三路相机的 15×20 feature map 展平)。
      按 head 聚合（mean/max）后切回每个相机的 15×20 网格，双线性上采样到 480×640，
      jet 着色 + alpha 叠加在「该 chunk 生成时那一帧」的原始相机图上。
    分析视图（ACT_VIZ_ANALYSIS=1，默认开；都是低帧率更新，不挤占 30Hz 主循环）：
      ① act_viz/attn_heads/<cam>  per-head 网格：8 个 head 各一张 15×20 热图，
         2×4 排列（左上→右下 = head 0..7，黑线分隔）→ 看 head 是否分工（盯目标物 / 盯末端 / 全局）。
         每个 head 用自己整段 chunk 的 1%~99.5% 分位归一化（否则弱 head 一片蓝看不出结构）。
         刷新间隔 ACT_VIZ_HEADS_EVERY（默认 10 tick ≈ 3Hz）。
      ② act_viz/timeline/<cam>  「步骤×空间」图（每个 chunk 一张）：
         横轴 = chunk 内第 k 步（0→99），纵轴 = token 索引（15×20 展平，每 20 行画灰线分隔
         feature-map 的行）→ 看注意力随动作推进的漂移（抓取前盯目标 / 移动中盯路径…）。
      ③ act_viz/mass/*  注意力质量曲线（每 tick 一个点）：
         mass(cam) = Σ_{j∈cam} A[k, j]（该 query 的注意力预算有多少落在这一路相机上），
         另有 latent/state 质量与当前 k。配合 TimeSeriesView 定量看"什么时候在看哪路相机"。
      ④ act_viz/selfattn  decoder 自注意力（每个 chunk 一张 100×100）：
         100 个动作 query 之间的注意力（行 = query 步，列 = key 步；ACT 的 decoder 无 mask，
         双向；q=k=token+pos_embed），看 chunk 内动作是否分段/成组。

为什么能做到"实时滚动"：
    ACT（chunk_size=100, n_action_steps=100, 无 temporal ensemble）每 100 次动作解码才前向一次
    （30Hz 下约 3.3s）。这里缓存整段注意力（100×3×15×20，约 360KB），每 N 次取动作换一帧显示
    「当前执行到 chunk 内第 k 步」的注意力分布，因此画面是连续变化的，但不需要额外前向。
    （k 直接对应 ACT 动作队列弹出到第几个，与下发的动作严格同步。）

离线 MP4（默认开启）：
    每渲染一帧同时写一份 H.264 视频，默认把 hand/front/top 三路横向拼成一张
    <输出目录>/act_attn_tiled.mp4（1920×480），跑完直接看这一个文件即可回顾整段
    注意力变化；帧率按 fps/(EVERY_N×interpolation_multiplier) 计算，视频时长=真实时长。
    与 Rerun 是否可用无关（--display_data=false 时也能只落 MP4）；Ctrl+C 正常退出会
    写完 moov 收尾，强杀（kill -9）会导致文件不完整。

实现约束（关键）：
  - 不改前向计算路径：decoder 的 multihead_attn 只挂 forward_pre_hook 抓 q/k 的引用，之后用模块
    自带的 in_proj_weight/in_proj_bias 事后复现 softmax(qkᵀ/√d)（与 PyTorch SDPA 数学等价）；
    不传 need_weights=True，避免把 attention 原路物化拖慢内联在控制线程里的前向。
  - 叠图渲染 + rr.log + MP4 编码全部丢到后台线程（只保留最新一帧），不占用 30Hz 控制循环。
  - 任何失败都只提示一次，绝不影响推理。

环境变量：
    ACT_VIZ_EVERY_N     每 N 次策略取动作刷一帧叠图（默认 1 ≈ 30Hz；调大则画面更省但更卡）
    ACT_VIZ_ANALYSIS    1（默认）= 额外输出 per-head 网格 / 步骤×空间图 / 质量曲线 / 自注意力；0 = 只留主叠图
    ACT_VIZ_HEADS_EVERY per-head 网格刷新间隔（单位 tick，默认 10 ≈ 3Hz）
    ACT_VIZ_HEADS       head 聚合方式：mean（默认）| max（仅影响主叠图与曲线；per-head 网格永远是单 head）
    ACT_VIZ_ALPHA       热力图最大叠加强度 0~1（默认 0.55，按注意力强弱渐变）
    ACT_VIZ_BLUEPRINT   1（默认）= 补发包含 act_viz 视图的 Rerun blueprint；0 = 不动布局
    ACT_VIZ_ENTITY_ROOT Rerun 实体前缀（默认 act_viz）
    ACT_VIZ_MP4         1（默认）= 同时落 MP4；0 = 只在线看
    ACT_VIZ_MP4_DIR     输出目录（默认 <本文件目录>/logs/act_viz_<时间戳>/）
    ACT_VIZ_MP4_MODE    tiled（默认，三路横拼一个文件）| per_cam（每路一个文件）| both
    ACT_VIZ_MP4_CRF     x264 质量，越小越清晰越大（默认 23）
    ACTION_TRACE        1 = 顺带装上 action_trace.py 的补丁（同一入口同时拿到动作追踪）
"""

from __future__ import annotations

import os
import sys
import threading
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

# ==================== 参数（环境变量） ====================
EVERY_N = max(1, int(os.environ.get("ACT_VIZ_EVERY_N", "1")))
ANALYSIS = os.environ.get("ACT_VIZ_ANALYSIS", "1") == "1"
HEADS_EVERY = max(1, int(os.environ.get("ACT_VIZ_HEADS_EVERY", "10")))
HEADS_AGG = os.environ.get("ACT_VIZ_HEADS", "mean").strip().lower()
ALPHA = min(1.0, max(0.0, float(os.environ.get("ACT_VIZ_ALPHA", "0.55"))))
SEND_BLUEPRINT = os.environ.get("ACT_VIZ_BLUEPRINT", "1") == "1"
ENTITY_ROOT = os.environ.get("ACT_VIZ_ENTITY_ROOT", "act_viz").strip("/")
MP4_ENABLED = os.environ.get("ACT_VIZ_MP4", "1") == "1"
MP4_MODE = os.environ.get("ACT_VIZ_MP4_MODE", "tiled").strip().lower()
MP4_DIR = os.environ.get("ACT_VIZ_MP4_DIR", "").strip()
MP4_CRF = int(os.environ.get("ACT_VIZ_MP4_CRF", "23"))
HEAD_TILE = (120, 160)  # ① per-head 网格里每个 head 的小图 (高, 宽)
HEAD_GAP = 2  # 网格分隔线宽度（取偶数才能保证总宽高为偶数，yuv420p 需要）
TIMELINE_SCALE = 2  # ② 步骤×空间图放大倍数
SELFATTN_SCALE = 3  # ④ 自注意力矩阵放大倍数
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# 总开关：自测脚本用它关掉注意力计算做耗时对比（不影响包装器正常使用）
ACTIVE = True

# ==================== 运行期状态 ====================
_lock = threading.Lock()
_capture: dict = {}  # 本次前向从 cross-attn pre-hook 抓到的 {"q","k","module"}
_capture_self: dict = {}  # 本次前向从 decoder self-attn pre-hook 抓到的 {"q","k","module"}
_last_qk: tuple | None = None  # 最近一次前向的 (q, k, module)，仅供自测/排查
_chunk_attn: np.ndarray | None = None  # (steps, n_cams, gh, gw) float32，原始（未归一化）
_chunk_heads: np.ndarray | None = None  # (n_heads, steps, n_cams, gh, gw)，逐 head 未聚合
_chunk_self: np.ndarray | None = None  # (steps, steps)，decoder 自注意力（head 平均）
_chunk_prefix: np.ndarray | None = None  # (steps,)，latent/robot_state token 的注意力质量
_chunk_base: list[np.ndarray] | None = None  # chunk 生成时那一帧的各相机底图 (H,W,3) uint8
_pending_base: list[np.ndarray] | None = None  # 本 tick 暂存的底图，forward 发生时提交给 _chunk_base
_cam_names: list[str] = []
_grid: tuple[int, int] | None = None  # backbone layer4 的 (gh, gw)
_n_steps = 0
_step = 0
_tick = 0
_chunk_id = 0  # 每前向一次 +1（后台线程据此判断"这个 chunk 的分析图还没画过"）
_rendered_chunk_id = -1  # 后台线程已画过分析图的 chunk

_installed = False
_warned: set[str] = set()
_blueprint_sent = False
_first_log_ts = 0.0
_last_chunk_ts = 0.0
_stats = {"forward": 0, "attn_ms": 0.0, "submit": 0, "render": 0, "logged": 0, "analysis": 0}


def _warn(msg: str) -> None:
    """同类提示只打一次，且绝不影响推理。"""
    if msg in _warned:
        return
    _warned.add(msg)
    print(f"[act_viz] 降级（推理不受影响）：{msg}", file=sys.stderr, flush=True)


# ==================== 图像工具 ====================
def _to_u8_hwc(x) -> np.ndarray | None:
    """torch/numpy 图像 -> (H,W,3) uint8；CHW 自动转 HWC，0~1 浮点自动放大到 0~255。"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(arr.max(initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _jet(x: np.ndarray) -> np.ndarray:
    """x∈[0,1] (H,W) -> jet 伪彩 (H,W,3) uint8（标准 jet 的分段线性近似）。"""
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _p_lo_hi(values: np.ndarray) -> tuple[float, float]:
    """1%~99.5% 分位（注意力值域极窄，直接用 min/max 会被个别尖峰压成全蓝）。"""
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.5))


def _norm01(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def _upsample01(h: np.ndarray, height: int, width: int) -> np.ndarray:
    """(h,w) float -> (height,width) float 双线性（用 torch，避免依赖 cv2）。"""
    t = torch.from_numpy(np.ascontiguousarray(h, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)[0, 0].numpy()


# ==================== MP4 录制 ====================
_mp4_lock = threading.Lock()
_mp4_writers: dict[str, "_Mp4Writer"] = {}  # key = 文件名（不含 .mp4）
_mp4_paths: list[str] = []
_mp4_count: dict[str, int] = {}  # key -> 已写入帧数
_mp4_bad: set[str] = set()  # 写失败的 key，只停它自己，不影响其它文件
_mp4_dir = ""
_mp4_closed = False


def _argv_value(flag: str) -> str | None:
    """从命令行取 --flag=value / --flag value（落 MP4 时要用到 --fps 与插值倍率）。"""
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f"--{flag}="):
            return arg.split("=", 1)[1]
        if arg == f"--{flag}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _mp4_fps() -> float:
    """可视化帧率 = 控制频率 / (EVERY_N × interpolator 倍率)，保证视频时长=真实时长。"""
    try:
        control_fps = float(_argv_value("fps") or 30.0)
    except ValueError:
        control_fps = 30.0
    try:
        multiplier = max(1, int(_argv_value("interpolation_multiplier") or 1))
    except ValueError:
        multiplier = 1
    return max(1.0, control_fps / (EVERY_N * multiplier))


class _Mp4Writer:
    """极简 PyAV H.264 写出（与 lerobot 数据集编码同一后端）。"""

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        import av

        self.path = path
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
        self.stream.width, self.stream.height = width, height  # 注意是 宽, 高（写反会导致画面被缩扁变形）
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(MP4_CRF), "preset": "veryfast"}

    def write(self, image: np.ndarray) -> None:
        import av

        h, w = image.shape[:2]
        if (w, h) != (self.stream.width, self.stream.height):
            # 尺寸不符时 PyAV 会隐式缩放，画面会被压扁/拉长 → 宁可报错也不要录出变形的视频
            raise ValueError(f"帧尺寸 {w}x{h} 与视频流 {self.stream.width}x{self.stream.height} 不一致")
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        try:
            for packet in self.stream.encode():  # flush
                self.container.mux(packet)
        finally:
            self.container.close()


def _mp4_fps_per(ticks: int) -> float:
    """每 ticks 个控制 tick 才产生一帧的视频帧率（分析视图用，保证时长=真实时长）。"""
    return max(1e-3, _mp4_fps() / max(1, ticks))


def _mp4_out_dir() -> str:
    """输出目录（默认 <本文件目录>/logs/act_viz_<启动时间戳>/），懒创建并复用。"""
    global _mp4_dir
    if not _mp4_dir:
        _mp4_dir = MP4_DIR or os.path.join(_HERE, "logs", f"act_viz_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(_mp4_dir, exist_ok=True)
    return _mp4_dir


def _mp4_writer(key: str, fps: float, width: int, height: int) -> "_Mp4Writer | None":
    """按 key 懒创建 writer；失败只提示一次并返回 None。"""
    if key in _mp4_writers:
        return _mp4_writers[key]
    if width % 2 or height % 2:  # yuv420p 要求偶数宽高
        _warn(f"MP4 需要偶数宽高，当前 {width}×{height}（{key}），该文件跳过")
        return None
    path = os.path.join(_mp4_out_dir(), f"{key}.mp4")
    _mp4_writers[key] = _Mp4Writer(path, fps, width, height)
    _mp4_paths.append(path)
    print(f"[act_viz] 开始录制 {os.path.basename(path)}（{width}x{height}, {fps:.2f} fps）", flush=True)
    return _mp4_writers[key]


def _mp4_put(key: str, fps: float, image: np.ndarray | None) -> None:
    """把一帧写进 key 对应的 MP4。H.264 编码在后台线程里做，不影响控制循环。"""
    if not MP4_ENABLED or image is None or _mp4_closed or key in _mp4_bad:
        return
    with _mp4_lock:
        try:
            writer = _mp4_writer(key, fps, int(image.shape[1]), int(image.shape[0]))
            if writer is None:
                _mp4_bad.add(key)
                return
            writer.write(image)
            _mp4_count[key] = _mp4_count.get(key, 0) + 1
        except Exception as exc:
            _mp4_bad.add(key)
            _warn(f"MP4 写入失败（{key}），该文件停止录制：{exc!r}")


def _write_mp4(frames: list[np.ndarray]) -> None:
    """主叠图落盘：tiled（三路横拼一个文件）/ per_cam（每路一个文件）。"""
    if not frames:
        return
    fps = _mp4_fps()
    same_size = len({f.shape for f in frames}) == 1
    if MP4_MODE in ("tiled", "both") and len(frames) == len(_cam_names) and same_size:
        _mp4_put("act_attn_tiled", fps, np.concatenate(frames, axis=1))
    if MP4_MODE in ("per_cam", "both"):
        for i, cam in enumerate(_cam_names):
            if i < len(frames):
                _mp4_put(f"act_attn_{cam}", fps, frames[i])


def _close_mp4() -> None:
    """收尾（写 moov），必须在进程退出前调用，否则文件不可播放。"""
    global _mp4_closed
    _mp4_closed = True  # 之后再有迟到的帧也不再写（不会重开文件）
    with _mp4_lock:
        for writer in _mp4_writers.values():
            try:
                writer.close()
            except Exception as exc:
                _warn(f"MP4 收尾失败：{exc!r}")
        _mp4_writers.clear()


# ==================== ACT 内部量捕获 ====================
def _backbone_hook(module, args, output) -> None:
    """只为拿 feature map 的空间尺寸（15×20），用于把 token 还原成像素网格。零额外算力。"""
    global _grid
    if not ACTIVE:
        return
    feat = output.get("feature_map") if isinstance(output, dict) else output
    if feat is None or feat.dim() != 4:
        return
    _grid = (int(feat.shape[-2]), int(feat.shape[-1]))


def _cross_attn_pre_hook(module, args, kwargs) -> None:
    """decoder 交叉注意力的输入（未经 in_proj 的 query/key）。只存引用，不改计算。"""
    if not ACTIVE:
        return
    query = kwargs.get("query", args[0] if args else None)
    key = kwargs.get("key", args[1] if len(args) > 1 else None)
    if query is None or key is None:
        return
    _capture["q"] = query
    _capture["k"] = key
    _capture["module"] = module


def _self_attn_pre_hook(module, args, kwargs) -> None:
    """decoder 自注意力的输入。ACT 里是 q = k = token + pos_embed，且没有任何 mask。"""
    if not ACTIVE or not ANALYSIS:
        return
    query = kwargs.get("query", args[0] if args else None)
    key = kwargs.get("key", args[1] if len(args) > 1 else None)
    if query is None or key is None:
        return
    _capture_self["q"] = query
    _capture_self["k"] = key
    _capture_self["module"] = module


def _cross_attn_raw(q: torch.Tensor, k: torch.Tensor, module) -> torch.Tensor:
    """复现 nn.MultiheadAttention 的注意力权重：(L,B,E),(S,B,E) -> (B,H,L,S)。

    与 PyTorch 的数学路径完全一致：q' = Wq·q, k' = Wk·k, A = softmax(q'k'ᵀ/√head_dim)。
    （ACT 里 kdim == vdim == embed_dim，所以用的是 in_proj_weight 分段。）
    """
    embed = module.embed_dim
    heads = module.num_heads
    head_dim = embed // heads
    weight = module.in_proj_weight
    bias = module.in_proj_bias
    wq, wk = weight[:embed], weight[embed : 2 * embed]
    bq, bk = (bias[:embed], bias[embed : 2 * embed]) if bias is not None else (None, None)

    qp = F.linear(q, wq, bq)  # (L, B, E)
    kp = F.linear(k, wk, bk)  # (S, B, E)
    length, batch, _ = qp.shape
    src_len = kp.shape[0]
    qp = qp.permute(1, 0, 2).reshape(batch, length, heads, head_dim).permute(0, 2, 1, 3)  # (B,H,L,hd)
    kp = kp.permute(1, 0, 2).reshape(batch, src_len, heads, head_dim).permute(0, 2, 1, 3)  # (B,H,S,hd)
    logits = torch.matmul(qp, kp.transpose(-1, -2)) * (head_dim**-0.5)
    return torch.softmax(logits, dim=-1)


def _aggregate_heads(attn: torch.Tensor) -> torch.Tensor:
    """(B,H,L,S) -> (B,L,S)。"""
    if HEADS_AGG == "max":
        return attn.amax(dim=1)
    return attn.mean(dim=1)


def _act_forward_hook(module, args, output) -> None:
    """ACT 前向结束后：把本次的交叉注意力切成每相机 (L, gh, gw) 缓存起来。

    ANALYSIS 打开时顺带缓存逐 head 注意力（①）、decoder 自注意力（④）和 prefix 质量（③）。
    """
    global _chunk_attn, _chunk_heads, _chunk_self, _chunk_prefix
    global _chunk_base, _step, _n_steps, _chunk_id, _last_chunk_ts, _last_qk
    if not ACTIVE:
        return
    q = _capture.pop("q", None)
    k = _capture.pop("k", None)
    attn_module = _capture.pop("module", None)
    q_s = _capture_self.pop("q", None)
    k_s = _capture_self.pop("k", None)
    attn_module_s = _capture_self.pop("module", None)
    if q is None or k is None or attn_module is None:
        return

    config = getattr(module, "config", None)
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            raw = _cross_attn_raw(q, k, attn_module)  # (B,H,L,S) 逐 head
            attn = _aggregate_heads(raw)  # (B,L,S)
            attn_np = attn[0].detach().float().cpu().numpy()
            heads_np = raw[0].detach().float().cpu().numpy() if ANALYSIS else None
            self_np = None
            if ANALYSIS and q_s is not None and k_s is not None and attn_module_s is not None:
                self_np = (
                    _aggregate_heads(_cross_attn_raw(q_s, k_s, attn_module_s))[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
        _last_qk = (q, k, attn_module)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        _warn(f"计算交叉注意力失败：{exc}")
        return

    n_heads_prefix = 1  # latent token
    if getattr(config, "robot_state_feature", None) is not None:
        n_heads_prefix += 1
    if getattr(config, "env_state_feature", None) is not None:
        n_heads_prefix += 1

    n_cams = len(_cam_names)
    steps, n_tokens = attn_np.shape
    n_img = n_tokens - n_heads_prefix
    if n_cams == 0 or _grid is None or n_img <= 0 or n_img % n_cams != 0:
        _warn(f"token 切分失败（tokens={n_tokens}, prefix={n_heads_prefix}, cams={n_cams}, grid={_grid}）")
        return
    per_cam = n_img // n_cams
    gh, gw = _grid
    if per_cam != gh * gw:
        _warn(f"token 网格不匹配（每相机 {per_cam} token，网格 {gh}×{gw}）")
        return

    def _split(arr: np.ndarray) -> np.ndarray:
        """(…, S) -> (…, n_cams, gh, gw)：跳过 prefix token，按相机切开再还原网格。"""
        return np.stack(
            [
                arr[..., n_heads_prefix + i * per_cam : n_heads_prefix + (i + 1) * per_cam].reshape(
                    *arr.shape[:-1], gh, gw
                )
                for i in range(n_cams)
            ],
            axis=-3,
        )

    per_cam_attn = _split(attn_np)  # (steps, n_cams, gh, gw)
    prefix_mass = attn_np[:, :n_heads_prefix].sum(axis=-1)  # (steps,)
    heads_cam = _split(heads_np) if heads_np is not None else None  # (H, steps, n_cams, gh, gw)
    if self_np is not None and self_np.shape[0] != self_np.shape[1]:
        _warn(f"自注意力不是方阵（{self_np.shape}），该视图跳过")
        self_np = None
    with _lock:
        if _pending_base is not None and len(_pending_base) == n_cams:
            _chunk_base = list(_pending_base)  # 与本次注意力同一帧的原始相机图
        _chunk_attn = per_cam_attn.astype(np.float32, copy=False)
        _chunk_heads = heads_cam
        _chunk_self = self_np
        _chunk_prefix = prefix_mass.astype(np.float32, copy=False)
        _n_steps = steps
        _step = 0
        _chunk_id += 1
        _stats["forward"] += 1
        _stats["attn_ms"] += elapsed_ms
        if heads_cam is not None:
            _stats["heads_mem_mb"] = heads_cam.nbytes / 1e6
        _last_chunk_ts = time.time()


def _install_hooks(policy) -> bool:
    """在 ACT 上挂 hook。返回是否成功（无图像输入的策略直接放弃）。"""
    global _installed, _cam_names
    model = getattr(policy, "model", None)
    config = getattr(policy, "config", None)
    if model is None or config is None:
        _warn("策略不是 ACT（没有 .model/.config），可视化未启用")
        return False
    image_features = list(getattr(config, "image_features", None) or [])
    if not image_features:
        _warn("策略没有图像输入，可视化未启用")
        return False
    backbone = getattr(model, "backbone", None)
    decoder = getattr(model, "decoder", None)
    if backbone is None or decoder is None:
        _warn("ACT 结构里没有 backbone/decoder，可视化未启用")
        return False

    _cam_names = [str(key).split(".")[-1] for key in image_features]
    backbone.register_forward_hook(_backbone_hook)
    for layer in decoder.layers:  # 多层时后一层会覆盖前一层，展示的是最后一层
        layer.multihead_attn.register_forward_pre_hook(_cross_attn_pre_hook, with_kwargs=True)
        if ANALYSIS:
            layer.self_attn.register_forward_pre_hook(_self_attn_pre_hook, with_kwargs=True)
    model.register_forward_hook(_act_forward_hook)
    _installed = True
    print(
        f"[act_viz] 已挂上 ACT 注意力 hook：相机={_cam_names}，"
        f"每 {EVERY_N} tick 刷一帧，head 聚合={HEADS_AGG}，alpha={ALPHA}，"
        f"分析视图={'开（per-head/步骤×空间/质量曲线/自注意力）' if ANALYSIS else '关'}",
        flush=True,
    )
    return True


def enable(policy) -> bool:
    """外部（含离线自测）可直接调用：挂 hook 并准备好状态。"""
    return _install_hooks(policy)


# ==================== 后台渲染 + Rerun ====================
_pending: tuple | None = None
_event = threading.Event()


def _has_session() -> bool:
    import rerun as rr

    ok = rr.get_global_data_recording() is not None
    if not ok:
        _warn("未检测到 Rerun 会话（--display_data=false ？），可视化跳过")
    return ok


def _log_image(entity: str, image: np.ndarray) -> bool:
    import rerun as rr

    if not _has_session():
        return False
    rr.log(entity, rr.Image(image).compress())
    return True


def _log_mass(attn: np.ndarray, prefix: np.ndarray | None, step: int) -> None:
    """③ 每 tick 记一个点：各路相机的注意力质量 + latent/state 质量 + 当前 k。

    mass(cam) = Σ_{j∈cam} A[k, j]，即该动作 query 的注意力预算有多少落在这一路相机上。
    """
    import rerun as rr

    if not _has_session():
        return
    for i, cam in enumerate(_cam_names):
        rr.log(f"{ENTITY_ROOT}/mass/{cam}", rr.Scalars(float(attn[step, i].sum())))
    if prefix is not None:
        rr.log(f"{ENTITY_ROOT}/mass/latent_state", rr.Scalars(float(prefix[step])))
    rr.log(f"{ENTITY_ROOT}/mass/step_k", rr.Scalars(float(step)))


# ==================== 分析视图渲染 ====================
def _render_heads(step: int, heads: np.ndarray | None) -> list[np.ndarray]:
    """① 每相机一张 per-head 网格（2×4，左上→右下 = head 0..7，黑线分隔）。

    每个 head 单独用它自己整段 chunk 的 1%~99.5% 分位归一化：各 head 的绝对量级差很多，
    统一标度会让弱的 head 一片蓝、看不出空间结构。
    """
    if heads is None:
        return []
    n_heads, steps, n_cams, gh, gw = heads.shape
    step = int(min(max(step, 0), steps - 1))
    cols = min(4, n_heads)
    rows = -(-n_heads // cols)
    th, tw = HEAD_TILE
    gap = HEAD_GAP
    out: list[np.ndarray] = []
    for cam in range(n_cams):
        grid = np.zeros((rows * (th + gap) - gap, cols * (tw + gap) - gap, 3), dtype=np.uint8)
        for h in range(n_heads):
            all_h = heads[h, :, cam]  # (steps, gh, gw)
            lo, hi = _p_lo_hi(all_h)
            tile = _jet(_upsample01(_norm01(all_h[step], lo, hi), th, tw))
            r, c = divmod(h, cols)
            y, x = r * (th + gap), c * (tw + gap)
            grid[y : y + th, x : x + tw] = tile
        out.append(grid)
    return out


def _render_timeline(attn: np.ndarray | None) -> list[np.ndarray]:
    """② 「步骤×空间」图：横轴 = chunk 内第 k 步，纵轴 = token（15×20 展平，灰线分隔 feature 行）。

    整段 chunk 的注意力一次画完（前向结束就全都有了），所以每个 chunk 只需渲染一张。
    """
    if attn is None:
        return []
    steps, n_cams, gh, gw = attn.shape
    s = TIMELINE_SCALE
    out: list[np.ndarray] = []
    for cam in range(n_cams):
        all_cam = attn[:, cam]  # (steps, gh, gw)
        lo, hi = _p_lo_hi(all_cam)
        m = _norm01(all_cam.reshape(steps, gh * gw).T, lo, hi)  # (tokens, steps)
        img = _jet(_upsample01(m, gh * gw * s, steps * s))
        for row in range(1, gh):
            img[row * gw * s : row * gw * s + 1] = 90  # feature-map 的行边界
        out.append(img)
    return out


def _render_selfattn(self_attn: np.ndarray | None) -> np.ndarray | None:
    """④ decoder 自注意力矩阵（行 = query 步，列 = key 步；无 mask，双向）。"""
    if self_attn is None:
        return None
    n = self_attn.shape[0]
    lo, hi = _p_lo_hi(self_attn)
    return _jet(_upsample01(_norm01(self_attn, lo, hi), n * SELFATTN_SCALE, n * SELFATTN_SCALE))


def _send_blueprint() -> None:
    """补发含 act_viz 视图的 blueprint（lerobot 的默认 blueprint 不含这些实体）。"""
    global _blueprint_sent
    import rerun as rr
    import rerun.blueprint as rrb

    if rr.get_global_data_recording() is None:
        return
    views = [
        rrb.Spatial2DView(origin=f"observation.images.{cam}", name=f"cam/{cam}") for cam in _cam_names
    ]
    views += [
        rrb.Spatial2DView(origin=f"{ENTITY_ROOT}/attn/{cam}", name=f"act_attn/{cam}") for cam in _cam_names
    ]
    if ANALYSIS:
        views += [
            rrb.Spatial2DView(origin=f"{ENTITY_ROOT}/attn_heads/{cam}", name=f"heads/{cam}")
            for cam in _cam_names
        ]
        views += [
            rrb.Spatial2DView(origin=f"{ENTITY_ROOT}/timeline/{cam}", name=f"step×space/{cam}")
            for cam in _cam_names
        ]
        views.append(rrb.Spatial2DView(origin=f"{ENTITY_ROOT}/selfattn", name="decoder self-attn"))
    views.append(rrb.TimeSeriesView(name="observation", contents=["observation/**"]))
    if ANALYSIS:
        views.append(rrb.TimeSeriesView(name="注意力质量", contents=[f"{ENTITY_ROOT}/mass/**"]))
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views)))
    _blueprint_sent = True
    print(
        f"[act_viz] 已补发 Rerun blueprint（{len(views)} 个视图：相机原图 + 注意力叠图"
        + (" + 分析视图 + 质量曲线）" if ANALYSIS else "）"),
        flush=True,
    )


def _render_and_log(base_imgs: list[np.ndarray], step: int, chunk_id: int) -> None:
    """主叠图（每 tick）+ 分析视图（低帧率/每 chunk），全部在后台线程里做。"""
    global _first_log_ts, _rendered_chunk_id
    with _lock:
        attn = _chunk_attn
        heads = _chunk_heads
        self_attn = _chunk_self
        prefix = _chunk_prefix
    if attn is None:
        return
    step = int(min(max(step, 0), attn.shape[0] - 1))
    step_attn = attn[step]  # (n_cams, gh, gw)

    overlays: list[np.ndarray] = []
    for i, cam in enumerate(_cam_names):
        if i >= len(base_imgs):
            continue
        base = base_imgs[i]
        heat = step_attn[i]
        # 归一化用整段 chunk 的 1%~99.5% 分位，保证不同 step 之间颜色可比
        cam_all = attn[:, i]
        lo, hi = _p_lo_hi(cam_all)
        t = _upsample01(_norm01(heat, lo, hi), base.shape[0], base.shape[1])
        rgb = _jet(t)
        # 叠加强度随注意力强弱渐变：弱处几乎保留原图，强处才上色，便于直接读出分布
        alpha = (ALPHA * t)[:, :, None]
        overlay = (
            base.astype(np.float32) * (1.0 - alpha) + rgb.astype(np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)
        overlays.append(overlay)
        if _log_image(f"{ENTITY_ROOT}/attn/{cam}", overlay):
            _stats["logged"] += 1
            if _first_log_ts == 0.0:
                _first_log_ts = time.time()
    _write_mp4(overlays)  # 与 Rerun 是否可用无关，离线 MP4 照常写
    _stats["render"] += 1

    if not ANALYSIS:
        return

    try:
        _log_mass(attn, prefix, step)  # ③ 每 tick 一个点
    except Exception as exc:
        _warn(f"记录注意力质量失败：{exc!r}")

    # ① per-head 网格：低帧率刷新（各 head 绝对量级不同，是"对比"用的视图）
    if _tick % HEADS_EVERY == 0:
        try:
            tiles = _render_heads(step, heads)
            for cam, img in zip(_cam_names, tiles):
                if _log_image(f"{ENTITY_ROOT}/attn_heads/{cam}", img):
                    _stats["logged"] += 1
            if len(tiles) == len(_cam_names) and len({t.shape for t in tiles}) == 1:
                _mp4_put("act_analysis_heads", _mp4_fps_per(HEADS_EVERY), np.concatenate(tiles, axis=0))
            _stats["analysis"] += 1
        except Exception as exc:
            _warn(f"渲染 per-head 网格失败：{exc!r}")

    # ② 步骤×空间图 + ④ 自注意力：每个 chunk 一张（前向一结束整段 chunk 的注意力就都齐了）
    if chunk_id != _rendered_chunk_id:
        _rendered_chunk_id = chunk_id
        chunk_ticks = max(int(attn.shape[0]), 1)  # 一个 chunk 覆盖多少个控制 tick
        try:
            strips = _render_timeline(attn)
            for cam, img in zip(_cam_names, strips):
                if _log_image(f"{ENTITY_ROOT}/timeline/{cam}", img):
                    _stats["logged"] += 1
            if len(strips) == len(_cam_names) and len({s.shape for s in strips}) == 1:
                _mp4_put(
                    "act_analysis_timeline", _mp4_fps_per(chunk_ticks), np.concatenate(strips, axis=1)
                )
            matrix = _render_selfattn(self_attn)
            if matrix is not None:
                if _log_image(f"{ENTITY_ROOT}/selfattn", matrix):
                    _stats["logged"] += 1
                _mp4_put("act_analysis_selfattn", _mp4_fps_per(chunk_ticks), matrix)
            _stats["analysis"] += 1
        except Exception as exc:
            _warn(f"渲染步骤×空间/自注意力失败：{exc!r}")


def _worker() -> None:
    """后台线程：只消费最新一帧，渲染 + 打日志，绝不阻塞控制循环。

    整个循环体兜底 try/except：任何异常都只提示一次并继续下一帧，绝不让线程静默退出
    （线程一死就再也画不出东西，且不会影响推理，属于最难发现的失败模式）。
    """
    global _pending, _blueprint_sent
    while True:
        # 0.5s 超时轮询：既等新帧，也让 blueprint 的"首帧后 2s"条件在没有新帧时也能被检查
        got = _event.wait(0.5)
        if got:
            with _lock:
                item, _pending = _pending, None
                _event.clear()
        else:
            item = None
        try:
            if item is not None:
                _render_and_log(*item)
            if (
                SEND_BLUEPRINT
                and not _blueprint_sent
                and _first_log_ts > 0.0
                and time.time() - _first_log_ts > 2.0  # 等 lerobot 自己的 blueprint 先发完
            ):
                _send_blueprint()
        except Exception as exc:  # 渲染/写 Rerun 失败不能影响推理
            _warn(f"可视化线程异常（已忽略，继续下一帧）：{exc!r}")


def _ensure_worker() -> None:
    if getattr(_ensure_worker, "started", False):
        return
    _ensure_worker.started = True
    threading.Thread(target=_worker, name="act_viz", daemon=True).start()


def _submit(step: int) -> None:
    global _pending
    with _lock:
        if _chunk_attn is None or not _chunk_base:
            return
        _pending = (list(_chunk_base), int(step), _chunk_id)
        _stats["submit"] += 1
    _event.set()


# ==================== 打补丁 ====================
def _stash_base_images(obs_frame: dict) -> None:
    """在 forward 之前把原始（未归一化）相机图暂存，供本次 chunk 的叠图当底图。"""
    global _pending_base
    if not _cam_names:
        return
    imgs = []
    for cam in _cam_names:
        raw = None
        for key in (f"observation.images.{cam}", cam):
            if key in obs_frame:
                raw = obs_frame[key]
                break
        img = _to_u8_hwc(raw)
        if img is None:
            break
        imgs.append(img)
    if len(imgs) == len(_cam_names):
        with _lock:
            _pending_base = imgs


def _patch_engine() -> None:
    """包装 SyncInferenceEngine.get_action：暂存底图 + 按 tick 滚动提交可视化帧。"""
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    original = SyncInferenceEngine.get_action

    def get_action(self, obs_frame):
        global _tick, _step
        if not ACTIVE or obs_frame is None:
            return original(self, obs_frame)
        if not _installed:
            try:
                _install_hooks(self._policy)
            except Exception as exc:
                _warn(f"挂 hook 失败：{exc}")
        try:
            _stash_base_images(obs_frame)
        except Exception as exc:
            _warn(f"暂存相机底图失败：{exc}")

        action = original(self, obs_frame)  # 内部可能触发 ACT 前向（≈每 100 tick 一次）

        try:
            _tick += 1
            if _chunk_attn is not None and _tick % EVERY_N == 0:
                _submit(_step)
            _step += 1
        except Exception as exc:
            _warn(f"提交可视化帧失败：{exc}")
        return action

    SyncInferenceEngine.get_action = get_action


def _print_stats() -> None:
    if not _installed:
        return
    n = max(_stats["forward"], 1)
    dropped = max(_stats["submit"] - _stats["render"], 0)
    dropped_pct = 100.0 * dropped / _stats["submit"] if _stats["submit"] else 0.0
    print(
        f"[act_viz] 结束：ACT 前向 {_stats['forward']} 次，"
        f"单次注意力计算均值 {_stats['attn_ms'] / n:.1f} ms；"
        f"提交 {_stats['submit']} 帧，渲染 {_stats['render']} 帧"
        f"（丢帧 {dropped}，{dropped_pct:.1f}%），写入 Rerun {_stats['logged']} 张；"
        f"分析视图渲染 {_stats['analysis']} 次"
        + (f"（逐 head 缓存 {_stats.get('heads_mem_mb', 0.0):.1f} MB/次前向）" if ANALYSIS else ""),
        flush=True,
    )
    if MP4_ENABLED and _mp4_paths:
        print("[act_viz] MP4 已保存：", flush=True)
        for path in _mp4_paths:
            key = os.path.splitext(os.path.basename(path))[0]
            print(f"    {path}（{_mp4_count.get(key, 0)} 帧）", flush=True)
        if not any(not os.path.basename(p).startswith("act_analysis") for p in _mp4_paths):
            print("    （只有分析视频：主叠图还没刷出帧，多半运行时间不足一个 chunk）", flush=True)
    elif MP4_ENABLED:
        print("[act_viz] MP4 未生成（没有渲染出任何帧）", flush=True)


def main() -> None:
    _ensure_worker()

    # 需要动作追踪时，顺带装上 action_trace.py 的补丁（同一个入口，参数完全一致）
    trace = None
    if os.environ.get("ACTION_TRACE") == "1":
        try:
            import action_trace as trace

            trace._patch_sync_engine()
            trace._patch_robot_send()
        except Exception as exc:
            _warn(f"装载 action_trace 失败：{exc}")
            trace = None

    _patch_engine()

    from lerobot.scripts.lerobot_rollout import main as rollout_main

    try:
        rollout_main()
    finally:
        globals()["ACTIVE"] = False  # 停止提交新帧，给后台线程一点时间把最后一帧写完
        time.sleep(0.3)
        _close_mp4()
        if trace is not None:
            try:
                trace._write_summary()
            except Exception:
                pass
        _print_stats()


if __name__ == "__main__":
    main()

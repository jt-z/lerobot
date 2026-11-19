#!/usr/bin/env python3
"""三路相机（hand/front/top）实时画面 vs 数据集初始零位画面对照工具。

用途：推理/采集前调整相机位姿。左边是相机实时画面，右边是数据集同一集开头的
零位帧，叠加（blend）后直接看“相机有没有摆回采集时的位置”。独立小工具，不依赖
lerobot 包，也不需要 CAN/机械臂，插上三路相机就能跑。

用法（在仓库根目录）:
  # 带窗口实时对照：必须用带 GUI 的解释器。conda 里的 python 是 headless 版，
  # 在 conda 环境里也要写全路径，否则 python3 仍指向 conda 的 python。
  /usr/bin/python3 self_scripts/b601_single/inference/lerobot/cam_align_3cam.py
      # 自动挑最新的、带视频的数据集（优先 $B601_DATA_ROOT，其次仓库根目录）

  python self_scripts/b601_single/inference/lerobot/cam_align_3cam.py \
      --dataset /home/jt/dev/lerobot/b601_americano_20261117_212428
  python ... --episode 1 --offset-frames 30     # 第 2 集、往后 30 帧（1 秒）的零位帧
  python ... --scale 0.8                        # 每格放大到 0.8，窗口更大
  python ... --width 1280 --height 720          # 用更高分辨率预览（对照图会缩放到同尺寸）
  python ... --snapshot                         # 不开窗口，抓一帧存对照图后退出
                                                # （conda 里的 OpenCV 常是 headless，用这个）

窗口布局（一列一路相机）:
  第 1 行 LIVE  相机实时画面（带九宫格 + 中心十字）
  第 2 行 REF   数据集零位帧（同样的参考线）
  第 3 行       按 m 在三种模式间切换：
                  BLEND 两路按 alpha 叠加
                  DIFF  absdiff 热力图
                  EDGE  红绿边缘对比（红=实时边缘，绿=参考边缘，黄=重合，对齐时轮廓变黄）

按键:
  q / ESC  退出          s  保存当前三路 live/ref/blend/diff/edge 到 _logs/cam_align/<时间戳>/
  g        参考线开关      m  第 3 行在 blend / diff / edge 之间循环
  a / A    blend alpha -/+ 0.1
  , / .    参考帧偏移 -/+ 1 帧（数据集里往后挪，避开自动曝光抖动）
  n / p    下一集 / 上一集
  r        重新载入参考帧
  h        打印按键帮助

参考帧来源（按数据集实际布局自动适配）:
  v3.0 视频：videos/{video_key}/chunk-{i:03d}/file-{j:03d}.mp4 + meta/episodes 的
             from_timestamp（同一 mp4 里按时间戳定位每一集的起点）
  v2.x 视频：videos/chunk-{i:03d}/{video_key}/episode_{ep:06d}.mp4
  图片数据集：images/{image_key}/episode-{ep:06d}/frame-{frame:06d}.png
  相机角色 → 特征键：hand/front/top → observation.images.hand/front/top

相机序列号取自 b601_common/devices/<host>.sh（与采集/推理同一份档案），
可用环境变量 B601_CAM_HAND / B601_CAM_FRONT / B601_CAM_TOP 覆盖。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROLES = ("hand", "front", "top")
DEFAULT_SAVE = os.path.expanduser("~/dev/lerobot/self_scripts/_logs/cam_align")
GREEN, YELLOW, RED, WHITE = (0, 220, 0), (0, 220, 220), (0, 0, 255), (245, 245, 245)


# ==================== 相机序列号（复用设备档案） ====================
def profile_serials(dev_dir: Path) -> dict[str, str]:
    """从 b601_common/devices/{_defaults,<host>}.sh 读 hand/front/top 序列号"""
    host = os.environ.get("B601_DEVICE") or os.uname().nodename
    host = {"mineru-box": "kfzbox", "mineru_box": "kfzbox",
            "kf-z790": "z790", "kf790": "z790"}.get(host, host)
    vals: dict[str, str] = {}
    for name in ("_defaults.sh", f"{host}.sh"):
        p = dev_dir / name
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        for role in ROLES:
            m = re.search(rf"^\s*export\s+B601_CAM_{role.upper()}=.*$", txt, re.M)
            if m:
                nums = re.findall(r"\d{4,}", m.group(0))
                if nums:
                    vals[role] = nums[0]
    for role in ROLES:
        env = os.environ.get(f"B601_CAM_{role.upper()}", "")
        if env:
            vals[role] = env
    return vals


def resolve_cams(dev_dir: Path) -> dict[str, str | None]:
    """角色 → /dev/v4l/by-id 路径；档案里没写的角色按剩余设备顺序兜底"""
    devs = sorted(glob.glob("/dev/v4l/by-id/*video-index0"))
    serials = profile_serials(dev_dir)
    out: dict[str, str | None] = {}
    used: set[str] = set()
    for role in ROLES:
        sn = serials.get(role, "")
        hit = None
        if sn:
            hits = sorted(glob.glob(f"/dev/v4l/by-id/*{sn}*video-index0"))
            hit = hits[0] if hits else None
        out[role] = hit
        if hit:
            used.add(hit)
    rest = [d for d in devs if d not in used]
    for role in ROLES:
        if out[role] is None and rest:
            out[role] = rest.pop(0)
            print(f"⚠️  {role} 没有档案序列号，按顺序兜底到 {os.path.basename(out[role])}")
    for role in ROLES:
        if out[role] is None:
            sn = serials.get(role, "")
            print(f"❌ 找不到 {role} 相机" + (f"（序列号 {sn}）" if sn else ""))
    return out


# ==================== 数据集定位与参考帧 ====================
def is_dataset(p: Path) -> bool:
    info = p / "meta" / "info.json"
    if not info.is_file():
        return False
    try:
        meta = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("total_episodes", 0) <= 0:
        return False
    return any((p / d).is_dir() for d in ("videos", "images", "data"))


def find_latest_dataset(roots: list[Path]) -> Path | None:
    hits: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir() and is_dataset(p):
                hits.append(p)
    if not hits:
        return None
    return max(hits, key=lambda p: (p / "meta" / "info.json").stat().st_mtime)


def resolve_dataset(arg: str, repo_root: Path) -> Path:
    roots = [Path(os.environ.get("B601_DATA_ROOT", "")).expanduser(),
             Path("~/b601_data").expanduser(),
             repo_root, Path.cwd().resolve()]
    if arg:
        p = Path(arg).expanduser()
        if p.is_dir():
            # 允许直接贴 .../videos 或子目录，向上找含 meta/info.json 的那层
            for cand in (p, p.parent, p.parent.parent):
                if (cand / "meta" / "info.json").is_file():
                    return cand.resolve()
            return p.resolve()
        for root in roots:
            cand = root / arg
            if cand.is_dir():
                return cand.resolve()
        raise SystemExit(f"❌ 数据集不存在：{arg}")
    p = find_latest_dataset(roots)
    if p is None:
        raise SystemExit("❌ 自动找不到数据集，请用 --dataset 指定（需含 meta/info.json 且 total_episodes>0）")
    return p


def read_episodes(root: Path):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet")))
    if not files and (root / "meta" / "episodes.parquet").is_file():
        files = [str(root / "meta" / "episodes.parquet")]
    if not files:
        return None
    return pq.read_table(files).to_pandas()


def _read_video_frame(path: Path, timestamp: float) -> np.ndarray | None:
    """按时间戳取一帧；h264 seek 后前几帧可能是空/花屏，多读几帧取最后的非空帧"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    frame = None
    for _ in range(8):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    return frame


def load_ref_frames(root: Path, episode: int, offset_frames: int) -> dict[str, np.ndarray]:
    """取一集开头（+ offset_frames）的三路参考帧"""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30))
    video_path_tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    image_path_tpl = info.get("image_path", "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png")
    eps = read_episodes(root)
    row = None
    if eps is not None and len(eps):
        match = eps[eps["episode_index"] == episode]
        if len(match):
            row = match.iloc[0]

    refs: dict[str, np.ndarray] = {}
    features = info.get("features", {})
    for role in ROLES:
        key = f"observation.images.{role}"
        if key not in features:
            continue
        feat = features[key]
        frame = None
        if feat.get("dtype") == "video" and row is not None:
            chunk = int(row.get(f"videos/{key}/chunk_index", 0))
            file_idx = int(row.get(f"videos/{key}/file_index", 0))
            ts = float(row.get(f"videos/{key}/from_timestamp", 0.0)) + offset_frames / fps
            path = root / video_path_tpl.format(video_key=key, chunk_index=chunk, file_index=file_idx)
            if path.is_file():
                frame = _read_video_frame(path, ts)
            if frame is None:
                # 老布局：videos/chunk-000/<key>/episode_000000.mp4
                legacy = root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode:06d}.mp4"
                if legacy.is_file():
                    frame = _read_video_frame(legacy, offset_frames / fps)
        elif feat.get("dtype") == "image":
            path = root / image_path_tpl.format(image_key=key, episode_index=episode,
                                                frame_index=max(0, offset_frames))
            if path.is_file():
                frame = cv2.imread(str(path))
        if frame is not None:
            refs[role] = frame
        else:
            print(f"⚠️  数据集里取不到 {role} 的参考帧（{key}）")
    return refs


# ==================== 相机采集线程 ====================
class CamReader:
    """每路一个读线程，主循环只取最新帧，避免三路串行 read 互相拖慢"""

    def __init__(self, path: str, width: int, height: int, fps: int, fourcc: str):
        self.path = path
        self.ok = False
        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)   # 设 1 会丢一半帧（见 00_dual_cam.py 注释）
        self.ok = True
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.count = 0
        self.failed = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok and f is not None:
                with self._lock:
                    self._frame = f
                    self.count += 1
            else:
                self.failed += 1
                time.sleep(0.005)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        time.sleep(0.05)
        self.cap.release()


# ==================== 画面拼装 ====================
def draw_grid(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    for i in (1, 2):
        cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.line(img, (0, h * i // 3), (w, h * i // 3), (60, 60, 60), 1, cv2.LINE_AA)
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 18, cy), (cx + 18, cy), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 18), (cx, cy + 18), (0, 255, 255), 1, cv2.LINE_AA)


def put(img: np.ndarray, text: str, color=GREEN, org=(8, 22), scale=0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = np.full((h, w, 3), 32, np.uint8)
    put(img, text, RED, (8, h // 2), 0.55)
    return img


def mae(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.abs(ga.astype(np.int16) - gb.astype(np.int16)).mean())


def edge_overlay(live: np.ndarray, ref: np.ndarray, t1: int, t2: int) -> np.ndarray:
    """红绿边缘对比：红=实时画面边缘，绿=数据集零位帧边缘，黄=两边重合（对齐时轮廓变黄）"""
    gl = cv2.GaussianBlur(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    gr = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    el = cv2.Canny(gl, t1, t2) > 0
    er = cv2.Canny(gr, t1, t2) > 0
    vis = np.zeros((*el.shape, 3), np.uint8)
    vis[el] = (0, 0, 255)          # 实时 → 红
    vis[er] = (0, 255, 0)          # 参考 → 绿
    vis[el & er] = (0, 255, 255)   # 重合 → 黄
    return vis


def score_color(v: float) -> tuple[int, int, int]:
    return GREEN if v < 25 else (YELLOW if v < 45 else RED)


def save_snapshot(outdir: Path, live: dict[str, np.ndarray], ref: dict[str, np.ndarray],
                  alpha: float, canvas: np.ndarray, canny: tuple[int, int]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        lf, rf = live.get(role), ref.get(role)
        if lf is not None:
            cv2.imwrite(str(outdir / f"{role}_live.png"), lf)
        if rf is not None:
            cv2.imwrite(str(outdir / f"{role}_ref.png"), rf)
        if lf is not None and rf is not None:
            h, w = lf.shape[:2]
            r2 = cv2.resize(rf, (w, h), interpolation=cv2.INTER_AREA)
            blend = cv2.addWeighted(lf, alpha, r2, 1.0 - alpha, 0)
            diff = cv2.absdiff(cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY))
            heat = cv2.applyColorMap(np.clip(diff.astype(np.int16) * 3, 0, 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
            cv2.imwrite(str(outdir / f"{role}_blend.png"), blend)
            cv2.imwrite(str(outdir / f"{role}_diff.png"), heat)
            cv2.imwrite(str(outdir / f"{role}_edge.png"), edge_overlay(lf, r2, *canny))
    cv2.imwrite(str(outdir / "montage.png"), canvas)
    print(f"📷 已保存对照图：{outdir}")


def build_canvas(live: dict[str, np.ndarray], refs: dict[str, np.ndarray], episode: int,
                 n_eps: int, offset_frames: int, width: int, height: int, scale: float,
                 grid: bool, alpha: float, mode: str, canny: tuple[int, int],
                 fps_txt: str) -> tuple[np.ndarray, dict[str, float]]:
    """3 列（hand/front/top）× 3 行（LIVE / REF / 第三行按 mode）；返回画布与各路 MAE"""
    cw = max(160, int(round(width * scale)))
    ch = max(120, int(round(height * scale)))
    head_h = 34
    canvas = np.full((head_h + ch * 3, cw * 3, 3), 18, np.uint8)
    put(canvas, f"ep {episode}/{n_eps - 1}  off {offset_frames:+d}f  alpha {alpha:.2f}  "
                f"live {width}x{height}  fps {fps_txt}", WHITE, (8, 24), 0.55)
    metrics: dict[str, float] = {}
    for col, role in enumerate(ROLES):
        x0 = col * cw
        lf, rf = live.get(role), refs.get(role)

        if lf is not None:
            cell = cv2.resize(lf, (cw, ch), interpolation=cv2.INTER_AREA)
            if grid:
                draw_grid(cell)
            put(cell, f"{role} LIVE", GREEN)
        else:
            cell = placeholder(cw, ch, f"{role} LIVE: no frame")
        canvas[head_h:head_h + ch, x0:x0 + cw] = cell

        if rf is not None:
            cell = cv2.resize(rf, (cw, ch), interpolation=cv2.INTER_AREA)
            if grid:
                draw_grid(cell)
            put(cell, f"{role} REF ep{episode} +{offset_frames}f", (255, 180, 0))
        else:
            cell = placeholder(cw, ch, f"{role} REF: missing")
        canvas[head_h + ch:head_h + 2 * ch, x0:x0 + cw] = cell

        if lf is not None and rf is not None:
            rs = cv2.resize(rf, (cw, ch), interpolation=cv2.INTER_AREA)
            ls = cv2.resize(lf, (cw, ch), interpolation=cv2.INTER_AREA)
            m = mae(ls, rs)
            metrics[role] = m
            if mode == "blend":
                cell = cv2.addWeighted(ls, alpha, rs, 1.0 - alpha, 0)
                put(cell, f"BLEND MAE {m:.1f}", score_color(m))
            elif mode == "diff":
                d = cv2.absdiff(cv2.cvtColor(ls, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY))
                cell = cv2.applyColorMap(np.clip(d.astype(np.int16) * 3, 0, 255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO)
                put(cell, f"DIFF MAE {m:.1f}", score_color(m))
            else:
                cell = edge_overlay(ls, rs, *canny)
                put(cell, f"EDGE red=live green=ref  MAE {m:.1f}", score_color(m))
        else:
            cell = placeholder(cw, ch, "no live/ref")
        canvas[head_h + 2 * ch:head_h + 3 * ch, x0:x0 + cw] = cell
    return canvas, metrics


def main() -> int:
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    dev_dir = repo_root / "self_scripts" / "b601_common" / "devices"

    ap = argparse.ArgumentParser(description="三路相机 vs 数据集零位帧实时对照")
    ap.add_argument("--dataset", default="", help="数据集目录（默认自动挑最新的）")
    ap.add_argument("--episode", type=int, default=0, help="参照的集号（默认 0）")
    ap.add_argument("--offset-frames", type=int, default=0, help="零位帧再往后挪几帧（默认 0）")
    ap.add_argument("--blend", type=float, default=0.5, help="叠加时实时画面权重（默认 0.5）")
    ap.add_argument("--width", type=int, default=640, help="预览分辨率宽（默认 640，与数据集一致）")
    ap.add_argument("--height", type=int, default=480, help="预览分辨率高（默认 480）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--scale", type=float, default=0.62, help="窗口内每格的缩放（默认 0.62）")
    ap.add_argument("--canny", default="50,150", help="红绿边缘对比的 Canny 双阈值 t1,t2（默认 50,150）")
    ap.add_argument("--snapshot", action="store_true",
                    help="不开窗口，抓一帧存对照图后退出（无 GUI 的 conda 环境用这个）")
    ap.add_argument("--no-grid", action="store_true", help="不画九宫格/中心十字")
    ap.add_argument("--save-dir", default=DEFAULT_SAVE, help="快照根目录")
    ap.add_argument("--cameras", default="", help="手工指定角色设备，如 hand=/dev/video0,front=...,top=...")
    a = ap.parse_args()

    try:
        canny = tuple(int(x) for x in a.canny.split(","))
        assert len(canny) == 2
    except (ValueError, AssertionError):
        raise SystemExit(f"❌ --canny 写法不对：{a.canny}（应形如 50,150）")

    root = resolve_dataset(a.dataset, repo_root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    n_eps = int(info.get("total_episodes", 1))
    episode = max(0, min(a.episode, n_eps - 1))
    if episode != a.episode:
        print(f"⚠️  --episode {a.episode} 越界，改用 {episode}（共 {n_eps} 集）")
    print(f"数据集：{root}")
    print(f"         {info.get('codebase_version')} fps={info.get('fps')} "
          f"集数={n_eps} 帧数={info.get('total_frames')}")

    if a.cameras:
        cams: dict[str, str | None] = {r: None for r in ROLES}
        for item in a.cameras.split(","):
            if "=" in item:
                role, dev = item.split("=", 1)
                if role.strip() in cams:
                    cams[role.strip()] = dev.strip()
    else:
        cams = resolve_cams(dev_dir)

    refs = load_ref_frames(root, episode, a.offset_frames)
    if not refs:
        print("❌ 数据集一路参考帧都没有，检查 --dataset / --episode")
        return 1

    readers: dict[str, CamReader] = {}
    for role in ROLES:
        dev = cams.get(role)
        if not dev:
            continue
        rd = CamReader(dev, a.width, a.height, a.fps, a.fourcc)
        if not rd.ok:
            print(f"❌ {role:5s} 打不开：{dev}")
            continue
        readers[role] = rd
    if not readers:
        print("❌ 一路相机都没打开，先确认 /dev/v4l/by-id 与 USB 带宽")
        return 2
    for role, rd in readers.items():
        f = None
        for _ in range(30):
            f = rd.latest()
            if f is not None:
                break
            time.sleep(0.05)
        if f is not None:
            h, w = f.shape[:2]
            print(f"✅ {role:5s} {os.path.basename(rd.path)} 实际 {w}x{h}")
        else:
            print(f"⚠️  {role:5s} 打开成功但还没有帧（USB 带宽/线材？）")

    grid = not a.no_grid
    mode = "blend"
    modes = ("blend", "diff", "edge")
    alpha = float(np.clip(a.blend, 0.0, 1.0))
    save_root = Path(a.save_dir).expanduser()

    # ---- 无窗口模式：抓一帧存对照图（含红绿边缘）就退出 ----
    if a.snapshot:
        live: dict[str, np.ndarray] = {}
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            for role, rd in readers.items():
                f = rd.latest()
                if f is not None:
                    live[role] = f
            if len(live) == len(readers):
                break
            time.sleep(0.05)
        canvas, metrics = build_canvas(live, refs, episode, n_eps, a.offset_frames,
                                       a.width, a.height, a.scale, grid, alpha, mode,
                                       canny, "snapshot")
        if metrics:
            print("  " + "  ".join(f"{r} MAE={v:5.1f}" for r, v in metrics.items()))
        outdir = save_root / time.strftime("%Y%m%d_%H%M%S")
        save_snapshot(outdir, live, refs, alpha, canvas, canny)
        edge_canvas, _ = build_canvas(live, refs, episode, n_eps, a.offset_frames,
                                      a.width, a.height, a.scale, grid, alpha, "edge",
                                      canny, "snapshot")
        cv2.imwrite(str(outdir / "montage_edge.png"), edge_canvas)
        for rd in readers.values():
            rd.close()
        print("   红=实时边缘 绿=参考边缘 黄=重合；拼图见 montage_edge.png，单路见 *_edge.png")
        return 0

    win = "B601 3cam align (q=quit s=save g=grid m=mode)"

    last_t = time.perf_counter()
    last_counts = {r: rd.count for r, rd in readers.items()}
    last_print = 0.0

    print("\n按键: q 退出 | s 存图 | g 参考线 | m blend/diff/edge | a/A 透明度 | "
          ",/. 参考帧偏移 | n/p 切集 | r 重载")
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            live: dict[str, np.ndarray] = {}
            for role, rd in readers.items():
                f = rd.latest()
                if f is not None:
                    live[role] = f

            now = time.perf_counter()
            dt = now - last_t
            fps_txt = []
            for role in ROLES:
                rd = readers.get(role)
                if rd is None:
                    continue
                fps_txt.append(f"{role[0]}:{(rd.count - last_counts.get(role, 0)) / dt:.0f}")
                last_counts[role] = rd.count
            last_t = now

            canvas, metrics = build_canvas(live, refs, episode, n_eps, a.offset_frames,
                                           a.width, a.height, a.scale, grid, alpha, mode,
                                           canny, " ".join(fps_txt))
            cv2.imshow(win, canvas)

            if metrics and now - last_print > 1.0:
                last_print = now
                print("  " + "  ".join(f"{r} MAE={v:5.1f}" for r, v in metrics.items())
                      + f"  (off {a.offset_frames:+d}f, ep {episode}, {mode})")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_snapshot(save_root / time.strftime("%Y%m%d_%H%M%S"),
                              live, refs, alpha, canvas, canny)
            elif key == ord("g"):
                grid = not grid
            elif key == ord("m"):
                mode = modes[(modes.index(mode) + 1) % len(modes)]
            elif key == ord("a"):
                alpha = float(np.clip(alpha - 0.1, 0.0, 1.0))
            elif key == ord("A"):
                alpha = float(np.clip(alpha + 0.1, 0.0, 1.0))
            elif key in (ord(","), ord(".")):
                a.offset_frames += -1 if key == ord(",") else 1
                refs = load_ref_frames(root, episode, a.offset_frames)
            elif key in (ord("n"), ord("p")):
                episode = (episode + (1 if key == ord("n") else -1)) % n_eps
                a.offset_frames = 0
                refs = load_ref_frames(root, episode, a.offset_frames)
            elif key == ord("r"):
                refs = load_ref_frames(root, episode, a.offset_frames)
            elif key == ord("h"):
                print("q 退出 | s 存图 | g 参考线 | m blend/diff/edge | a/A 透明度 | "
                      ",/. 参考帧偏移 | n/p 切集 | r 重载")
    except KeyboardInterrupt:
        pass
    except cv2.error as e:
        msg = str(e).splitlines()[-1][:160]
        print(f"❌ OpenCV 窗口不可用：{msg}")
        print(f"   当前解释器：{sys.executable}（这个 python 的 OpenCV 是 headless 版）")
        print("   在 conda 环境里直接写 python3 仍然是 conda 的 python，要写系统解释器全路径：")
        print("     /usr/bin/python3 self_scripts/b601_single/inference/lerobot/cam_align_3cam.py")
        print("   或者 conda deactivate 之后再跑；不想开窗口就加 --snapshot 存图。")
        return 3
    finally:
        for rd in readers.values():
            rd.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"\n完成。参考帧：{root} 第 {episode} 集 +{a.offset_frames} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

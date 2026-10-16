"""Post one F1TENTH policy release or result to the Discord webhook.

The follow video is rendered by ``training/render_follow.py`` (the script that
produced overlay_follow.mp4). This module records the trajectory ``.npz`` that
script reads, pulls two W&B charts, and uploads one multipart message. Every
post includes the evaluated checkpoint: attached when it fits Discord's
documented per-file webhook limit, otherwise uploaded to the existing S3
bucket and linked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Discord API default per-file upload limit is 20 MiB.
# https://docs.discord.com/developers/reference#uploading-files
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
S3_BUCKET = "storage-o9g473recrgmg1p06"
S3_ENDPOINT_URL = "https://t3.storageapi.dev"
S3_REGION = "auto"
S3_PREFIX = "rl/model-release"
KIND_TITLES = {
    "release": "F1TENTH policy release",
    "result": "F1TENTH policy result",
    "loss": "F1TENTH policy loss",
}
CARS = 8
RENDER_SCRIPT = Path(__file__).resolve().parent / "render_follow.py"
RELEASE_ENV = Path.home() / ".config/f1tenth/model-release.env"
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
THROUGHPUT_RE = re.compile(r"transitions/s=([0-9]+(?:\.[0-9]+)?)")
WANDB_URL_RE = re.compile(
    r"https?://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)"
)


def load_release_env(path: Path = RELEASE_ENV) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def s3_bucket() -> str:
    return os.environ.get("S3_BUCKET", S3_BUCKET).strip() or S3_BUCKET


def s3_endpoint() -> str:
    return (
        os.environ.get("S3_ENDPOINT_URL", S3_ENDPOINT_URL).strip() or S3_ENDPOINT_URL
    ).rstrip("/")


def s3_checkpoint_key(filename: str, run_id: str | None = None) -> str:
    prefix = S3_PREFIX.strip("/")
    if run_id:
        return f"{prefix}/{run_id}/{filename}"
    return f"{prefix}/{filename}"


def s3_object_url(key: str) -> str:
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in key.split("/"))
    return f"{s3_endpoint()}/{s3_bucket()}/{encoded}"


def planned_s3_url(_path: Path, key: str) -> str:
    return s3_object_url(key)


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def put_s3_object(path: Path, key: str) -> str:
    access = os.environ.get("S3_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("S3_SECRET_ACCESS_KEY", "").strip()
    region = os.environ.get("S3_REGION", S3_REGION).strip() or S3_REGION
    if not access or not secret:
        raise SystemExit("S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY are required")
    body = path.read_bytes()
    payload_hash = hashlib.sha256(body).hexdigest()
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    host = urllib.parse.urlparse(s3_endpoint()).netloc
    canonical_uri = "/" + "/".join(
        urllib.parse.quote(part, safe="") for part in [s3_bucket(), *key.split("/")]
    )
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        [
            "PUT",
            canonical_uri,
            "",
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    credential_scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    date_key = _sign(("AWS4" + secret).encode("utf-8"), datestamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, "s3")
    signing_key = _sign(service_key, "aws4_request")
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    request = urllib.request.Request(
        s3_endpoint() + canonical_uri,
        data=body,
        headers={
            "Host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "Authorization": authorization,
            "Content-Type": "application/octet-stream",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"S3 rejected the checkpoint ({exc.code}): {detail}") from None
    return s3_object_url(key)


def deliver_checkpoint(
    path: Path,
    filename: str,
    *,
    run_id: str | None = None,
    limit: int | None = None,
    upload=put_s3_object,
) -> dict:
    if limit is None:
        limit = MAX_ATTACHMENT_BYTES
    if path.stat().st_size <= limit:
        return {
            "mode": "attach",
            "filename": filename,
            "path": path,
            "url": None,
            "key": None,
        }
    key = s3_checkpoint_key(filename, run_id)
    url = upload(path, key)
    return {
        "mode": "s3",
        "filename": filename,
        "path": path,
        "url": url,
        "key": key,
    }


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
        parts.append(f"{minutes:02d}m")
    elif minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs:02d}s" if parts else f"{secs}s")
    return " ".join(parts)


def parse_run_log(text: str) -> dict:
    stamps = TIMESTAMP_RE.findall(text)
    if len(stamps) >= 2:
        start = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S")
        training_time = format_duration((end - start).total_seconds())
    elif len(stamps) == 1:
        training_time = "0s"
    else:
        training_time = "unknown"
    rates = THROUGHPUT_RE.findall(text)
    throughput = f"{rates[-1]} transitions/s" if rates else "unknown"
    return {"training_time": training_time, "throughput": throughput}


def load_run_snapshot(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint_meta(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a policy artifact")
    transitions = payload.get("env_transitions")
    if transitions is None:
        match = re.search(r"policy_(\d+)\.pt$", path.name)
        transitions = int(match.group(1)) if match else None
    version = payload.get("policy_format_version")
    obs_dim = payload.get("actor_obs_dim")
    policy = "unknown"
    if version is not None and obs_dim is not None:
        policy = f"v{int(version)}, obs {int(obs_dim)}"
    elif version is not None:
        policy = f"v{int(version)}"
    return {
        "transitions": None if transitions is None else int(transitions),
        "policy": policy,
        "payload": payload,
    }


def running_count(lengths) -> tuple[int, int]:
    import numpy as np

    lengths = np.asarray(lengths)
    n = int(lengths.shape[0])
    if n == 0:
        return 0, 0
    last_t = int(lengths.max()) - 1
    running = int(np.sum(lengths > last_t))
    return running, n


def load_follow_eval(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "running" in data and isinstance(data["running"], str) and "/" in data["running"]:
        return data["running"]
    if "running" in data and "cars" in data:
        return f"{int(data['running'])}/{int(data['cars'])}"
    return None


def parse_wandb_url(url: str) -> tuple[str, str, str] | None:
    match = WANDB_URL_RE.match(url.strip())
    if match:
        return match.group(1), match.group(2), match.group(3)
    parts = url.strip().split("/")
    if len(parts) == 3 and ":" not in url:
        return parts[0], parts[1], parts[2]
    return None


def fetch_wandb_rows(wandb_url: str) -> list[dict]:
    parsed = parse_wandb_url(wandb_url)
    if parsed is None:
        raise ValueError(f"cannot parse W&B run from {wandb_url}")
    entity, project, run_id = parsed
    import wandb

    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    keys = ["_step", "env_transitions", "train/mean_ep_reward", "env/laps_completed"]
    return list(run.scan_history(keys=keys))


def write_history_charts(rows: list[dict], reward_path: Path, laps_path: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reward_x, reward_y = [], []
    laps_x, laps_y = [], []
    for row in rows:
        step = row.get("env_transitions", row.get("_step"))
        if step is None:
            continue
        reward = row.get("train/mean_ep_reward")
        laps = row.get("env/laps_completed")
        if reward is not None:
            reward_x.append(step)
            reward_y.append(reward)
        if laps is not None:
            laps_x.append(step)
            laps_y.append(laps)
    notes = []
    if reward_x:
        _plot(plt, reward_x, reward_y, "Mean episode reward", reward_path)
    else:
        notes.append("train/mean_ep_reward")
    if laps_x:
        _plot(plt, laps_x, laps_y, "Laps completed", laps_path)
    else:
        notes.append("env/laps_completed")
    return notes


def _plot(plt, xs, ys, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(xs, ys, color="#3d7eff", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("transitions")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_follow_video(
    npz: Path,
    mp4: Path,
    *,
    fps: int = 10,
    width: int = 1120,
    height: int = 700,
) -> None:
    if not RENDER_SCRIPT.is_file():
        raise SystemExit(f"follow renderer missing: {RENDER_SCRIPT}")
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--npz",
        str(npz),
        "--mp4",
        str(mp4),
        "--fps",
        str(fps),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    subprocess.run(cmd, check=True)


def record_follow_rollout(
    checkpoint: Path,
    run_dir: Path | None,
    npz_path: Path,
    *,
    seconds: float,
    fps: int,
    seed: int,
    device_arg: str,
    track: str | None,
) -> str:
    """Record an 8-car CPU rollout into the npz schema ``render_follow.py`` reads."""
    if device_arg != "cpu":
        raise SystemExit(
            "Refusing to record on a non-CPU device while a training arm may "
            "own the GPU. Pass --device cpu."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    from evaluation import actor_is_recurrent, load_sensor_actor_bundle
    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt
    from f1tenth_env.eval_viz import yaw_from_quat_wxyz
    from f1tenth_env.utils import compute_track_boundaries
    from standalone_trainer import DEFAULT_CONFIG, build_env_cfg

    snapshot = load_run_snapshot(run_dir)
    cfg = copy.deepcopy(snapshot.get("config") or DEFAULT_CONFIG)
    if track:
        cfg["env"]["track"] = track
    cfg["env"]["domain_randomization"] = {
        **cfg["env"].get("domain_randomization", {}),
        "enabled": False,
    }
    cfg["env"]["opponent_strategy"] = "scripted"
    device = torch.device("cpu")
    rt.configure(float_dtype=torch.float32, int_dtype=torch.int32, dev=device, eps=1e-12)
    meta = load_checkpoint_meta(checkpoint)
    payload = meta["payload"]
    actor, normalizer, _arch, _payload = load_sensor_actor_bundle(
        checkpoint,
        device,
        expected_actor_obs_dim=int(payload["actor_obs_dim"]),
        expected_action_dim=int(payload["action_dim"]),
        expected_layout_version=int(payload["actor_layout_version"]),
        expected_critic_obs_dim=payload.get("critic_obs_dim"),
        norm_eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        norm_clip=float(cfg["obs"].get("norm_clip", 10.0)),
        require_obs_norm=True,
    )
    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="fixed",
        launch_strategy_data={"num_cars": CARS},
    )
    env = F1tenthEnv(
        num_envs=CARS,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )
    try:
        return _rollout_to_npz(
            env,
            actor,
            normalizer,
            cfg,
            npz_path,
            seconds=seconds,
            fps=fps,
            seed=seed,
            torch=torch,
            np=np,
            yaw_from_quat_wxyz=yaw_from_quat_wxyz,
            compute_track_boundaries=compute_track_boundaries,
            actor_is_recurrent=actor_is_recurrent,
        )
    finally:
        env.close()


def _rollout_to_npz(env, actor, normalizer, cfg, npz_path, **deps) -> str:
    torch = deps["torch"]
    np = deps["np"]
    seconds = deps["seconds"]
    fps = deps["fps"]
    seed = deps["seed"]
    control_dt = float(cfg["env"]["sim_dt"]) * int(cfg["env"]["control_interval"])
    stride = max(1, int(round((1.0 / fps) / control_dt)))
    n_frames = max(2, int(round(seconds * fps)))
    clip = float(cfg["env"]["clip_actions"])
    center = np.asarray(env.track_state["centerline"], dtype=np.float64)[:, :2]
    left, right = deps["compute_track_boundaries"](
        center.astype(np.float32),
        np.asarray(env.track_state["w_tr_left"], dtype=np.float32),
        np.asarray(env.track_state["w_tr_right"], dtype=np.float32),
    )
    p0 = center[0]
    p1 = center[min(1, len(center) - 1)]
    yaw0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    travel = np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))
    opp_i = int(np.searchsorted(travel, 3.0)) + 1
    opp_i = min(max(opp_i, 1), len(center) - 1)
    raw, _extras = env.reset_to(
        p0,
        yaw0,
        2.0,
        opponent_pose=center[opp_i],
        opponent_yaw=yaw0,
        opponent_speed=2.0,
        seed=seed,
        with_sensors=True,
    )
    obs = raw["actor"].to(torch.float32)
    recurrent = deps["actor_is_recurrent"](actor)
    hidden = None
    if recurrent:
        hidden = actor.initial_hidden(
            CARS, device=obs.device, dtype=torch.float32
        )
    alive = [True] * CARS
    crashed = [False] * CARS
    ego_xy = [[] for _ in range(CARS)]
    ego_yaw = [[] for _ in range(CARS)]
    ego_speed = [[] for _ in range(CARS)]
    opp_xy = [[] for _ in range(CARS)]
    opp_yaw = [[] for _ in range(CARS)]

    def record() -> None:
        st = env.read_state()
        xy = st["base_pos"][:, :2].cpu().numpy()
        yaws = [
            deps["yaw_from_quat_wxyz"](q.tolist()) for q in st["base_quat"]
        ]
        speed = torch.linalg.norm(st["base_lin_vel"][:, :2], dim=-1).cpu().numpy()
        oxy = st["opp_base_pos"][:, :2].cpu().numpy()
        oyaw = [
            deps["yaw_from_quat_wxyz"](q.tolist()) for q in st["opp_base_quat"]
        ]
        for i in range(CARS):
            if not alive[i]:
                continue
            ego_xy[i].append(xy[i])
            ego_yaw[i].append(yaws[i])
            ego_speed[i].append(float(speed[i]))
            opp_xy[i].append(oxy[i])
            opp_yaw[i].append(oyaw[i])

    record()
    with torch.no_grad():
        for _frame in range(1, n_frames):
            for _sub in range(stride):
                model_obs = normalizer.normalize(obs)
                if recurrent:
                    actions, _, hidden = actor.step(
                        model_obs,
                        hidden,
                        reset_mask=None,
                        deterministic=True,
                        with_logprob=False,
                    )
                else:
                    actions, _ = actor(
                        model_obs, deterministic=True, with_logprob=False
                    )
                actions = actions.clamp(-clip, clip)
                raw, _reward, done, _extras = env.step(
                    actions.to(torch.float32),
                    n_steps=int(cfg["env"]["control_interval"]),
                    with_sensors=True,
                )
                obs = raw["actor"].to(torch.float32)
                done_np = done.detach().cpu().numpy().astype(bool)
                if recurrent and hidden is not None:
                    done_b = done.to(device=hidden.device, dtype=torch.bool)
                    if done_b.any():
                        hidden[done_b] = 0
                for i in range(CARS):
                    if alive[i] and bool(done_np[i]):
                        alive[i] = False
                        crashed[i] = True
            record()
    lengths = np.array([len(ego_xy[i]) for i in range(CARS)], dtype=np.int32)
    payload = {
        "lengths": lengths,
        "crashed": np.asarray(crashed, dtype=bool),
        "car_length": np.float64(cfg["env"].get("car_length", 0.568)),
        "car_width": np.float64(cfg["env"].get("car_width", 0.296)),
        "left": np.asarray(left, dtype=np.float64),
        "right": np.asarray(right, dtype=np.float64),
        "centerline": center,
    }
    for i in range(CARS):
        payload[f"ego_xy_{i}"] = np.asarray(ego_xy[i], dtype=np.float64)
        payload[f"ego_yaw_{i}"] = np.asarray(ego_yaw[i], dtype=np.float64)
        payload[f"ego_speed_{i}"] = np.asarray(ego_speed[i], dtype=np.float64)
        payload[f"opp_xy_{i}"] = np.asarray(opp_xy[i], dtype=np.float64)
        payload[f"opp_yaw_{i}"] = np.asarray(opp_yaw[i], dtype=np.float64)
    np.savez(npz_path, **payload)
    running, n = running_count(lengths)
    return f"{running}/{n}"


def build_embed(fields: dict, summary: str, charts_note: str | None) -> dict:
    description = summary.strip()
    if charts_note:
        description = f"{description}\n\n{charts_note}"
    body = [
        ("Track", fields["track"]),
        ("Training time", fields["training_time"]),
        ("Throughput", fields["throughput"]),
        ("Transitions", fields["transitions"]),
        ("Policy", fields["policy"]),
        ("Eval", fields["eval"]),
        ("W&B", fields["wandb"]),
    ]
    if fields.get("full_lap"):
        body.append(("Full lap", fields["full_lap"]))
    if fields.get("checkpoint_ref"):
        body.append(("Checkpoint", fields["checkpoint_ref"]))
    long_fields = ("W&B", "Full lap", "Checkpoint")
    discord_fields = [
        {"name": name, "value": str(value), "inline": name not in long_fields}
        for name, value in body
    ]
    kind = fields.get("kind", "release")
    embed = {
        "title": KIND_TITLES.get(kind, KIND_TITLES["release"]),
        "description": description,
        "color": 3447003,
        "fields": discord_fields,
    }
    return embed


def build_payload(embed: dict, names: list[str]) -> dict:
    attachments = [{"id": i, "filename": name} for i, name in enumerate(names)]
    embeds = [embed]
    if "reward.png" in names:
        embeds[0]["image"] = {"url": "attachment://reward.png"}
    if "laps.png" in names:
        extra = {"image": {"url": "attachment://laps.png"}}
        embeds.append(extra)
    return {"embeds": embeds, "attachments": attachments}


def encode_multipart(payload: dict, files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = "----f1tenthRelease" + uuid.uuid4().hex
    chunks: list[bytes] = []
    crlf = b"\r\n"

    def add(part: bytes) -> None:
        chunks.append(b"--" + boundary.encode() + crlf + part)

    add(
        b'Content-Disposition: form-data; name="payload_json"'
        + crlf
        + b"Content-Type: application/json"
        + crlf
        + crlf
        + json.dumps(payload).encode()
        + crlf
    )
    for index, (filename, path) in enumerate(files):
        if filename.endswith(".png"):
            mime = "image/png"
        elif filename.endswith(".mp4"):
            mime = "video/mp4"
        else:
            mime = "application/octet-stream"
        header = (
            f'Content-Disposition: form-data; name="files[{index}]"; '
            f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode()
        add(header + path.read_bytes() + crlf)
    chunks.append(b"--" + boundary.encode() + b"--" + crlf)
    return b"".join(chunks), "multipart/form-data; boundary=" + boundary


def post_webhook(webhook_url: str, payload: dict, files: list[tuple[str, Path]]) -> None:
    body, content_type = encode_multipart(payload, files)
    separator = "&" if "?" in webhook_url else "?"
    request = urllib.request.Request(
        webhook_url + separator + "wait=true",
        data=body,
        headers={
            "Content-Type": content_type,
            "User-Agent": "f1tenth-model-release",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"Discord rejected the release ({exc.code}): {detail}") from None


def resolve_release(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir) if args.run_dir else None
    checkpoint = Path(args.checkpoint)
    snapshot = load_run_snapshot(run_dir)
    log_text = ""
    if run_dir is not None and (run_dir / "run.log").is_file():
        log_text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    parsed = parse_run_log(log_text)
    meta = load_checkpoint_meta(checkpoint)
    track = args.track or snapshot.get("config", {}).get("env", {}).get("track") or "unknown"
    training_time = args.training_time or parsed["training_time"]
    throughput = args.throughput or parsed["throughput"]
    transitions = meta["transitions"]
    transitions_text = "unknown" if transitions is None else str(transitions)
    policy_name = (
        f"policy_{transitions}.pt" if transitions is not None else checkpoint.name
    )
    wandb_url = args.wandb_url
    if not wandb_url and snapshot.get("run_id"):
        wandb_url = f"f1tenth-genesis / {snapshot['run_id']}"
    running = args.running
    if running is None and run_dir is not None:
        running = load_follow_eval(run_dir / "follow_eval.json")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir or checkpoint.parent) / "release"
    video = Path(args.video) if args.video else None
    npz = Path(args.npz) if args.npz else None
    if args.render:
        if args.device != "cpu":
            raise SystemExit(
                "Refusing to record on a non-CPU device while a training arm may "
                "own the GPU. Pass --device cpu."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        npz = out_dir / "follow_rollout.npz"
        video = out_dir / "overlay_follow.mp4"
        running = record_follow_rollout(
            checkpoint,
            run_dir,
            npz,
            seconds=args.seconds,
            fps=args.fps,
            seed=args.seed,
            device_arg=args.device,
            track=args.track,
        )
        render_follow_video(npz, video, fps=args.fps)
        (out_dir / "follow_eval.json").write_text(
            json.dumps({"running": running, "seconds": args.seconds}) + "\n",
            encoding="utf-8",
        )
    elif npz is not None and video is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        video = out_dir / "overlay_follow.mp4"
        render_follow_video(npz, video, fps=args.fps)
        if running is None:
            import numpy as np

            running_n, cars = running_count(np.load(npz)["lengths"])
            running = f"{running_n}/{cars}"
    elif video is None and run_dir is not None:
        candidate = run_dir / "overlay_follow.mp4"
        if candidate.is_file():
            video = candidate
    if running is None and npz is not None and npz.is_file():
        import numpy as np

        running_n, cars = running_count(np.load(npz)["lengths"])
        running = f"{running_n}/{cars}"
    reward = Path(args.reward_chart) if args.reward_chart else None
    laps = Path(args.laps_chart) if args.laps_chart else None
    if reward is None and run_dir is not None and (run_dir / "reward.png").is_file():
        reward = run_dir / "reward.png"
    if laps is None and run_dir is not None and (run_dir / "laps.png").is_file():
        laps = run_dir / "laps.png"
    charts_note = None
    if reward is None and laps is None and args.wandb_url and not args.dry_run:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = fetch_wandb_rows(args.wandb_url)
            reward = out_dir / "reward.png"
            laps = out_dir / "laps.png"
            missing = write_history_charts(rows, reward, laps)
            if not reward.is_file():
                reward = None
            if not laps.is_file():
                laps = None
            if missing:
                charts_note = (
                    "W&B history was missing "
                    + " and ".join(missing)
                    + ", so those charts are omitted."
                )
        except Exception as exc:
            reward = None
            laps = None
            charts_note = f"W&B history was not available ({exc.__class__.__name__})."
    elif reward is None and laps is None:
        charts_note = "W&B history was not available, so this release has no training charts."
    if reward is not None and laps is not None:
        charts_note = None
    eval_text = f"running {running or 'unknown'}"
    if args.seconds and args.render:
        eval_text += f" at {args.seconds:g} s"
    elif running and not args.render:
        eval_text += " at 60 s"
    return {
        "track": track,
        "training_time": training_time,
        "throughput": throughput,
        "transitions": transitions_text,
        "policy": meta["policy"],
        "eval": eval_text,
        "wandb": wandb_url or "not linked",
        "full_lap": args.full_lap,
        "summary": args.summary,
        "charts_note": charts_note,
        "video": video,
        "reward": reward,
        "laps": laps,
        "checkpoint": checkpoint,
        "policy_name": policy_name,
        "kind": args.kind,
        "run_id": snapshot.get("run_id"),
    }


def named_files(
    release: dict, *, attach_checkpoint: bool = True
) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    if release["video"] is not None:
        files.append(("overlay_follow.mp4", release["video"]))
    if release["reward"] is not None:
        files.append(("reward.png", release["reward"]))
    if release["laps"] is not None:
        files.append(("laps.png", release["laps"]))
    if attach_checkpoint:
        files.append((release["policy_name"], release["checkpoint"]))
    return files


def print_release(
    release: dict, files: list[tuple[str, Path]], delivery: dict | None = None
) -> None:
    print(f"kind: {release.get('kind', 'release')}")
    print(f"track: {release['track']}")
    print(f"training time: {release['training_time']}")
    print(f"throughput: {release['throughput']}")
    print(f"transitions: {release['transitions']}")
    print(f"policy format: {release['policy']}")
    print(release["eval"])
    print(f"summary: {release['summary']}")
    if release.get("full_lap"):
        print(f"full lap: {release['full_lap']}")
    print(f"wandb: {release['wandb']}")
    if delivery is not None:
        if delivery["mode"] == "attach":
            print(f"checkpoint: attach {delivery['filename']}")
        else:
            print(f"checkpoint: s3 {delivery['url']}")
    if release["charts_note"]:
        print(release["charts_note"])
    print("attachments:")
    total = 0
    for name, path in files:
        size = path.stat().st_size
        total += size
        print(f"  {name} {size}")
    print(f"total bytes: {total}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Announce one policy release on Discord")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--wandb-url", default=None)
    parser.add_argument("--full-lap", default=None)
    parser.add_argument("--track", default=None)
    parser.add_argument("--training-time", default=None)
    parser.add_argument("--throughput", default=None)
    parser.add_argument("--running", default=None, help="Eval result, e.g. 6/8")
    parser.add_argument("--video", default=None)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--reward-chart", default=None)
    parser.add_argument("--laps-chart", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--kind",
        choices=tuple(KIND_TITLES),
        default="release",
        help="release, result, or loss; every kind includes the checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.summary.strip():
        raise SystemExit("--summary is required")
    if args.device != "cpu":
        raise SystemExit(
            "Refusing to record on a non-CPU device while a training arm may "
            "own the GPU. Pass --device cpu."
        )
    if args.render:
        # Hide the GPU before the first torch import in this process.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    release = resolve_release(args)
    if not release["checkpoint"].is_file():
        raise SystemExit(f"missing checkpoint: {release['checkpoint']}")
    if not args.dry_run:
        load_release_env()
    upload = planned_s3_url if args.dry_run else put_s3_object
    delivery = deliver_checkpoint(
        release["checkpoint"],
        release["policy_name"],
        run_id=release.get("run_id"),
        upload=upload,
    )
    release["checkpoint_ref"] = (
        delivery["url"] if delivery["mode"] == "s3" else delivery["filename"]
    )
    files = named_files(release, attach_checkpoint=delivery["mode"] == "attach")
    missing = [path for _name, path in files if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise SystemExit(f"missing attachment: {joined}")
    print_release(release, files, delivery)
    if args.dry_run:
        print("dry-run: not posting")
        return
    if args.kind == "release" and release["video"] is None:
        raise SystemExit("a follow video is required; pass --video, --npz, or --render")
    webhook = os.environ.get("DISCORD_MODEL_RELEASE_WEBHOOK", "").strip()
    if not webhook:
        raise SystemExit("DISCORD_MODEL_RELEASE_WEBHOOK is not set")
    embed = build_embed(
        release,
        release["summary"],
        release["charts_note"],
    )
    payload = build_payload(embed, [name for name, _path in files])
    post_webhook(webhook, payload, files)
    print("posted one Discord message")


if __name__ == "__main__":
    main()

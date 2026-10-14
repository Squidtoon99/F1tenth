---
name: model-release
description: Announce an F1TENTH policy on the Discord release channel with a follow-camera eval video, W&B charts, and the checkpoint. Use when a trained policy is ready to share.
---

# Model release

Post one Discord message when a policy is ready to share. Not on every checkpoint and not at the end of every training run.

The webhook URL lives only in `DISCORD_MODEL_RELEASE_WEBHOOK`. It stays out of the repo, command lines, and logs.

## What to pass

- Checkpoint path (`policy_<transitions>.pt`)
- W&B run URL (`f1tenth-genesis`)
- `--summary`, one line on what changed. Required. Do not invent it.
- `--full-lap`, the full-lap numbers for this policy

## Sequence

1. Dry-run. This does not call Discord or W&B.
2. Read the printed track, training time, throughput, transitions, policy format, `running n/8`, summary, and attachment sizes.
3. Post once. Do not post a second time to check.

Stay on CPU (`--device cpu`). A live training arm keeps the GPU.

The follow video comes from `training/render_follow.py`, the script that produced `overlay_follow.mp4`. Record a trajectory and hand it that file. Do not replace the camera.

## Commands

```bash
python training/announce_model_release.py \
  --checkpoint <run>/checkpoints/policy_<transitions>.pt \
  --run-dir <run> \
  --summary "one line on what changed" \
  --wandb-url https://wandb.ai/<entity>/f1tenth-genesis/runs/<id> \
  --full-lap "median 12.4s, best 11.8s" \
  --render \
  --device cpu \
  --dry-run
```

Drop `--dry-run` to post. `DISCORD_MODEL_RELEASE_WEBHOOK` must be set. `--render` records eight fixed-spawn policy cars and one scripted centerline opponent, writes the `.npz`, and calls `render_follow.py` for a 60 s, 10 fps, 1120×700 clip.

Pass `--video` or `--npz` instead of `--render` to reuse a clip. `--track`, `--training-time`, and `--throughput` override the values parsed from `config.json` and `run.log`.

Charts are mean episode reward and laps completed versus transitions. If that W&B history is missing, the message still posts and the embed says so. Combined attachments must stay under 10 MB; the script refuses before posting when they do not.

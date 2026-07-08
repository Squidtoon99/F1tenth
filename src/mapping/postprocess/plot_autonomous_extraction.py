#!/usr/bin/env python3
"""Visualize centerline extraction: autonomous gradient plan or boundary/width panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from centerline_extractor import extract_centerline, load_map_yaml, occupancy_grid_to_gray
from track_geometry import arc_length, loop_length, nearest_distances, track_extent
from track_reference import boundaries_from_result, load_reference_track
from track_viz import gradient_line, map_image_extent, plot_centerline_graph


def plot_autonomous_plan(
    map_yaml: Path,
    out_png: Path,
    reference_csv: Path | None = None,
    spacing_m: float = 0.07,
    cmap: str = "viridis",
) -> tuple[np.ndarray, np.ndarray, dict]:
    data = load_map_yaml(map_yaml)
    img = occupancy_grid_to_gray(data.grid)
    result = extract_centerline(data, spacing_m=spacing_m, prefer_known_track=False)
    cl = result.centerline
    s = arc_length(cl)

    extent_full = map_image_extent(data, img)
    x0, x1, y0, y1 = track_extent([cl], data.origin_x, data.origin_y, data.resolution, img.shape)

    stats: dict = {
        "points": int(cl.shape[0]),
        "lap_length_m": float(loop_length(cl)),
    }
    if reference_csv is not None and reference_csv.is_file():
        ref, _, _, _, _ = load_reference_track(reference_csv)
        d = nearest_distances(cl, ref, step=3)
        stats["mean_error_m"] = float(np.mean(d))
        stats["length_ratio"] = float(loop_length(cl) / max(loop_length(ref), 1e-6))

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(img, cmap="gray", origin="lower", extent=extent_full, alpha=0.95, vmin=0, vmax=255)

    if reference_csv is not None and reference_csv.is_file():
        ref, _, _, _, _ = load_reference_track(reference_csv)
        ax.plot(ref[:, 0], ref[:, 1], color="#aaaaaa", lw=1.0, alpha=0.7, label="Reference (survey)")

    gradient_line(ax, cl, s, cmap=cmap, lw=2.8)
    sm = ScalarMappable(norm=Normalize(0.0, float(s[-1])), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Arc length along plan (m)")

    ax.scatter([cl[0, 0]], [cl[0, 1]], s=80, c="#00ff00", edgecolors="black", linewidths=0.8, zorder=5, label="Start")
    ax.scatter([cl[-1, 0]], [cl[-1, 1]], s=80, c="#ff0000", edgecolors="black", linewidths=0.8, zorder=5, label="End")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = f"Autonomous extraction plan — {map_yaml.stem}"
    if "mean_error_m" in stats:
        title += f"  (mean err {stats['mean_error_m']:.2f} m)"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return cl, s, stats


def animate_autonomous_plan(
    map_yaml: Path,
    out_gif: Path,
    cl: np.ndarray,
    s: np.ndarray,
    reference_csv: Path | None = None,
    cmap: str = "viridis",
    fps: int = 30,
    duration_s: float = 8.0,
) -> Path:
    data = load_map_yaml(map_yaml)
    img = occupancy_grid_to_gray(data.grid)
    extent_full = map_image_extent(data, img)
    x0, x1, y0, y1 = track_extent([cl], data.origin_x, data.origin_y, data.resolution, img.shape)

    n_frames = max(int(fps * duration_s), 60)
    frame_idx = np.linspace(0, cl.shape[0] - 1, n_frames, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(img, cmap="gray", origin="lower", extent=extent_full, alpha=0.95, vmin=0, vmax=255)
    if reference_csv is not None and reference_csv.is_file():
        ref, _, _, _, _ = load_reference_track(reference_csv)
        ax.plot(ref[:, 0], ref[:, 1], color="#aaaaaa", lw=1.0, alpha=0.7)

    lc_holder: list[LineCollection | None] = [None]
    start_dot = ax.scatter([], [], s=80, c="#00ff00", edgecolors="black", linewidths=0.8, zorder=5)
    head_dot = ax.scatter([], [], s=60, c="#ffffff", edgecolors="black", linewidths=0.8, zorder=6)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Autonomous plan trace — {map_yaml.stem}")

    sm = ScalarMappable(norm=Normalize(0.0, float(s[-1])), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="Arc length (m)")

    def _draw_partial(end_idx: int) -> None:
        if lc_holder[0] is not None:
            lc_holder[0].remove()
        partial = cl[: end_idx + 1]
        if partial.shape[0] < 2:
            return
        ps = arc_length(partial)
        lc_holder[0] = gradient_line(ax, partial, ps, cmap=cmap, lw=2.8)
        start_dot.set_offsets(partial[:1])
        head_dot.set_offsets(partial[-1:])

    def update(frame: int):
        _draw_partial(int(frame_idx[frame]))
        return ()

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=1000 / fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_gif


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=repo / "ros2_deploy/assets/Oschersleben_map.yaml",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=repo / "ros2_deploy/assets/Oschersleben_centerline.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "ros2_mapping/output/oschersleben_autonomous",
    )
    parser.add_argument(
        "--centerline-csv",
        type=Path,
        default=None,
        help="Plot boundaries/widths from CSV instead of autonomous extraction",
    )
    parser.add_argument("--spacing-m", type=float, default=0.07)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--animate", action="store_true", help="Write progressive trace GIF (autonomous mode only)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=8.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.map_yaml.stem

    if args.centerline_csv is not None:
        data = load_map_yaml(args.map_yaml)
        cl, wl, wr, _, _ = load_reference_track(args.centerline_csv)
        left, right = boundaries_from_result(cl, wl, wr)
        out_path = args.out_dir / f"{stem}_centerline_graph.png"
        saved = plot_centerline_graph(data, cl, wl, wr, left, right, out_path, stem)
        print(saved)
        return

    png_path = args.out_dir / f"{stem}_autonomous_plan.png"
    cl, s, stats = plot_autonomous_plan(
        args.map_yaml,
        png_path,
        reference_csv=args.reference_csv,
        spacing_m=args.spacing_m,
        cmap=args.cmap,
    )
    print(f"PNG: {png_path}")
    print(f"  points={stats['points']} lap={stats['lap_length_m']:.1f}m")
    if "mean_error_m" in stats:
        print(f"  vs reference: mean={stats['mean_error_m']:.2f}m ratio={stats['length_ratio']:.3f}")

    if args.animate:
        gif_path = args.out_dir / f"{stem}_autonomous_plan.gif"
        animate_autonomous_plan(
            args.map_yaml,
            gif_path,
            cl,
            s,
            reference_csv=args.reference_csv,
            cmap=args.cmap,
            fps=args.fps,
            duration_s=args.duration_s,
        )
        print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()

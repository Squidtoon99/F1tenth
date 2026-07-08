"""Shared matplotlib helpers for track extraction visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from centerline_extractor import MapData, occupancy_grid_to_gray
from track_geometry import arc_length, loop_length, resample_to_count, track_extent


def map_image_extent(data: MapData, img: np.ndarray) -> list[float]:
    return [
        data.origin_x,
        data.origin_x + img.shape[1] * data.resolution,
        data.origin_y,
        data.origin_y + img.shape[0] * data.resolution,
    ]


def gradient_line(
    ax,
    centerline: np.ndarray,
    s: np.ndarray,
    cmap: str = "viridis",
    lw: float = 2.8,
) -> LineCollection:
    pts = centerline.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = Normalize(vmin=0.0, vmax=float(s[-1]))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw, capstyle="round")
    lc.set_array(s[:-1])
    ax.add_collection(lc)
    return lc


def plot_width_profiles(
    ref_wl: np.ndarray,
    ref_wr: np.ndarray,
    ext_wl: np.ndarray,
    ext_wr: np.ndarray,
    out_path: Path,
    count: int = 800,
    title: str = "Track half-widths: reference vs post-processed",
) -> None:
    ref_wl_r = resample_to_count(ref_wl, count)
    ref_wr_r = resample_to_count(ref_wr, count)
    ext_wl_r = resample_to_count(ext_wl, count)
    ext_wr_r = resample_to_count(ext_wr, count)
    idx = np.arange(count)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(idx, ref_wl_r, color="#2ca02c", label="Reference left", lw=1.2)
    axes[0].plot(idx, ext_wl_r, color="#ff7f0e", label="Parsed left", lw=1.0, alpha=0.85)
    axes[0].set_ylabel("Left half-width (m)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(idx, ref_wr_r, color="#9467bd", label="Reference right", lw=1.2)
    axes[1].plot(idx, ext_wr_r, color="#d62728", label="Parsed right", lw=1.0, alpha=0.85)
    axes[1].set_ylabel("Right half-width (m)")
    axes[1].set_xlabel("Centerline sample index (resampled)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_centerline_graph(
    data: MapData,
    centerline: np.ndarray,
    wl: np.ndarray,
    wr: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    out_path: Path,
    title: str,
) -> Path:
    img = occupancy_grid_to_gray(data.grid)
    extent_full = map_image_extent(data, img)
    x0, x1, y0, y1 = track_extent([centerline, left, right], data.origin_x, data.origin_y, data.resolution, img.shape)
    s = arc_length(centerline)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.imshow(img, cmap="gray", origin="lower", extent=extent_full, alpha=0.95, vmin=0, vmax=255)
    ax.plot(centerline[:, 0], centerline[:, 1], color="#0066cc", lw=1.8, label="Centerline")
    ax.plot(left[:, 0], left[:, 1], color="#ff9900", lw=1.0, alpha=0.9, label="Left boundary")
    ax.plot(right[:, 0], right[:, 1], color="#cc00cc", lw=1.0, alpha=0.9, label="Right boundary")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    ax2.plot(s, wl, color="#2ca02c", lw=1.2, label="Left half-width")
    ax2.plot(s, wr, color="#9467bd", lw=1.2, label="Right half-width")
    ax2.set_xlabel("Arc length (m)")
    ax2.set_ylabel("Half-width (m)")
    ax2.set_title(f"Track widths  (lap {loop_length(centerline):.1f} m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_disparity_overlay(
    out_path: Path,
    data: MapData,
    extracted_cl: np.ndarray,
    ref_cl: np.ndarray,
    ref_left: np.ndarray,
    iteration: int,
    mean_err: float,
    title_suffix: str = "",
    cmap: str = "plasma",
) -> None:
    img = occupancy_grid_to_gray(data.grid)
    extent = map_image_extent(data, img)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(img, cmap="gray", origin="lower", extent=extent, alpha=0.92)
    ax.plot(ref_cl[:, 0], ref_cl[:, 1], color="#00cc00", lw=1.5, alpha=0.9, label="Truth centerline")
    ax.plot(ref_left[:, 0], ref_left[:, 1], color="#88ff88", lw=0.8, alpha=0.5, label="Truth left")

    s = arc_length(extracted_cl)
    gradient_line(ax, extracted_cl, s, cmap=cmap, lw=2.2)
    ax.scatter([ref_cl[0, 0]], [ref_cl[0, 1]], c="lime", s=60, edgecolors="k", zorder=5)
    ax.set_aspect("equal")
    ax.set_title(f"Iter {iteration} — mean err {mean_err:.2f} m {title_suffix}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

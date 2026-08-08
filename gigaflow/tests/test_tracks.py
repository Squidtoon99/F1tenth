"""Focused CPU tests for pinned track prep and packed atlas."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth import tracks as T
from gigaflow_f1tenth.sim.geometry import (
    make_synthetic_oval_atlas,
    make_two_track_atlas,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
PIN = ROOT / "configs" / "track_pin.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tracks"
PREPARED_ATLAS = Path.home() / ".cache" / "gigaflow" / "tracks"

PREVIEW_RADIUS_M = 10.0
PREVIEW_HALF_WIDTH_M = 1.1
PREVIEW_POINTS = 720


def test_pin_file_has_checksums_and_license_metadata():
    pin = T.load_pin(PIN)
    assert pin["license"] == "GPL-3.0"
    assert pin["repo"] == "f1tenth/f1tenth_racetracks"
    assert pin["ref"] == "v1.0.0"
    assert len(pin["tracks"]) == 23
    for name, digest in pin["tracks"].items():
        assert len(digest) == 64
        assert name.isidentifier() or name.replace("_", "").isalnum()


def test_validate_centerline_schema_rejects_bad_tables():
    good = T.make_synthetic_oval_table()
    T.validate_centerline_table(good)
    bad = dict(good)
    bad.pop("w_tr_left_m")
    with pytest.raises(T.TrackError, match="missing columns"):
        T.validate_centerline_table(bad)
    zero_w = dict(good)
    zero_w["w_tr_right_m"] = np.zeros_like(good["w_tr_right_m"])
    with pytest.raises(T.TrackError, match="positive"):
        T.validate_centerline_table(zero_w)


def test_normalize_strips_duplicate_endpoint_and_validates_geometry():
    table = T.load_centerline_csv(FIXTURES / "oval_centerline.csv")
    point, wr, wl = T.normalize_centerline_table(table)
    assert point.shape[0] == table["x_m"].shape[0] - 1
    stats = T.validate_track_geometry(point, wr, wl)
    assert stats["length_m"] > 0.0
    assert stats["width_min_m"] >= T.MIN_VEHICLE_CLEARANCE_M
    assert abs(stats["signed_area"]) > 0.0


def test_pack_atlas_ragged_offsets_luts_edts_and_capacity():
    tables = [
        T.load_centerline_csv(FIXTURES / "oval_centerline.csv"),
        T.load_centerline_csv(FIXTURES / "stadium_centerline.csv"),
    ]
    built = [
        T.build_track_arrays(
            tables[0],
            name="oval",
            source_sha="a" * 64,
            lut_resolution=0.5,
            edt_resolution=0.2,
            max_agents=6,
            license_id="local-fixture",
            optional_local=True,
        ),
        T.build_track_arrays(
            tables[1],
            name="stadium",
            source_sha="b" * 64,
            lut_resolution=0.5,
            edt_resolution=0.2,
            max_agents=6,
            license_id="local-fixture",
            optional_local=True,
        ),
    ]
    atlas = T.pack_built_tracks(built, sampling="stratified_shuffle")
    view = atlas.view()
    assert view.num_tracks == 2
    assert view.offsets.shape == (3,)
    assert view.offsets[-1] == view.centerline_xy.shape[0]
    assert view.lut_offsets[-1] == view.nearest_segment_lut.shape[0]
    assert view.edt_offsets[-1] == view.edt_distance.shape[0]
    assert view.centerline_xy.dtype == np.float32
    assert view.tangents_xy.shape == view.centerline_xy.shape
    assert view.widths_rl.shape == view.centerline_xy.shape
    assert view.capacity.shape == (2,)
    assert int(view.capacity.min()) >= 1
    assert int(view.capacity.max()) <= 6
    man = atlas.manifest()
    assert man[0].valid and man[1].valid
    assert man[0].geometry_hash != man[1].geometry_hash


def test_stratified_sampling_covers_all_tracks_each_round():
    lengths = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    ids = T.sample_track_ids(
        3, 9, seed=0, sampling="stratified_shuffle", lengths=lengths
    )
    assert ids.shape == (9,)
    for start in (0, 3, 6):
        assert set(ids[start : start + 3].tolist()) == {0, 1, 2}
    ids2 = T.sample_track_ids(
        3, 9, seed=0, sampling="stratified_shuffle", lengths=lengths
    )
    assert np.array_equal(ids, ids2)


def test_balanced_length_sampling_prefers_longer_tracks():
    lengths = np.asarray([1.0, 100.0], dtype=np.float64)
    ids = T.sample_track_ids(
        2, 2000, seed=1, sampling="balanced_length", lengths=lengths
    )
    frac_long = float(np.mean(ids == 1))
    assert frac_long > 0.9


def test_capacity_and_population_respect_max_agents():
    table = T.make_synthetic_oval_table(radius=20.0, n=120, width=2.0)
    point, wr, wl = T.normalize_centerline_table(table)
    stats = T.validate_track_geometry(point, wr, wl)
    cap = T.compute_track_capacity(
        stats["length_m"], wr, wl, max_agents=8, car_width_m=0.296, car_length_m=0.568
    )
    assert 1 <= cap <= 8
    narrow = T.compute_track_capacity(
        stats["length_m"],
        np.full_like(wr, 0.2),
        np.full_like(wl, 0.2),
        max_agents=8,
    )
    assert narrow <= cap
    track_ids = np.zeros(16, dtype=np.int32)
    capacities = np.asarray([cap], dtype=np.int32)
    counts = T.sample_active_counts(
        track_ids,
        capacities,
        density_bins=("sparse", "medium", "dense"),
        solo_world_fraction=0.25,
        max_agents=8,
        seed=0,
    )
    assert counts.shape == (16,)
    assert int(counts.min()) >= 1
    assert int(counts.max()) <= min(8, cap)
    assert np.any(counts == 1)


def test_pair_bin_and_h2h_sparse_trap():
    track_ids = np.zeros(32, dtype=np.int32)
    capacities = np.asarray([8], dtype=np.int32)
    for seed in range(16):
        sparse = T.sample_active_counts(
            track_ids,
            capacities,
            density_bins=("sparse",),
            solo_world_fraction=0.0,
            max_agents=2,
            seed=seed,
        )
        assert set(sparse.tolist()) == {1}
        pair = T.sample_active_counts(
            track_ids,
            capacities,
            density_bins=("pair",),
            solo_world_fraction=0.0,
            max_agents=2,
            seed=seed,
        )
        assert set(pair.tolist()) == {2}


def test_edt_zero_on_walls_positive_inside():
    occupied = np.zeros((21, 21), dtype=bool)
    occupied[0, :] = True
    occupied[-1, :] = True
    occupied[:, 0] = True
    occupied[:, -1] = True
    dist = T.euclidean_distance_transform(occupied)
    assert dist[0, 0] == 0.0
    assert dist[10, 10] > 5.0


def test_projection_wraps_lap_progress():
    table = T.make_synthetic_oval_table(radius=10.0, n=90, width=1.1)
    built = T.build_track_arrays(
        table,
        name="oval",
        source_sha="c" * 64,
        lut_resolution=0.5,
        edt_resolution=0.25,
        max_agents=4,
        license_id="local-fixture",
        optional_local=True,
    )
    # Query near start and near end of the loop.
    pos = np.asarray(
        [
            [10.0, 0.0],
            [10.0 * np.cos(-0.05), 10.0 * np.sin(-0.05)],
        ],
        dtype=np.float64,
    )
    s, lat, seg = T.project_to_centerline(
        pos, built.point, built.cum_length, built.length_m
    )
    assert s.shape == (2,)
    assert np.all(s >= 0.0) and np.all(s < built.length_m + 1.0e-3)
    assert np.all(np.abs(lat) < 0.5)
    assert seg.dtype == np.int32


def test_prepare_tracks_local_fixtures_cache_roundtrip(tmp_path: Path):
    cfg = load_config(SMOKE)
    local = tmp_path / "local"
    local.mkdir()
    shutil.copy(FIXTURES / "oval_centerline.csv", local / "oval_centerline.csv")
    atlas = T.prepare_tracks(
        cfg,
        str(tmp_path),
        pin_path=PIN,
        lut_resolution=0.5,
        edt_resolution=0.25,
        skip_download=True,
    )
    assert atlas.view().num_tracks == 1
    assert (tmp_path / T.MANIFEST_FILENAME).exists()
    assert (tmp_path / T.ATLAS_FILENAME).exists()
    loaded = T.load_atlas(str(tmp_path), device="cpu")
    assert loaded.view().num_tracks == 1
    assert loaded.manifest()[0].name == "oval"
    ids = loaded.sample_track_ids(4, seed=2)
    assert ids.shape == (4,)
    assert set(ids.tolist()) == {0}


def test_download_pinned_track_checksum_and_license_cache(tmp_path: Path):
    pin = T.load_pin(PIN)
    # One real upstream track keeps the network test focused and offline-cacheable.
    name = "Oschersleben"
    tiny = {
        "repo": pin["repo"],
        "ref": pin["ref"],
        "commit": pin["commit"],
        "license": pin["license"],
        "license_url": pin["license_url"],
        "attribution": pin["attribution"],
        "tracks": {name: pin["tracks"][name]},
        "optional_local": [],
    }
    cache = tmp_path / "cache"
    paths = T.download_pinned_sources(cache, pin=tiny)
    assert name in paths
    assert T._sha256_file(paths[name]) == tiny["tracks"][name]
    assert (cache / "meta" / T.LICENSE_FILENAME).exists()
    assert "GPL" in (cache / "meta" / T.LICENSE_FILENAME).read_text(encoding="utf-8")
    assert (cache / "meta" / T.ATTRIBUTION_FILENAME).exists()
    # Second call must reuse the checksummed cache entry.
    mtime = paths[name].stat().st_mtime_ns
    paths2 = T.download_pinned_sources(cache, pin=tiny)
    assert paths2[name] == paths[name]
    assert paths2[name].stat().st_mtime_ns == mtime


def test_checksum_mismatch_rejected_against_real_bytes(tmp_path: Path):
    pin = T.load_pin(PIN)
    tiny = {
        "repo": pin["repo"],
        "ref": pin["ref"],
        "commit": pin["commit"],
        "license": pin["license"],
        "license_url": pin["license_url"],
        "attribution": pin["attribution"],
        "tracks": {"Oschersleben": "0" * 64},
        "optional_local": [],
    }
    with pytest.raises(T.TrackError, match="checksum mismatch"):
        T.download_pinned_sources(tmp_path, pin=tiny)


def test_pin_count_matches_config_default():
    pin = T.load_pin(PIN)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.tracks.num_tracks == 23
    assert len(pin["tracks"]) == 23
    assert cfg.tracks.num_tracks == len(pin["tracks"])


@pytest.fixture(scope="module")
def fine_oval():
    """Densely sampled circle: analytic ground truth with negligible chord bias."""
    view = make_synthetic_oval_atlas(
        radius=PREVIEW_RADIUS_M,
        half_width=PREVIEW_HALF_WIDTH_M,
        num_points=PREVIEW_POINTS,
    )
    return view, float(np.asarray(view.lengths)[0])


@pytest.fixture(scope="module")
def two_track():
    return make_two_track_atlas()


def _capsule_table(
    *,
    straight_m: float = 20.0,
    radius: float = 5.0,
    spacing_m: float = 0.25,
    half_width: float = 1.5,
) -> dict[str, np.ndarray]:
    """Counter-clockwise stadium: exact straights joined by exact semicircles."""
    half = 0.5 * straight_m
    n_straight = int(round(straight_m / spacing_m))
    n_arc = int(round(np.pi * radius / spacing_m))
    along = np.arange(n_straight, dtype=np.float64) / n_straight * straight_m
    arc = np.arange(n_arc, dtype=np.float64) / n_arc * np.pi
    xs = np.concatenate(
        [
            -half + along,
            half + radius * np.cos(-0.5 * np.pi + arc),
            half - along,
            -half + radius * np.cos(0.5 * np.pi + arc),
        ]
    )
    ys = np.concatenate(
        [
            np.full(n_straight, -radius),
            radius * np.sin(-0.5 * np.pi + arc),
            np.full(n_straight, radius),
            radius * np.sin(0.5 * np.pi + arc),
        ]
    )
    return T.centerline_table_from_arrays(
        xs, ys, np.full(xs.shape[0], half_width), np.full(xs.shape[0], half_width)
    )


def _single_track_view(table, name: str) -> T.PackedTrackAtlasView:
    built = T.build_track_arrays(
        table,
        name=name,
        source_sha="synthetic",
        max_agents=4,
        license_id="local-fixture",
        optional_local=True,
    )
    return T.pack_built_tracks([built]).view()


def _circle_pose(radius: float, length: float, s: np.ndarray):
    """Pose on the ideal circle at arc length ``s`` (CCW, yaw along tangent)."""
    angle = 2.0 * np.pi * np.asarray(s, dtype=np.float64) / length
    return radius * np.cos(angle), radius * np.sin(angle), angle + 0.5 * np.pi


def _per_sample(preview: torch.Tensor, num_samples: int = T.TRACK_PREVIEW_SAMPLES):
    arr = preview.detach().numpy()
    return arr.reshape(
        *arr.shape[:-1], num_samples, T.TRACK_PREVIEW_SAMPLE_DIM
    )


def _to_world(samples: np.ndarray, x: np.ndarray, y: np.ndarray, yaw: np.ndarray):
    cos_yaw = np.cos(yaw)[..., None]
    sin_yaw = np.sin(yaw)[..., None]
    ego_x = samples[..., 0]
    ego_y = samples[..., 1]
    return (
        x[..., None] + cos_yaw * ego_x - sin_yaw * ego_y,
        y[..., None] + sin_yaw * ego_x + cos_yaw * ego_y,
    )


def _arc_advance(world_x, world_y, s_ego, length):
    """Metres of centerline travelled from the ego to each sample, on a circle."""
    ego_angle = 2.0 * np.pi * np.asarray(s_ego, dtype=np.float64) / length
    delta = np.mod(np.arctan2(world_y, world_x) - ego_angle[..., None], 2.0 * np.pi)
    return delta * length / (2.0 * np.pi)


def test_lookahead_preview_shape_speed_scaling_and_nearest_first_order(fine_oval):
    view, length = fine_oval
    s = np.asarray([[0.0, 0.25 * length], [0.5 * length, 0.75 * length]])
    speed = np.asarray([[0.0, 2.0], [4.0, 9.0]])
    x, y, yaw = _circle_pose(PREVIEW_RADIUS_M, length, s)
    preview = T.sample_track_lookahead(
        view,
        track_id=torch.zeros(s.shape, dtype=torch.int64),
        s=torch.as_tensor(s, dtype=torch.float32),
        x=torch.as_tensor(x, dtype=torch.float32),
        y=torch.as_tensor(y, dtype=torch.float32),
        yaw=torch.as_tensor(yaw, dtype=torch.float32),
        speed=torch.as_tensor(speed, dtype=torch.float32),
    )
    assert preview.shape == (2, 2, T.TRACK_PREVIEW_DIM)
    assert preview.dtype == torch.float32
    assert torch.isfinite(preview).all()

    samples = _per_sample(preview)
    advance = _arc_advance(*_to_world(samples, x, y, yaw), s, length)
    # max(speed * 6, 5) metres, split into K uniform steps, nearest sample first.
    lookahead = np.maximum(speed * T.TRACK_PREVIEW_SPEED_HORIZON_S,
                           T.TRACK_PREVIEW_MIN_LOOKAHEAD_M)
    assert lookahead.max() < length
    step = np.arange(1, T.TRACK_PREVIEW_SAMPLES + 1) / T.TRACK_PREVIEW_SAMPLES
    assert np.allclose(advance, lookahead[..., None] * step, atol=1.0e-3)
    assert np.all(np.diff(advance, axis=-1) > 0.0)
    # Speed 0 falls back to the 5 m floor, not a zero-length preview.
    assert advance[0, 0, -1] == pytest.approx(T.TRACK_PREVIEW_MIN_LOOKAHEAD_M, abs=1e-3)
    # Leading dims are free, so a [T, S] rollout batch is the same computation.
    time_major = T.sample_track_lookahead(
        view,
        track_id=torch.zeros((1,) + s.shape, dtype=torch.int64),
        s=torch.as_tensor(s[None], dtype=torch.float32),
        x=torch.as_tensor(x[None], dtype=torch.float32),
        y=torch.as_tensor(y[None], dtype=torch.float32),
        yaw=torch.as_tensor(yaw[None], dtype=torch.float32),
        speed=torch.as_tensor(speed[None], dtype=torch.float32),
    )
    assert time_major.shape == (1, 2, 2, T.TRACK_PREVIEW_DIM)
    assert torch.equal(time_major[0], preview)


def test_lookahead_preview_matches_analytic_circle_geometry(fine_oval):
    view, length = fine_oval
    s = np.asarray([3.0])
    x, y, yaw = _circle_pose(PREVIEW_RADIUS_M, length, s)
    preview = T.sample_track_lookahead(
        view,
        track_id=torch.zeros(1, dtype=torch.int64),
        s=torch.as_tensor(s, dtype=torch.float32),
        x=torch.as_tensor(x, dtype=torch.float32),
        y=torch.as_tensor(y, dtype=torch.float32),
        yaw=torch.as_tensor(yaw, dtype=torch.float32),
        speed=torch.zeros(1),
    )
    samples = _per_sample(preview)
    world_x, world_y = _to_world(samples, x, y, yaw)
    radius = np.hypot(world_x, world_y)
    # Samples ride the inscribed polygon, so they sit within one chord sag of R.
    sag = PREVIEW_RADIUS_M * (1.0 - np.cos(np.pi / PREVIEW_POINTS))
    assert np.all(radius <= PREVIEW_RADIUS_M + 1.0e-4)
    assert np.all(radius >= PREVIEW_RADIUS_M - sag - 1.0e-4)
    # Ego frame: forward is +x, and 5 m of a 10 m-radius circle bends left.
    assert np.all(samples[..., 0] > 0.0)
    assert samples[0, 0, 1] == pytest.approx(0.0, abs=5.0e-3)
    assert np.all(np.diff(samples[0, :, 1]) > 0.0)
    assert np.allclose(samples[..., 2], PREVIEW_HALF_WIDTH_M, atol=1.0e-6)
    assert np.allclose(samples[..., 3], PREVIEW_HALF_WIDTH_M, atol=1.0e-6)
    assert np.allclose(samples[..., 4], 1.0 / PREVIEW_RADIUS_M, atol=1.0e-5)


def test_lookahead_preview_wraps_across_the_start_line(fine_oval):
    view, length = fine_oval
    # One agent's 5 m preview crosses s = 0; the other is a quarter lap earlier.
    # A 720-point circle is exactly invariant under that shift, so the two
    # ego-frame previews must agree.
    s = np.asarray([length - 1.0, length - 1.0 - 0.25 * length])
    x, y, yaw = _circle_pose(PREVIEW_RADIUS_M, length, s)
    preview = T.sample_track_lookahead(
        view,
        track_id=torch.zeros(2, dtype=torch.int64),
        s=torch.as_tensor(s, dtype=torch.float64),
        x=torch.as_tensor(x, dtype=torch.float64),
        y=torch.as_tensor(y, dtype=torch.float64),
        yaw=torch.as_tensor(yaw, dtype=torch.float64),
        speed=torch.zeros(2, dtype=torch.float64),
    )
    samples = _per_sample(preview)
    assert np.allclose(samples[0], samples[1], atol=2.0e-4)
    advance = _arc_advance(*_to_world(samples, x, y, yaw), s, length)
    # Wrapping must not break monotonic arc progress or curvature.
    assert np.all(np.diff(advance, axis=-1) > 0.0)
    assert advance[0, -1] == pytest.approx(T.TRACK_PREVIEW_MIN_LOOKAHEAD_M, abs=1e-3)
    assert np.allclose(samples[..., 4], 1.0 / PREVIEW_RADIUS_M, atol=1.0e-5)
    # The lap boundary really is inside this preview window.
    assert s[0] + T.TRACK_PREVIEW_MIN_LOOKAHEAD_M > length


def test_lookahead_preview_multi_track_batch_matches_single_track_atlases(two_track):
    view = two_track
    lengths = np.asarray(view.lengths, dtype=np.float64)
    assert view.num_tracks == 2
    solo = [
        _single_track_view(
            T.make_synthetic_oval_table(radius=8.0, n=64, width=1.1), "oval_a"
        ),
        _single_track_view(
            T.make_synthetic_oval_table(radius=6.0, n=48, width=1.0), "oval_b"
        ),
    ]
    s = np.asarray([1.5, 4.25])
    speed = np.asarray([3.0, 0.0])
    x = np.asarray([0.0, 0.0])
    y = np.asarray([0.0, 0.0])
    yaw = np.asarray([0.3, -1.2])
    packed = T.sample_track_lookahead(
        view,
        track_id=torch.as_tensor([0, 1]),
        s=torch.as_tensor(s, dtype=torch.float32),
        x=torch.as_tensor(x, dtype=torch.float32),
        y=torch.as_tensor(y, dtype=torch.float32),
        yaw=torch.as_tensor(yaw, dtype=torch.float32),
        speed=torch.as_tensor(speed, dtype=torch.float32),
    )
    for track in (0, 1):
        alone = T.sample_track_lookahead(
            solo[track],
            track_id=torch.zeros(1, dtype=torch.int64),
            s=torch.as_tensor(s[track : track + 1], dtype=torch.float32),
            x=torch.as_tensor(x[track : track + 1], dtype=torch.float32),
            y=torch.as_tensor(y[track : track + 1], dtype=torch.float32),
            yaw=torch.as_tensor(yaw[track : track + 1], dtype=torch.float32),
            speed=torch.as_tensor(speed[track : track + 1], dtype=torch.float32),
        )
        assert torch.allclose(packed[track], alone[0], atol=1.0e-6)
    # Each agent reads its own track: radius 8 versus radius 6, widths 1.1 / 1.0.
    samples = _per_sample(packed)
    assert np.allclose(samples[0, :, 4], 1.0 / 8.0, atol=2.0e-3)
    assert np.allclose(samples[1, :, 4], 1.0 / 6.0, atol=4.0e-3)
    assert np.allclose(samples[0, :, 2:4], 1.1, atol=1.0e-6)
    assert np.allclose(samples[1, :, 2:4], 1.0, atol=1.0e-6)
    assert lengths[0] > lengths[1]


def test_lookahead_preview_curvature_zero_on_straight_and_matches_arc():
    straight_m, radius = 20.0, 5.0
    view = _single_track_view(
        _capsule_table(straight_m=straight_m, radius=radius), "capsule"
    )
    # s = 2 m sits on the bottom straight heading +x; s = straight + pi*r/2 is
    # the apex of the right semicircle. Both previews stay inside their feature.
    s = np.asarray([2.0, straight_m + 0.5 * np.pi * radius])
    preview = T.sample_track_lookahead(
        view,
        track_id=torch.zeros(2, dtype=torch.int64),
        s=torch.as_tensor(s, dtype=torch.float32),
        x=torch.as_tensor(
            [-0.5 * straight_m + s[0], radius + 0.5 * straight_m],
            dtype=torch.float32,
        ),
        y=torch.as_tensor([-radius, 0.0], dtype=torch.float32),
        yaw=torch.as_tensor([0.0, 0.5 * np.pi], dtype=torch.float32),
        speed=torch.zeros(2),
    )
    samples = _per_sample(preview)
    straight, arc = samples[0], samples[1]
    assert np.all(np.abs(straight[:, 4]) < 1.0e-4)
    assert np.allclose(arc[:, 4], 1.0 / radius, atol=1.0e-3)
    # Straight preview: uniformly spaced dead ahead of a car aligned with it.
    step = np.arange(1, T.TRACK_PREVIEW_SAMPLES + 1) / T.TRACK_PREVIEW_SAMPLES
    ahead = T.TRACK_PREVIEW_MIN_LOOKAHEAD_M * step
    assert np.allclose(straight[:, 0], ahead, atol=1.0e-4)
    assert np.allclose(straight[:, 1], 0.0, atol=1.0e-4)
    assert np.allclose(samples[..., 2:4], 1.5, atol=1.0e-6)


def test_lookahead_preview_rejects_degenerate_sampling(fine_oval):
    view, _ = fine_oval
    args = dict(
        track_id=torch.zeros(1, dtype=torch.int64),
        s=torch.zeros(1),
        x=torch.zeros(1),
        y=torch.zeros(1),
        yaw=torch.zeros(1),
        speed=torch.zeros(1),
    )
    with pytest.raises(T.TrackError, match="num_samples"):
        T.sample_track_lookahead(view, num_samples=0, **args)
    with pytest.raises(T.TrackError, match="min_lookahead_m"):
        T.sample_track_lookahead(view, min_lookahead_m=0.0, **args)
    with pytest.raises(T.TrackError, match="speed_horizon_s"):
        T.sample_track_lookahead(view, speed_horizon_s=-1.0, **args)
    short = T.sample_track_lookahead(view, num_samples=4, **args)
    assert short.shape == (1, 4 * T.TRACK_PREVIEW_SAMPLE_DIM)


@pytest.mark.skipif(
    not (PREPARED_ATLAS / T.MANIFEST_FILENAME).exists(),
    reason="prepared track atlas cache not present",
)
def test_lookahead_preview_on_prepared_multi_track_atlas():
    atlas = T.load_atlas(str(PREPARED_ATLAS), device="cpu")
    view = atlas.view()
    n = int(view.num_tracks)
    offsets = np.asarray(view.offsets, dtype=np.int64)
    centerline = np.asarray(view.centerline_xy, dtype=np.float64)
    lengths = np.asarray(view.lengths, dtype=np.float64)
    track_id = np.arange(n, dtype=np.int64)
    # One agent per real track, each half a metre short of the start line.
    s = lengths - 0.5
    start = centerline[offsets[:n]]
    rng = np.random.default_rng(0)
    speed = rng.uniform(0.0, 6.0, size=n)
    preview = T.sample_track_lookahead(
        view,
        track_id=torch.as_tensor(track_id),
        s=torch.as_tensor(s, dtype=torch.float32),
        x=torch.as_tensor(start[:, 0], dtype=torch.float32),
        y=torch.as_tensor(start[:, 1], dtype=torch.float32),
        yaw=torch.zeros(n),
        speed=torch.as_tensor(speed, dtype=torch.float32),
    )
    assert preview.shape == (n, T.TRACK_PREVIEW_DIM)
    assert torch.isfinite(preview).all()
    samples = _per_sample(preview)
    assert np.all(samples[..., 2:4] > 0.0)
    # Consecutive samples are one arc step apart, so their chord cannot exceed it.
    lookahead = np.maximum(speed * T.TRACK_PREVIEW_SPEED_HORIZON_S,
                           T.TRACK_PREVIEW_MIN_LOOKAHEAD_M)
    step = lookahead / T.TRACK_PREVIEW_SAMPLES
    chord = np.linalg.norm(np.diff(samples[..., 0:2], axis=-2), axis=-1)
    assert np.all(chord <= step[:, None] + 2.0e-3)
    assert np.all(chord >= step[:, None] * 0.4)

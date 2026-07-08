# mapping/postprocess

Offline track-geometry tooling. This is **not** a ROS 2 / colcon package (no
`package.xml`), so `colcon build` ignores it. It runs standalone on a workstation
to turn a recorded SLAM map or a reference raceline into the race assets
(centerlines, boundaries) consumed by the RL and algorithmic stacks.

## Contents
- `centerline_extractor.py`, `track_geometry.py`, `track_reference.py` - core
  geometry: extract a centerline from an occupancy map, resample it, compute
  boundaries and loop metrics.
- `build_centerline_from_map.py`, `build_centerline_from_raceline.py`,
  `slam_map_to_race.py` - CLI entry points that produce `*_centerline.csv`.
- `validate_track_alignment.py`, `run_e2e_validation.py`, `offline_mapping_e2e.py`
  - validation / alignment checks against a reference track.
- `overlay_tracks.py`, `track_viz.py`, `plot_autonomous_extraction.py` - plots.
- `test/` - unit tests for the geometry + alignment code (`pytest`).

## Usage
```bash
pip install -r requirements.txt
python build_centerline_from_map.py --help
pytest        # runs the offline geometry tests
```

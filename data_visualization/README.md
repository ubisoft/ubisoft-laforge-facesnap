# Multi4D benchmark — visualization scripts

`render_benchmark.py` renders a capture frame from any camera view, either
textured or flat-shaded (no texture/UV needed).

```
pip install numpy Pillow moderngl
```

## Inputs

Each subject folder provides:

- `<subject>/mesh_plys_uvs/<frame>.ply` — mesh geometry + UV for one frame
- `<subject>/cameras/<take>_cameras.json` — camera calibration for the take
- `<subject>/textures/<frame>_*.jpg` — diffuse texture for one frame (textured render only)

## Usage

```
python render_benchmark.py \
    --ply Frame027087.ply \
    --texture Frame027087_u1_v1_diffuse.jpg \
    --cameras-json cameras_take.json \
    --out renders
```

- `--no-texture` — flat-shaded geometry only, no texture/UV needed (omit `--texture`)
- `--camera-names MV-002,MV-004` — limit to specific cameras (default: all)
- `--max-size 900` — cap the longest output side in pixels (default 1024, 0 = native camera resolution)
- `--ambient 1.0` — flat brightness multiplier on the texture

One `<frame>_<camera>.png` per camera is written to `--out` (default: `renders/` next to the PLY).

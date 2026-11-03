# Multi4D Benchmark — Point-to-Surface Evaluation

Reference implementation of the point-to-surface (P2S) error used to develop Multi4D.

A complete demo frame is provided in `demo_data/`, to verify the installation and usage end to end. 

The point-to-surface distance itself is not reimplemented here: it's loaded
directly from a local clone of [TEMPEH](https://github.com/TimoBolkart/TEMPEH)
(Bolkart et al., CVPR 2023) at runtime. See **Installation** below.

To launch the demo:

```bash
python evaluate.py --tempeh-path /path/to/TEMPEH --out ./demo_out
```

## Contents

```
src/
├── evaluate.py            entry point: score one mesh against one scan
├── facesnap_p2s/          
│   ├── scan_io.py         readers: scan (.ply/.abc.cache), mesh (.obj), transform
│   ├── transforms.py      the raw <-> canonical coordinate map
│   ├── scan.py            BenchmarkScan: load, pose, area-sample, clip
│   ├── p2sloss.py         loads the point-to-surface distance class from a local TEMPEH clone
│   └── rendering.py       the per-frame comparison figure
└── demo_data/             one complete frame: scan, transform, mesh
```

## Installation

Requires a CUDA GPU: the surface sampling and the distance query are
[kaolin](https://kaolin.readthedocs.io) kernels.

```bash
conda create -y -n facesnap python=3.10
conda activate facesnap
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install matplotlib
```

You also need a local clone of [TEMPEH](https://github.com/TimoBolkart/TEMPEH)
-- TEMPEH has its own license, see their repo for terms; we don't distribute
or modify their code. 

`p2sloss.py` loads one file out of your clone at
runtime (`utils/point_to_surface_loss.py`) -- you do not need TEMPEH's own
`requirements.txt`. Clone it once, following its own instructions, then
point `--tempeh-path` (or the `TEMPEH_PATH` environment variable) at that
clone's root directory when running `evaluate.py`.

`facesnap_p2s/p2sloss.py` uses p2s loss from TEMPEH with `gmo_sigma=0`
(which makes the robustifier the identity function) and takes the square
root of the returned distances.

_Notes:_

- **Pick the torch/CUDA pair for your GPU, then match kaolin to it.** kaolin
  ships wheels per `(torch, cuda)` pair, listed at
  `https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-<ver>_cu<ver>.html`.
  The combination above is verified on an RTX 5070 (Blackwell, `sm_120`). 

## Evaluation

### 1. Prepare your meshes

For each frame you want scored, you need four things:

| | |
|---|---|
| `--scan` | the ground-truth scan, `<Frame>_scan.ply`, as released |
| `--transform` | that scan's transform, `<Frame>_scan_transform.json`|
| `--mesh` | **your** registered mesh for that frame, `.obj` (tris or quads) |
| `--mesh-scale` | the divisor taking your mesh into the scan's units |

All metrics are computed in the raw coordinate space of the provided photogrammetry scans. Predictions in a different coordinate system must first be aligned to the raw scans.

In the Multi-4D benchmark, we temporarily transform scans into canonical space (using `<Frame>_scan_transform.json`) to simplify facial bounding box selection. Selected points are then transformed back to raw space for metric computation. This step is optional; the bounding box can also be defined directly in raw space.

For quad meshes, triangulation follows [a,b,c,d] → [a,b,c] and [c,d,a]. Different triangulations may affect the metrics.

### 2. Benchmark evaluation

```bash
python evaluate.py \
    --tempeh-path /path/to/TEMPEH \
    --scan       demo_data/Frame044172_scan.ply \
    --transform  demo_data/Frame044172_scan_transform.json \
    --mesh       my_method/Frame044172.obj \
    --mesh-scale 1.0 \
    --clip-bbox  -22 15 -17 6.5 4 13 \
    --method-name MyMethod \
    --out        ./results/Frame044172
```

`--clip-bbox` is the subject's face region `x0 x1 y0 y1 z0 z1` in canonical
space. For Multi4D the bbox was defined per subject. 

Writes two files into `--out`:

- **`scores.json`** — `mean`, `median`, `std` in mm, plus `points_scored` and
  every setting used.
- **`<Frame>.png`** — three panels, *raw scan*, *your mesh*, *P2S error* as a
  heat map over the scored points, with a shared colour bar.

Useful flags: `--no-figure` to skip rendering, `--sample-points` to change the
sample budget (default `300000`, the published setting), `--err-max` to rescale
the colour bar, `--help` for the rest.

**Note on aggregating results:** the provided demo data is intended only to
illustrate the evaluation pipeline and is **not representative of the
performance reported in the paper**. The per-subject results in the paper are
obtained by averaging the per-frame scores across all evaluated frames of each
subject.

## Coordinate frames and the scan transform

There are two frames, and the only reason the second one exists is the bounding
box.

```
raw        the scan's native capture frame.  Arbitrary position and
           orientation -- wherever the head happened to be.
             |
             |   v_aligned = R @ (scale * v_raw) + trans      <- the transform
             v                                                   .json
canonical  A single hand-calibrated axis-aligned box per subject isolates the
           face here -- this space is used only to define that box;
           scores are measured back in raw space.
```

So a run does: pose the scan into canonical space, sample, **clip there**, then
map the surviving points back to raw to measure against the mesh. The transform
is read from the `.json` that ships beside each frame (`scale`, `rot_euler` in
radians composed `Rz @ Ry @ Rx`, `trans`).

**If your bounding box is already expressed in raw coordinates, none of this is
needed** — clip in raw space and skip the round trip. The library supports it
directly: `BenchmarkScan(scan_path, transform=None)` stays in raw throughout.

## Demo data

`demo_data/` holds one complete frame — **Subject8870559 / ROM_1 /
Frame044172**, scored against Topo4D — so the install can be verified offline.
See [`demo_data/README.md`](demo_data/README.md) for the files and the transform
values.


## Notes on convention

**Released `.ply` scans are Z-up; the transform is Y-up.** The scans were
exported from Blender, so `load_scan` rotates them back on load:
`raw = (ply_x, ply_z, -ply_y)`. Skipping this leaves the scan rotated 90° about
X relative to its transform, and the canonical box then clips the wrong region —
giving wrong scores that do not look obviously broken. If you write your own
reader, apply it (`facesnap_p2s.scan_io.ply_axes_to_raw`).


## Citation

This metric is computed using TEMPEH's point-to-surface distance class. If
you use it, please also cite:

```bibtex
@inproceedings{TMPEH:CVPR:2023,
    title = {Instant Multi-View Head Capture through Learnable Registration},
    author = {Bolkart, Timo and Li, Tianye and Black, Michael J.},
    booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {768-779},
    year = {2023}
}
```

## License

Covered by this repository's license, except the TEMPEH code you clone
yourself to run it, which remains under TEMPEH's own license -- see their
repo for terms; we don't distribute or modify their code -- see the
Installation section above.

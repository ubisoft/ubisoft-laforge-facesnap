# Demo Data

This directory contains one sample frame from **Subject8870559 / ROM_1 / Frame044172**, along with its corresponding Topo4D registration, to demonstrate the evaluation pipeline without requiring access to the full dataset.

Run the evaluation from the parent directory:

```bash
python evaluate.py --out ./demo_out
```

## Files

| File | Description |
|---|---|
| `Frame044172_scan.ply` | Ground-truth photogrammetry scan |
| `Frame044172_scan_transform.json` | Raw-to-canonical coordinate transformation |
| `Frame044172_topo4d_face.obj` | Registered mesh produced by Topo4D |

The demo uses the following default parameters:

```bash
--mesh-scale 0.01860523201775983
--clip-bbox -22 15 -17 6.5 4 13
```

## Coordinate System

The provided scan is stored in **Blender's Z-up coordinate system**. The evaluation pipeline converts it to the capture system's Y-up coordinates:

```python
raw_x =  ply_x
raw_y =  ply_z
raw_z = -ply_y
```

This conversion is handled automatically by `facesnap_p2s.scan_io.load_scan()`.

The accompanying JSON file provides the transformation from raw to canonical coordinates:

```python
v_aligned = R @ (scale * v_raw) + trans
```

Here, `R` is constructed from Euler angles in radians (`Rz @ Ry @ Rx`).

The canonical coordinate system is used only to simplify facial bounding box selection. Selected points are transformed back into raw space for metric computation. Alternatively, the facial bounding box can be defined directly in raw space.

See the parent `README.md` for further details.

## Evaluation Results

**The demo frame is intended only to verify the evaluation pipeline and is not representative of the method's overall performance.**

The per-subject results reported in the paper are obtained by averaging per-frame scores across all evaluated frames of each subject. The provided demo frame should not be used independently to report or compare method accuracy.
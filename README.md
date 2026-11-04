Copyright Ubisoft(c) 2026
# Multi4D Benchmark

This GitHub repository hosts the Multi4D benchmark, a new 4D facial performance benchmark on range of motion
based on the Multiface dataset. Multi4D is part of our work FaceSnap: Real-Time Personalized Lightstage Facial Performance Capture.
For more details, see our [Project Page](https://ubisoft-laforge.github.io/character/facesnap/) and [Arxiv](https://arxiv.org/abs/2608.31033).

## Contents
1. [Data details](#data-details)
2. [Structure](#structure)
3. [Installation](#installation)
4. [Visualization and rendering](#visualization-and-rendering)
5. [Metric calculation](#metric-calculation)
6. [Citation](#citation)
7. [License](#license)

## Data details

Multi4D provides per-frame raw mesh scans for 6 subjects performing the same range-of-motion sequence. For each frame alongside the raw mesh, we provide the camera calibration for the capture and diffuse textures.

To obtain the raw meshes used as ground truth, [RealityScan](https://www.realityscan.com/) was used on 38 camera views from the Multiface dataset, covering 360 degrees around the full head.

| Subject | Frames | Cameras | Example vertices | Example triangles |
|---|---|---|---|---|
| Acteur2643814 | 1181 | 36 | 249,984 | 399,930 |
| Acteur8870559 | 1143 | 36 | 258,505 | 399,956 |
| Acteur5372021 | 1140 | 36 | 255,242 | 399,978 |
| Acteur6795937 | 1172 | 35 | 238,011 | 399,940 |
| Acteur7889059 | 1679 | 31 | 254,428 | 399,876 |
| Acteur2914589 | 1689 | 28 | 263,127 | 399,960 |

8,004 frames total. Vertex count varies per frame (~238,000–263,000) since each frame is an independent reconstruction; triangle count is consistently ~399,900–400,000.

Per subject, the data is laid out as:

- `mesh_plys_uvs/<frame>.ply` — raw mesh for one frame, binary PLY, per-vertex UV
- `cameras/*_cameras.json` — camera rig calibration (constant for the whole take)
- `textures/<frame>_u1_v1_diffuse.jpg` — diffuse texture for one frame

## Structure

```
ubisoft-laforge-facesnap/
├── metrics/src/            point-to-surface (P2S) evaluation pipeline
│   ├── evaluate.py         entry point: score one mesh against one scan
│   ├── facesnap_p2s/       library code (scan I/O, transforms, P2S distance, rendering)
│   └── demo_data/          one complete sample frame, for an offline install check
└── data_visualization/     scripts for viewing a capture frame
    └── render_benchmark.py       textured/flat-shaded renders from any camera
```

## Installation

This repo has two independent, lightweight toolkits — install only what you need:

- **Visualization** (`data_visualization/`): `pip install numpy Pillow moderngl`
- **P2S metric** (`metrics/src/`): requires a CUDA GPU and `kaolin` — see [`metrics/src/README.md`](metrics/src/README.md) for exact setup

### Data download

Coming soon.

## Visualization and rendering

Render any raw mesh from any camera view, textured or flat-shaded. 

See [`data_visualization/README.md`](data_visualization/README.md) for setup, example commands, and flags.

## Metric calculation

Reference implementation of the point-to-surface (P2S) error used to develop Multi4D.

To sanity-check the pipeline on the provided demo frame:

```bash
cd metrics/src
python evaluate.py --out ./demo_out
```

See [`metrics/src/README.md`](metrics/src/README.md) for the full installation steps (requires a CUDA GPU and `kaolin`), evaluating your own meshes, and coordinate-frame conventions.

## Citation

If you use Multi4D, please cite our paper and the [Multiface technical report](https://arxiv.org/abs/2207.11243):

```bibtex
@article{hussain2026facesnap,
  title={FaceSnap: Real-Time Personalized Lightstage Facial Performance Capture},
  author={Hussain, Rukhshanda and Artru, No{\'e} and Got, Emeline and Hafemann, Luiz Gustavo and Messier, Alexandre and Dearlove, Brandon and Cruz, Rafael MO and Dib, Abdallah and Granger, Eric},
  journal={arXiv preprint arXiv:2608.31033},
  year={2026}
}
@inproceedings{wuu2022multiface,
  title={Multiface: A Dataset for Neural Face Rendering},
  author = {Wuu, Cheng-hsin and Zheng, Ningyuan and Ardisson, Scott and Bali, Rohan and Belko, Danielle and Brockmeyer, Eric and Evans, Lucas and Godisart, Timothy and Ha, Hyowon and Huang, Xuhua and Hypes, Alexander and Koska, Taylor and Krenn, Steven and Lombardi, Stephen and Luo, Xiaomin and McPhail, Kevyn and Millerschoen, Laura and Perdoch, Michal and Pitts, Mark and Richard, Alexander and Saragih, Jason and Saragih, Junko and Shiratori, Takaaki and Simon, Tomas and Stewart, Matt and Trimble, Autumn and Weng, Xinshuo and Whitewolf, David and Wu, Chenglei and Yu, Shoou-I and Sheikh, Yaser},
  booktitle={arXiv},
  year={2022},
  doi = {10.48550/ARXIV.2207.11243},
  url = {https://arxiv.org/abs/2207.11243}
}
```

## License

This repository and data is released under the [Ubisoft Machine Learning License (Research-Only - Copyleft)](LICENSE.txt): use, reproduction, and distribution are permitted for academic/research purposes only, with no commercial use.

Multi4D's ground-truth meshes are derived from the [Multiface](https://arxiv.org/abs/2207.11243) dataset ([GitHub repository](https://github.com/facebookresearch/multiface)), which carries its own licensing terms — see the Multiface repository for details. Data and code derived from Multiface in this project may only be used for research purposes, consistent with both licenses.


Copyright Ubisoft(c) 2026

# Model provenance and attribution

## Trained checkpoints

`isl_transformer.pt`, `isl_bilstm.pt` and `classes.json` in this directory
were trained by the SIGN-X project on a 28-class subset of the **INCLUDE**
dataset and are distributed under the same terms that apply to that data.

| | |
|---|---|
| Training data | INCLUDE — Greetings, Pronouns, Colours categories (28 classes; 370 train / 60 validation / 150 test videos, the dataset's official split) |
| Dataset authors | Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar, Mitesh M. Khapra (IIT Madras / AI4Bharat) |
| Dataset license | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Sources | Videos: https://zenodo.org/records/4010759 · Metadata/split: https://huggingface.co/datasets/ai4bharat/INCLUDE |
| Changes made | Videos were not redistributed. MediaPipe Holistic landmarks were extracted from them, normalized into 330-dimensional per-frame features, and used to train the models. |
| Training records | `backend/experiments/isl_include_subset28_{transformer,bilstm}.yaml`; held-out results in `backend/training/logs/isl_include_subset28/evaluation_results_*.json` (test accuracy 0.720 Transformer, 0.727 BiLSTM) |
| Integrity | SHA-256 pinned in `backend/tests/test_model_integrity.py` |

Please cite the dataset when using these checkpoints:

> Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar, and Mitesh Khapra.
> 2020. INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.
> In *Proceedings of the 28th ACM International Conference on Multimedia
> (MM '20)*. https://doi.org/10.1145/3394171.3413528

The INCLUDE authors do not endorse SIGN-X or these models. The videos were
recorded in Chennai, Tamil Nadu, and do not represent all regional variation
in ISL; see `docs/dataset.md` for known limitations.

## MediaPipe HolisticLandmarker

`mediapipe/holistic_landmarker.task` is not part of this repository. It is
Google's MediaPipe Tasks model, downloaded by `scripts/download_models.*`
and by `backend/Dockerfile` (which verifies its SHA-256), and is governed by
Google's own license for that asset.

## Source code

The SIGN-X source code is MIT-licensed (see `LICENSE`). That license does not
cover the dataset or the third-party model asset described above.

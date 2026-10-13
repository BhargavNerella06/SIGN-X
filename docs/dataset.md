# SIGN-X — Dataset Guide

SIGN-X does not bundle or auto-download any dataset. This document lists
**real, verifiable** Indian Sign Language (ISL) resources so you can obtain
one yourself, plus the exact layout the training pipeline expects.

> ISL is not ASL. Do not substitute an American Sign Language dataset and
> expect ISL-relevant results — the two sign languages are unrelated.

## Stage 2: the INCLUDE subset actually used

SIGN-X's first real-data experiment (`backend/experiments/isl_include_subset28.yaml`)
uses **INCLUDE** (AI4Bharat / IIT Madras), restricted to 3 of its 15
categories: **Greetings, Pronouns, Colours — 28 classes total**.

| | |
|---|---|
| Dataset | INCLUDE (AI4Bharat / IIT Madras) |
| License | **CC-BY-4.0** (verified via the HuggingFace dataset card and the Zenodo record's metadata) |
| Metadata source | https://huggingface.co/datasets/ai4bharat/INCLUDE (parquet: `parent_label`, `label`, `video_path`, `include_50`; official train/val/test split) |
| Video source | https://zenodo.org/records/4010759 (per-category zip archives, ~57 GB total across all 15 categories) |
| Categories used | Greetings (9 classes), Pronouns (8 classes), Colours (11 classes) |
| Sample count | **766 selected metadata rows → 580 unique videos** after de-duplication (see "Duplicate metadata rows" below). The 580 are split **370 train / 60 validation / 150 test**, 20–22 unique videos per class. The earlier "253 / 231 / 282 samples" figures counted metadata rows, not unique videos. |
| Why this subset | Falls inside the project's "start with 20-30 signs" guidance (Section 2), every class has at least 20 unique videos, and its zips total ~7.9 GB vs. the full dataset's ~57 GB — the only categories practical to download in this build environment (see bandwidth note below). |
| Signer metadata | **Not published** in the HuggingFace metadata — signer-independent splitting could not be verified. The dataset's own official train/val/test split is used as-is rather than re-shuffled (see "Data leakage" below). |
| Citation | Sridhar, Ganesan, Kumar, Khapra. "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition." ACM Multimedia 2020. https://doi.org/10.1145/3394171.3413528 |

### Exact download commands used

```powershell
# 1. Metadata (tiny, ~115 KB total)
cd backend/datasets
mkdir _include_meta; cd _include_meta
curl -L -o train.parquet https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main/data/train-00000-of-00001.parquet
curl -L -o val.parquet   https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main/data/val-00000-of-00001.parquet
curl -L -o test.parquet  https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main/data/test-00000-of-00001.parquet

# 2. Videos for the 3 chosen categories (~7.9 GB total)
cd ../_include_zips
curl -L -o Greetings_1of2.zip https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content
curl -L -o Greetings_2of2.zip https://zenodo.org/api/records/4010759/files/Greetings_2of2.zip/content
curl -L -o Pronouns_1of2.zip  https://zenodo.org/api/records/4010759/files/Pronouns_1of2.zip/content
curl -L -o Pronouns_2of2.zip  https://zenodo.org/api/records/4010759/files/Pronouns_2of2.zip/content
curl -L -o Colours_1of2.zip   https://zenodo.org/api/records/4010759/files/Colours_1of2.zip/content
curl -L -o Colours_2of2.zip   https://zenodo.org/api/records/4010759/files/Colours_2of2.zip/content

# 3. Extract (each zip contains its own top-level category folder, so they merge safely)
mkdir extracted
for f in *.zip; do unzip -o "$f" -d extracted; done
```

**Recommended instead of step 2**: `bash scripts/download_include_subset.sh`
(Git Bash) downloads the same six zips with HTTP-range resume and only
marks a file complete once its MD5 matches the checksum published by the
Zenodo record API. Safe to re-run after an interruption. A plain `curl`
download that drops mid-transfer leaves a truncated zip behind that looks
"finished"; this happened once while building this project.

**Bandwidth note**: download speed from Zenodo in this build environment
varied from ~0.3-0.5 MB/s up to ~1 MB/s per stream. The resumable script
downloaded all six zips (7.87 GB) in about 30 minutes using 3 parallel
streams. This is an environment limitation, not a Zenodo or dataset issue.

### Data leakage prevention

- The adapter (`backend/training/datasets/include.py`) reads INCLUDE's own
  **official** `train`/`val`/`test` split from the published metadata and
  does **not** re-shuffle it — the project's split-integrity rule (Section 4)
  takes precedence over convenience.
- Signer identity is not part of the published metadata, so we could not
  verify signer-independent separation between splits ourselves. This is
  stated here explicitly rather than assumed.

### Duplicate metadata rows

The published HuggingFace metadata lists **5,250 rows but only 4,257 unique
`video_path`s**, and 424 of the duplicate rows put the same video in a
*different* split (e.g. one clip in both `train` and `test`). Used as-is,
that would train on test videos and inflate test accuracy.

In the 28-class subset this is **766 metadata rows → 580 unique videos**;
before the fix, 55 test videos were also in train. The adapter
(`IncludeAdapter._deduplicate`) keeps each video exactly once, in the
highest-priority split it appears in, with priority **test > validation >
train**. A video listed in any held-out split is therefore never trained on,
and the adapter logs how many rows it dropped.

One side effect: a few classes lose more training clips than others
(IT 7, HELLO 9, THANK_YOU 9 train videos, against 14–15 for most classes).

### Unique landmark filenames

Each landmark file is named `<readable slug>__<first 10 hex chars of
SHA-1(source video_path)>.npy`. The name is deterministic and derived only
from the source path. The hash is needed because INCLUDE contains
file pairs such as `MVI_0057.MOV` / `MVI_0057_.MOV` whose readable slugs
are identical. `build_dataset_from_adapter` also refuses to run if two
samples would map to the same output file (compared case-insensitively,
as on Windows).

### Landmark extraction: one fresh MediaPipe instance per video

MediaPipe's `HolisticLandmarker` keeps internal state between calls, even
in IMAGE mode. Running the same video twice through one instance gave
different hand landmarks (~5% of hand-detection flags flipped), while two
fresh instances gave bit-identical output. Extraction therefore creates,
uses and closes a new instance for every video
(`training.extract_landmarks.extract_video_fresh`), so a video's features
never depend on which videos were processed before it.

## Temporal preprocessing (clip → 32 frames)

INCLUDE clips are 33–109 frames long (median 60, ~2.5 s at 25 fps), so
every clip is longer than the models' 32-frame input. The loader
(`training/dataset.py`) and the realtime buffer
(`app/inference/temporal_buffer.py`) both use one shared function,
`app.vision.preprocessing.resample_or_pad`:

- **Longer than 32 frames:** 32 frames are sampled at uniformly spaced
  positions across the *entire* clip, always including the first and last
  frame, in chronological order and deterministically.
- **Exactly 32:** unchanged.
- **Shorter than 32:** zero-padded at the start, with the padding masked
  out (`mask = False`).

Previously, long clips were truncated to their *last* 32 frames. That kept
a median of only 49% of each clip's hand movement, and less than half for
314 of 580 clips. Augmentation (temporal crop, frame dropping) runs
*before* resampling.

In realtime, the buffer keeps `realtime_window_frames` raw frames (default
32, about 3.2 s at the frontend's default 10 fps capture) and passes them
through the same function to produce the model's `sequence_length` (32)
frames. Whether that window is the right size for live signing has **not
yet been verified on a real webcam**.

## Other recommended ISL datasets

### 1. INCLUDE (AI4Bharat / IIT Madras) — full dataset
- **What**: 4,287 isolated-sign videos across all 263 word signs, 15
  categories, recorded by experienced signers in Chennai, Tamil Nadu (Stage
  2 above uses only 3 of these 15 categories / 28 of these 263 classes). An
  `INCLUDE-50` subset (50 signs, `include_50` column) also exists.
- **Paper**: "INCLUDE: A Large Scale Dataset for Indian Sign Language
  Recognition", ACM Multimedia 2020 — https://dl.acm.org/doi/10.1145/3394171.3413528
- **Code/models**: https://github.com/AI4Bharat/INCLUDE
- **Data**: https://huggingface.co/datasets/ai4bharat/INCLUDE
- **License**: CC-BY-4.0.
- **Best for**: isolated-sign classification — closest match to Stage 1–6
  of this project. Extending Stage 2 to more/all 263 classes means
  downloading more category zips from the same Zenodo record and adding
  them to `subset_classes` in an experiment config.

### 2. ISL-CSLTR (Indian Sign Language Continuous Sign Language Translation)
- **What**: sentence-level continuous ISL corpus — 700 annotated videos,
  100 spoken-language sentences, 7 signers (per the dataset's public
  Mendeley description), contributed by Navajeevan Residential School for
  the Deaf and SASTRA Deemed University. Useful for the continuous-
  recognition stage (Section 14) once isolated recognition works.
- **Mendeley Data**: https://data.mendeley.com/datasets/kcmpdxky7p/1 (single
  `ISL_CSLRT_Corpus.zip`, ~8.9 GB, public direct-download link, no login required)
- **Mirror on Kaggle**: https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset
- **License**: CC-BY-4.0 (per the Mendeley record's `data_licence` field).
- **Best for**: continuous sentence-level recognition (Stage 7+).
- **Status in this repo**: `backend/training/datasets/isl_csltr.py` is a
  real adapter *skeleton* for this dataset, but it was **not downloaded or
  exercised** in Stage 2 — see that file's docstring for exactly what is
  and isn't verified. Isolated-sign recognition (the INCLUDE subset above)
  was the Stage 2 priority per the project's own staged architecture.

### 3. Kaggle — "Indian Sign Language (ISL)"
- **What**: image-based static ISL alphabet/digit dataset (single images,
  not temporal sequences).
- **Link**: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
- **Note**: because SIGN-X is a *temporal* sign-language recognizer (Section 2),
  this dataset alone is not directly usable for word-level sign training —
  it's useful only for a static fingerspelling sub-component, not covered
  by the current architecture.

### 4. Kaggle — "Indian Sign Language words with Landmarks"
- **What**: ISL word-sign videos with landmark coordinates already extracted.
- **Link**: https://www.kaggle.com/datasets/kaushikyh/indian-sign-language-words-with-landmarks
- **Note**: if you use this, you'll need to remap its landmark format to
  SIGN-X's feature layout (`app/vision/feature_extractor.py`) rather than
  running `training/extract_landmarks.py`, since the landmarks were
  extracted with a different pipeline/point ordering than MediaPipe's
  HolisticLandmarker.

Always re-check each dataset's current license/usage terms yourself before
training or publishing a model on it — terms can change and this document
is not legal advice.

## Adapter-driven workflow (for datasets with official metadata, like INCLUDE)

Datasets that publish their own class/split metadata (like INCLUDE's
parquet files) don't go through the manual `datasets/raw/<CLASS>/` +
`prepare_dataset.py` random-split workflow below — instead, a
`training.datasets.*` adapter (`backend/training/datasets/`) reads that
metadata directly and resolves it against the downloaded video files,
preserving the dataset's own official split (de-duplicated as described
above). The command sequence for Stage 2 is below. Steps 1–4 have been
run; training and evaluation (step 5) have **not** been run yet, and no
model results exist.

```powershell
cd backend

# 1. Visual validation (Section 7) — confirms MediaPipe is actually detecting hands/pose on real footage
python -m training.visualize_landmarks --config experiments/isl_include_subset28.yaml --num_videos 8

# 2. Build the train/validation/test landmark dataset (one fresh MediaPipe instance per video;
#    resumable — existing .npy files are kept, delete the output folder for a clean rebuild).
#    Takes roughly 45 min on CPU for the 580 videos.
python -m training.build_dataset_from_adapter --config experiments/isl_include_subset28.yaml

# 3. Data quality report (Section 6) from the extracted landmarks: streams one file
#    at a time, takes seconds. Omit --from_landmarks to re-run MediaPipe on every
#    raw video instead (slow; about as expensive as step 2).
python -m training.dataset_report --config experiments/isl_include_subset28.yaml --from_landmarks

# 4. (verification) Run the test suite
python -m pytest -q

# 5. Train + evaluate — BiLSTM baseline, then Transformer
python -m training.train    --config experiments/isl_include_subset28_bilstm.yaml
python -m training.evaluate --config experiments/isl_include_subset28_bilstm.yaml
python -m training.train    --config experiments/isl_include_subset28.yaml
python -m training.evaluate --config experiments/isl_include_subset28.yaml
```

`training/datasets/registry.py` maps an experiment config's `dataset.adapter:`
value (`include` or `isl_csltr`) to its adapter class. Adding a new dataset
means writing one more `DatasetAdapter` subclass, not touching any other
part of the training or inference pipeline.

## Expected layout for `training/extract_landmarks.py`

Whichever dataset you choose, reorganize the raw video files into:

```
backend/datasets/raw/<CLASS_NAME>/<video files: .mp4/.avi/.mov/.mkv>
```

`<CLASS_NAME>` should match (or be mapped to) the vocabulary in
`app/config.py`'s `vocabulary` setting, e.g. `HELLO`, `THANK_YOU`, `YES`, ...
You are not required to use the default 20-word vocabulary — set
`SIGNX_VOCABULARY` (or edit `training/config.yaml`) to whatever class list
your chosen dataset actually provides.

## No dataset yet?

Run `python -m datasets.sample.generate_sample_dataset` to generate a small
**synthetic** landmark dataset (see `backend/datasets/README.md`). It
contains no real sign data and exists solely to verify the training
pipeline runs end to end before you plug in real data.

## Recommended starting scale

Per Section 13, start with the small ~20-word demo vocabulary already
configured, not hundreds of classes. Once isolated recognition is validated
on that scale, extend the vocabulary and move to continuous recognition
(Section 14) using a sentence-level dataset like ISL-CSLTR.

# SIGN-X Datasets

This directory holds the ISL dataset once you've obtained one — **no dataset
is bundled or auto-downloaded**, per project policy (see `docs/dataset.md`
for recommended sources and their licenses).

## Expected layout

```
datasets/
├── raw/                          # you place the downloaded dataset here
│   ├── HELLO/
│   │   ├── clip_001.mp4
│   │   └── clip_002.mp4
│   ├── THANK_YOU/
│   │   └── ...
│   └── <CLASS_NAME>/*.mp4
│
├── processed/<dataset_name>/     # produced by training/extract_landmarks.py
│   └── <CLASS_NAME>/*.npy
│
└── <dataset_name>/                # produced by training/prepare_dataset.py
    ├── class_mapping.json
    ├── train/<CLASS_NAME>/*.npy
    ├── validation/<CLASS_NAME>/*.npy
    └── test/<CLASS_NAME>/*.npy
```

## Pipeline

```powershell
# 1. Place your downloaded dataset under datasets/raw/<CLASS_NAME>/*.mp4
# 2. Extract MediaPipe landmark sequences
python -m training.extract_landmarks --dataset_name isl_demo

# 3. Split into train/validation/test
python -m training.prepare_dataset --dataset_name isl_demo

# 4. Train
python -m training.train --config training/config.yaml

# 5. Evaluate
python -m training.evaluate --config training/config.yaml
```

## No dataset yet?

`datasets/sample/generate_sample_dataset.py` generates a small **synthetic**
landmark dataset so you can exercise the full training/evaluation pipeline
immediately. It contains no real sign language data whatsoever — a model
trained on it learns nothing about actual ISL and must never be used to
report real accuracy numbers. It exists purely to prove the pipeline works
end to end before you have real data:

```powershell
python -m datasets.sample.generate_sample_dataset --dataset_name isl_demo --samples_per_class 20
python -m training.prepare_dataset --dataset_name isl_demo
python -m training.train --config training/config.yaml
```

See `docs/dataset.md` for real ISL dataset recommendations.

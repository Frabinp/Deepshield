# DeepShield

DeepShield is a deepfake detection project with three branches:

- Image detector on face crops
- Video detector on frame clips
- Audio detector on spectrograms

The current workspace supports two aligned fusion modes:

- Image + video fusion on FaceForensics++ data
- Standalone audio detection on ASVspoof 2019 LA data

Why not 3-way audio-video-image fusion right now?

- The FaceForensics++ videos in this workspace do not contain usable audio.
- `data/preprocessing/extract_audio.py` confirms `No audio / silent: 800`.
- Because of that, the aligned fusion pipeline automatically falls back to `image + video`.

## Folder layout

- `data/processed/faces` : face crops from FaceForensics++
- `data/processed/frames` : frame clips from FaceForensics++
- `data/processed/spectrograms` : ASVspoof spectrograms for the standalone audio branch
- `data/processed/spectrograms_faceforensics` : extracted FaceForensics spectrograms (empty if source videos are silent)
- `data/processed/manifests/faceforensics_multimodal.json` : aligned manifest for fusion/evaluation

## Main scripts

- `data/preprocessing/faceforensics_ids.py`
- `data/preprocessing/migrate_faceforensics_ids.py`
- `data/preprocessing/extract_faces.py`
- `data/preprocessing/extract_frames.py`
- `data/preprocessing/extract_audio.py`
- `data/preprocessing/build_faceforensics_manifest.py`
- `train/train_image.py`
- `train/train_video.py`
- `train/train_audio.py`
- `train/train_fusion.py`
- `eval/evaluate.py`
- `eval/ablation.py`
- `app/gradio_app.py`

## Recommended run order

1. One-time migration for existing processed FaceForensics files:

```powershell
venv\Scripts\python.exe data\preprocessing\migrate_faceforensics_ids.py
```

2. Build visual preprocessing:

```powershell
venv\Scripts\python.exe data\preprocessing\extract_faces.py
venv\Scripts\python.exe data\preprocessing\extract_frames.py
```

3. Try FaceForensics audio extraction and build the manifest:

```powershell
venv\Scripts\python.exe data\preprocessing\extract_audio.py
venv\Scripts\python.exe data\preprocessing\build_faceforensics_manifest.py
```

4. Train individual branches:

```powershell
venv\Scripts\python.exe train\train_image.py
venv\Scripts\python.exe train\train_video.py
venv\Scripts\python.exe train\train_audio.py
```

5. Train fusion:

```powershell
venv\Scripts\python.exe train\train_fusion.py
```

6. Evaluate:

```powershell
venv\Scripts\python.exe eval\evaluate.py
venv\Scripts\python.exe eval\ablation.py
```

7. Launch demo:

```powershell
venv\Scripts\python.exe app\gradio_app.py
```

## Important implementation notes

- Fusion checkpoint loading is backward-compatible with old raw `state_dict` checkpoints.
- New fusion checkpoints save metadata including `active_modalities`.
- If the manifest has aligned audio, fusion will use `image + video + audio`.
- If the manifest only has aligned visual data, fusion will use `image + video`.
- The audio app tab still uses the standalone ASVspoof-trained audio branch.
- FaceForensics preprocessing now uses stable filename-based sample IDs such as `youtube_033` or `actors_01_hugging_happy`.
- When you add new raw FaceForensics videos, rerun visual preprocessing and rebuild the manifest. New files will no longer shift existing sample IDs.

## Current status in this workspace

- Aligned FaceForensics image+video samples: 800
- Aligned FaceForensics image+video+audio samples: 0
- Result: fusion training currently runs in `image + video` mode
- Stable FaceForensics IDs are active in processed visual assets and the multimodal manifest

## Current baseline

Latest local evaluation report:

- `eval/results/latest_evaluation.json`
- Generated on: `2026-03-13`

Current metrics:

- Image AUC: `0.6154`
- Audio AUC: `0.9339`
- Video AUC: `0.6744`
- Fusion AUC: `0.6398`

Current calibrated thresholds from the deployed checkpoints:

- Image threshold: `0.2161`
- Audio threshold: `0.0159`
- Video threshold: `0.3232`
- Fusion threshold: `0.2449`

Current interpretation:

- Audio is the strongest branch.
- Video is the strongest visual branch.
- Fusion is working, but it is not yet outperforming the standalone video model.
- The visual bottleneck is now data quality and model capacity, not pipeline stability.

## Remaining work

Highest priority:

1. Add more real FaceForensics-style visual videos to reduce the current fake-heavy imbalance.
2. Rebuild visual preprocessing and the aligned manifest after new real videos are added.
3. Retrain image, video, and fusion on the expanded visual set.

Next best improvements:

1. Upgrade `models/video_model.py` from the current EfficientNet-B0 + LSTM baseline to a stronger temporal-attention design.
2. Run multi-seed visual training sweeps and keep only the best measured checkpoints.
3. Do error analysis on `eval/results/latest_predictions.csv` and remove weak crops, bad frames, and low-quality source videos.

Still blocked by data:

1. True `image + video + audio` fusion is not available with the current local FaceForensics data because the videos are silent.
2. To unlock full 3-way fusion, a dataset with aligned visual and usable audio is needed.

## Recommended next run after adding more real videos

```powershell
venv\Scripts\python.exe data\preprocessing\extract_faces.py
venv\Scripts\python.exe data\preprocessing\extract_frames.py
venv\Scripts\python.exe data\preprocessing\build_faceforensics_manifest.py
venv\Scripts\python.exe data\preprocessing\audit_splits.py
venv\Scripts\python.exe train\train_image.py
venv\Scripts\python.exe train\train_video.py
venv\Scripts\python.exe train\train_fusion.py
venv\Scripts\python.exe eval\evaluate.py
```

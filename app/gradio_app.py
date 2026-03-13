"""
gradio_app.py - DeepShield demo.
Multimodal deepfake detector: image, video, audio.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_env import configure_wandb_environment

configure_wandb_environment()

from models.audio_model import AudioDetector
from models.checkpoint_utils import (
    get_active_modalities,
    get_decision_threshold,
    load_fusion_models,
    load_module_checkpoint,
    mask_fusion_features,
)
from models.fusion_model import FusionModel
from models.image_model import ImageDetector
from models.video_model import VideoDetector

try:
    from explain.gradcam import GradCAM, overlay_heatmap, TRANSFORM as GRADCAM_TRANSFORM
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


try:
    from facenet_pytorch import MTCNN as _MTCNN

    _mtcnn = _MTCNN(keep_all=False, device="cpu", post_process=False)
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False
    print("MTCNN not available, using OpenCV face detection")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

N_FRAMES = 16
SAMPLE_RATE = 16000
N_MELS = 128
IMG_CKPT = "checkpoints/best_image.pt"
AUD_CKPT = "checkpoints/best_audio.pt"
VID_CKPT = "checkpoints/best_video.pt"
FUS_CKPT = "checkpoints/best_fusion.pt"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_latest_metric_summary():
    latest_report = Path("eval/results/latest_evaluation.json")
    if not latest_report.exists():
        return "Latest evaluation metrics not available yet."

    try:
        with open(latest_report, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return "Latest evaluation metrics could not be loaded."

    parts = []
    results = payload.get("results", {})
    for key in ("image", "audio", "video", "fusion"):
        auc = results.get(key, {}).get("auc")
        if isinstance(auc, (int, float)):
            parts.append(f"{key.title()} AUC={auc:.4f}")
    return " | ".join(parts) if parts else "Latest evaluation metrics not available yet."


def load_models():
    img_model = ImageDetector().to(DEVICE)
    img_checkpoint = load_module_checkpoint(img_model, IMG_CKPT, map_location=DEVICE)
    img_model.eval()

    aud_model = AudioDetector().to(DEVICE)
    aud_checkpoint = load_module_checkpoint(aud_model, AUD_CKPT, map_location=DEVICE)
    aud_model.eval()

    vid_model = VideoDetector().to(DEVICE)
    vid_checkpoint = load_module_checkpoint(vid_model, VID_CKPT, map_location=DEVICE)
    vid_model.eval()

    fusion_image_model = ImageDetector().to(DEVICE)
    fusion_model = FusionModel().to(DEVICE)
    fusion_checkpoint = load_fusion_models(
        fusion_image_model,
        fusion_model,
        FUS_CKPT,
        image_checkpoint_path=IMG_CKPT,
        map_location=DEVICE,
    )
    fusion_image_model.eval()
    fusion_model.eval()

    thresholds = {
        "image": get_decision_threshold(img_checkpoint, default=0.5),
        "audio": get_decision_threshold(aud_checkpoint, default=0.5),
        "video": get_decision_threshold(vid_checkpoint, default=0.5),
        "fusion": get_decision_threshold(fusion_checkpoint, default=0.5),
    }
    active_modalities = get_active_modalities(fusion_checkpoint, default=("image",))
    return (
        img_model,
        aud_model,
        vid_model,
        fusion_image_model,
        fusion_model,
        active_modalities,
        thresholds,
    )


(
    image_model,
    audio_model,
    video_model,
    fusion_image_model,
    fusion_model,
    FUSION_ACTIVE_MODALITIES,
    DECISION_THRESHOLDS,
) = load_models()
print("All models loaded.")


def crop_face(pil_img):
    if MTCNN_AVAILABLE:
        try:
            boxes, _ = _mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(pil_img.width, x2 + margin)
                y2 = min(pil_img.height, y2 + margin)
                return pil_img.crop((x1, y1, x2, y2)), "face cropped (MTCNN)"
        except Exception:
            pass

    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(pil_img.width, x + w + margin)
        y2 = min(pil_img.height, y + h + margin)
        return pil_img.crop((x1, y1, x2, y2)), "face cropped (OpenCV)"

    return pil_img, "no face detected - using full image"


def predict_image(pil_img, model=None):
    model = model or image_model
    image_tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features, logit = model(image_tensor)
    probability = torch.sigmoid(logit).item()
    return probability, features


def predict_audio(audio_path):
    try:
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        target_len = SAMPLE_RATE * 4
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8)).resize((128, 128))
        mel_arr = np.array(mel_img, dtype=np.float32) / 255.0
        mel_tensor = torch.from_numpy(mel_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features, logit = audio_model(mel_tensor)
        probability = torch.sigmoid(logit).item()
        return probability, features
    except Exception as exc:
        print(f"Audio error: {exc}")
        return None, torch.zeros(1, 512, device=DEVICE)


def extract_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total_frames - 1, 0), N_FRAMES, dtype=int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(TRANSFORM(Image.fromarray(frame)))
        else:
            frames.append(torch.zeros(3, 224, 224))
    capture.release()
    return torch.stack(frames).unsqueeze(0).to(DEVICE)


def extract_audio_from_video(video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_audio = tmp_file.name

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-t",
            "4",
            temp_audio,
        ],
        capture_output=True,
    )
    if result.returncode == 0 and os.path.exists(temp_audio):
        return temp_audio
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    return None


def format_result(probability, threshold):
    is_fake = probability > threshold
    label = "FAKE" if is_fake else "REAL"
    confidence = probability if is_fake else 1 - probability
    return label, confidence


def predict_image_only(image):
    if image is None:
        return "No image provided", {}

    pil_img = Image.fromarray(image).convert("RGB")
    pil_img, face_note = crop_face(pil_img)
    probability, _ = predict_image(pil_img)
    label, confidence = format_result(probability, DECISION_THRESHOLDS["image"])

    # Grad-CAM overlay
    gradcam_overlay = None
    if GRADCAM_AVAILABLE:
        try:
            target_layer = image_model.backbone.conv_head
            cam = GradCAM(image_model, target_layer)
            input_tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            target_class = 1 if probability > DECISION_THRESHOLDS["image"] else 0
            heatmap = cam(input_tensor, target_class=target_class)
            resized_img = pil_img.resize((224, 224))
            gradcam_overlay = overlay_heatmap(resized_img, heatmap)
        except Exception:
            pass

    result_text = f"{label}  (confidence: {confidence:.1%}) [{face_note}]"
    scores = {
        "Fake": float(probability),
        "Real": float(1 - probability),
        "Threshold": float(DECISION_THRESHOLDS["image"]),
    }
    return result_text, scores, gradcam_overlay


def predict_video_full(video):
    if video is None:
        return "No video provided", {}

    frames = extract_frames(video)
    frames_flip = torch.flip(frames, dims=[4])
    with torch.no_grad():
        video_features, video_logit = video_model(frames)
        _, video_logit_flip = video_model(frames_flip)
    video_prob = 0.5 * (
        torch.sigmoid(video_logit).item() + torch.sigmoid(video_logit_flip).item()
    )

    capture = cv2.VideoCapture(video)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ok, frame = capture.read()
    capture.release()
    if ok:
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_frame, _ = crop_face(pil_frame)
        image_prob, _ = predict_image(pil_frame, model=image_model)
        _, fusion_image_features = predict_image(pil_frame, model=fusion_image_model)
    else:
        image_prob = 0.5
        fusion_image_features = torch.zeros(1, 512, device=DEVICE)

    audio_path = extract_audio_from_video(video)
    try:
        if audio_path:
            audio_prob, audio_features = predict_audio(audio_path)
        else:
            audio_prob = None
            audio_features = torch.zeros(1, 512, device=DEVICE)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    fusion_image_features, fusion_video_features, fusion_audio_features = mask_fusion_features(
        fusion_image_features,
        video_features,
        audio_features,
        FUSION_ACTIVE_MODALITIES,
    )
    with torch.no_grad():
        fusion_out = fusion_model(
            fusion_image_features,
            fusion_video_features,
            fusion_audio_features,
        )
    fusion_prob = torch.sigmoid(fusion_out["logit"]).item()
    fusion_uncertainty = fusion_out["uncertainty"].item()

    label, confidence = format_result(video_prob, DECISION_THRESHOLDS["video"])
    details = {
        "Video (Fake)": round(video_prob, 3),
        "Image (Fake)": round(image_prob, 3),
        "Fusion (Fake)": round(fusion_prob, 3),
        "Fusion Uncertainty": round(fusion_uncertainty, 4),
        "Video Threshold": round(DECISION_THRESHOLDS["video"], 3),
        "Image Threshold": round(DECISION_THRESHOLDS["image"], 3),
        "Fusion Threshold": round(DECISION_THRESHOLDS["fusion"], 3),
        "Primary Decision": "video_tta",
    }
    if audio_prob is not None:
        details["Audio (Fake)"] = round(audio_prob, 3)
        details["Audio Threshold"] = round(DECISION_THRESHOLDS["audio"], 3)

    return f"{label}  (confidence: {confidence:.1%})", details


def predict_audio_only(audio):
    if audio is None:
        return "No audio provided", {}

    probability, _ = predict_audio(audio)
    if probability is None:
        return "Audio processing failed", {}

    label, confidence = format_result(probability, DECISION_THRESHOLDS["audio"])
    return f"{label}  (confidence: {confidence:.1%})", {
        "Fake": float(probability),
        "Real": float(1 - probability),
        "Threshold": float(DECISION_THRESHOLDS["audio"]),
    }


with gr.Blocks(title="DeepShield") as demo:
    gr.Markdown(
        """
        # DeepShield - Multimodal Deepfake Detector
        **BTech Final Year Project** | EfficientNet-B0 + LSTM + Late Fusion
        Upload an image, video, or audio file to detect deepfakes.
        """
    )

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="numpy")
                with gr.Column():
                    image_label = gr.Textbox(label="Result")
                    image_scores = gr.Label(label="Confidence Scores")
            image_button = gr.Button("Detect", variant="primary")
            image_button.click(
                predict_image_only,
                inputs=image_input,
                outputs=[image_label, image_scores],
            )

        with gr.Tab("Video"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                with gr.Column():
                    video_label = gr.Textbox(label="Result")
                    video_scores = gr.Label(label="Per-Modality Scores")
            video_button = gr.Button("Detect", variant="primary")
            video_button.click(
                predict_video_full,
                inputs=video_input,
                outputs=[video_label, video_scores],
            )

        with gr.Tab("Audio"):
            with gr.Row():
                audio_input = gr.Audio(label="Upload Audio", type="filepath")
                with gr.Column():
                    audio_label = gr.Textbox(label="Result")
                    audio_scores = gr.Label(label="Confidence Scores")
            audio_button = gr.Button("Detect", variant="primary")
            audio_button.click(
                predict_audio_only,
                inputs=audio_input,
                outputs=[audio_label, audio_scores],
            )

    gr.Markdown(
        """
        ---
        **Datasets:** FaceForensics++ (visual) | ASVspoof 2019 LA (audio)
        """
    )
    gr.Markdown(f"**Latest local metrics:** {load_latest_metric_summary()}")

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())

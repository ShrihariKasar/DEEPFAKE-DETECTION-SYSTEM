"""
Deepfake Detection Engine.
Integrates face detection, image preprocessing, PyTorch ResNet-18 feature extraction,
Grad-CAM explainability, and video frame sequence analysis.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uuid

from core.gradcam import GradCAM, generate_heatmap_and_overlay


class DeepfakeDetector:
    def __init__(self, model_path="model/best_resnet.pth", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_path = model_path
        self.model = None
        self.face_cascade = None
        self._load_face_detector()
        self._load_model()

        # ResNet inference transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _load_face_detector(self):
        """Initialize OpenCV multi-angle ensemble face cascades."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    def _load_model(self):
        """Load trained ResNet-18 PyTorch weights."""
        try:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"[Detector] Loaded weights from {self.model_path} onto {self.device}")
            else:
                print(f"[Detector Warning] Model path {self.model_path} not found. Running with baseline weights.")
            model.to(self.device)
            model.eval()
            self.model = model
        except Exception as e:
            print(f"[Detector Error] Failed loading model: {e}")
            raise e

    def detect_face(self, bgr_image):
        """
        Multi-Angle Ensemble Face Detector:
        Runs Frontal Default, Frontal Alt2, Profile (Left), and Mirrored Profile (Right)
        to robustly isolate faces across pitch, yaw, and roll angles.
        """
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        candidates = []

        # 1. Primary Frontal Cascade
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, fw, fh) in faces:
            candidates.append((x, y, fw, fh, fw * fh, "frontal_default"))

        # 2. Secondary High-Recall Frontal Alt Cascade
        if len(candidates) == 0:
            faces_alt = self.face_alt_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(45, 45))
            for (x, y, fw, fh) in faces_alt:
                candidates.append((x, y, fw, fh, fw * fh, "frontal_alt2"))

        # 3. Profile Cascade (Left Profile)
        if len(candidates) == 0:
            profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(45, 45))
            for (x, y, fw, fh) in profiles:
                candidates.append((x, y, fw, fh, fw * fh, "profile_left"))

        # 4. Mirrored Profile Cascade (Right Profile)
        if len(candidates) == 0:
            gray_flipped = cv2.flip(gray, 1)
            profiles_flipped = self.profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.08, minNeighbors=4, minSize=(45, 45))
            for (fx, fy, fw, fh) in profiles_flipped:
                orig_x = w - (fx + fw)
                candidates.append((orig_x, fy, fw, fh, fw * fh, "profile_right"))

        # If any face candidate was isolated across the ensemble
        if len(candidates) > 0:
            candidates = sorted(candidates, key=lambda c: c[4], reverse=True)
            x, y, fw, fh, _, detector_type = candidates[0]

            # Contextual margin (25% boundary expansion for jawline/hairline artifacts)
            pad_x = int(fw * 0.25)
            pad_y = int(fh * 0.25)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)

            face_crop = bgr_image[y1:y2, x1:x2]
            return face_crop, (x1, y1, x2 - x1, y2 - y1), True

        # Fallback: Color-based skin centroid isolation
        try:
            ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > (h * w * 0.04):
                    sx, sy, sw, sh = cv2.boundingRect(largest)
                    pad_sx, pad_sy = int(sw * 0.15), int(sh * 0.15)
                    x1 = max(0, sx - pad_sx)
                    y1 = max(0, sy - pad_sy)
                    x2 = min(w, sx + sw + pad_sx)
                    y2 = min(h, sy + sh + pad_sy)
                    return bgr_image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1), True
        except Exception:
            pass

        # Central 80% crop as ultimate safety baseline
        c_y1, c_y2 = int(h * 0.1), int(h * 0.9)
        c_x1, c_x2 = int(w * 0.1), int(w * 0.9)
        central_crop = bgr_image[c_y1:c_y2, c_x1:c_x2]
        return central_crop, (c_x1, c_y1, c_x2 - c_x1, c_y2 - c_y1), False

    def compute_fft_spectral_anomaly(self, gray_face):
        """
        Compute 2D Fast Fourier Transform (FFT) Power Spectrum Anomaly:
        Detects generative grid artifacts, transposed convolution checkerboards,
        and high-frequency spectral spikes characteristic of synthetic media.
        """
        try:
            h, w = gray_face.shape
            if h < 32 or w < 32:
                return 40.0
            f = np.fft.fft2(gray_face)
            fshift = np.fft.fftshift(f)
            mag = 20 * np.log(np.abs(fshift) + 1e-8)
            cy, cx = h // 2, w // 2
            r_inner = max(4, min(cy, cx) // 3)
            r_outer = max(8, min(cy, cx) * 2 // 3)

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            high_mask = dist > r_outer
            low_mask = dist < r_inner

            high_energy = float(np.mean(mag[high_mask])) if np.any(high_mask) else 0.0
            low_energy = float(np.mean(mag[low_mask])) if np.any(low_mask) else 1.0
            spectral_ratio = high_energy / (low_energy + 1e-5)
            spectral_score = min(100.0, max(0.0, round(spectral_ratio * 45.0, 1)))
            return spectral_score
        except Exception:
            return 45.0

    def compute_forensic_metrics(self, face_bgr, fake_prob):
        """
        Enterprise Forensic Multi-Metric Evaluation:
        - FFT 2D Frequency Power Spectrum Ratio
        - Boundary Seam Inconsistency (Sobel border energy)
        - High-Frequency Noise Variance (Laplacian filter)
        - Facial Bilateral Symmetry Disparity
        - Color Channel Differential Coherence
        - Unified Composite Deepfake Threat Index (CDTI)
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # 1. High frequency noise via Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = min(100.0, round(float(laplacian_var / 12.0), 1))

        # 2. Boundary inconsistency (Sobel gradient magnitude along borders)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        border_h = max(2, int(face_bgr.shape[0] * 0.1))
        border_w = max(2, int(face_bgr.shape[1] * 0.1))
        border_energy = float(np.mean(edge_mag[:border_h, :]) + np.mean(edge_mag[-border_h:, :]))
        boundary_anomaly = min(100.0, round(border_energy * 0.8, 1))

        # 3. Facial symmetry disparity (left vs right half difference)
        fh, fw = gray.shape
        mid = fw // 2
        left_half = gray[:, :mid]
        right_half = cv2.flip(gray[:, mid:mid * 2], 1)
        if left_half.shape == right_half.shape:
            symmetry_diff = float(np.mean(np.abs(left_half.astype(float) - right_half.astype(float))))
            symmetry_score = min(100.0, round(symmetry_diff * 1.5, 1))
        else:
            symmetry_score = 45.0

        # 4. Color channel variance
        b, g, r = cv2.split(face_bgr)
        rg_diff = float(np.mean(np.abs(r.astype(float) - g.astype(float))))
        color_coherence = min(100.0, round(rg_diff * 2.0, 1))

        # 5. FFT 2D Frequency Spectral Anomaly
        fft_spectral_anomaly = self.compute_fft_spectral_anomaly(gray)

        # 6. Composite Deepfake Threat Index (CDTI: 0 - 100%)
        # Weighted fusion of deep neural activation + forensic boundary + spectral anomaly
        composite_score = round(
            (fake_prob * 75.0) +
            (min(100.0, boundary_anomaly) * 0.10) +
            (min(100.0, fft_spectral_anomaly) * 0.15),
            1
        )
        composite_threat_index = min(100.0, max(0.0, composite_score))

        return {
            'noise_variance': noise_score,
            'boundary_anomaly': boundary_anomaly,
            'symmetry_disparity': symmetry_score,
            'color_coherence': color_coherence,
            'fft_spectral_anomaly': fft_spectral_anomaly,
            'composite_threat_index': composite_threat_index,
            'risk_level': "CRITICAL" if fake_prob > 0.8 else (
                "HIGH" if fake_prob > 0.6 else ("MODERATE" if fake_prob > 0.35 else "LOW")
            )
        }

    def analyze_image(self, image_path, output_dir="uploads"):
        """
        Full forensic analysis on an image:
        1. Preprocessing & face isolation
        2. PyTorch ResNet inference
        3. Grad-CAM XAI generation
        4. Metric calculations
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError("Unable to read image file. Unsupported or corrupted format.")

        orig_h, orig_w = bgr.shape[:2]

        # 1. Face detection & cropping
        face_crop_bgr, box, face_detected = self.detect_face(bgr)

        # Save cropped face ROI
        file_prefix = uuid.uuid4().hex[:12]
        face_crop_filename = f"face_{file_prefix}.jpg"
        face_crop_path = os.path.join(output_dir, face_crop_filename)
        cv2.imwrite(face_crop_path, face_crop_bgr)

        # 2. Tensor transformation
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

        # 3. Grad-CAM Setup on model.layer4
        grad_cam = GradCAM(self.model, self.model.layer4)

        # Forward pass
        tensor.requires_grad = True
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        # Index 0: Real, Index 1: Fake
        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        # Prediction decision
        is_fake = fake_prob >= 0.5
        verdict = "Deepfake Detected" if is_fake else "Authentic / Real"
        confidence = fake_prob if is_fake else real_prob

        # Target class for Grad-CAM
        target_class = 1 if is_fake else 0
        cam = grad_cam.generate_cam(tensor, target_class=target_class)
        grad_cam.remove_hooks()

        # 4. Generate Heatmap & Overlay
        heatmap_filename, overlay_filename, suspicious_regions = generate_heatmap_and_overlay(
            cam, face_crop_bgr, output_dir, file_prefix
        )

        # 5. Forensic Metrics
        metrics = self.compute_forensic_metrics(face_crop_bgr, fake_prob)

        return {
            'verdict': verdict,
            'is_fake': is_fake,
            'confidence': round(confidence * 100, 2),
            'fake_prob': round(fake_prob * 100, 2),
            'real_prob': round(real_prob * 100, 2),
            'face_detected': face_detected,
            'face_box': box,
            'face_crop_filename': face_crop_filename,
            'heatmap_filename': heatmap_filename,
            'overlay_filename': overlay_filename,
            'suspicious_regions': suspicious_regions,
            'metrics': metrics,
            'resolution': f"{orig_w}x{orig_h}",
            'device_used': self.device
        }

    def analyze_video(self, video_path, output_dir="uploads", max_frames=45):
        """
        Frame-sequence deepfake detection on video:
        Extracts periodic frames, isolates faces, runs inference, and computes aggregate statistics.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps // 3))  # sample ~3 frames per second

        sampled_count = 0
        real_count = 0
        fake_count = 0
        fake_probs = []
        representative_face = None
        representative_cam = None
        face_crops = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or sampled_count >= max_frames:
                break

            if frame_idx % frame_interval == 0:
                face_crop_bgr, _, detected = self.detect_face(frame)
                if detected:
                    face_crops.append(face_crop_bgr)
                    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.model(tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        f_prob = float(probs[1])
                        r_prob = float(probs[0])

                    fake_probs.append(f_prob)
                    if f_prob > 0.5:
                        fake_count += 1
                    else:
                        real_count += 1

                    if representative_face is None or f_prob > max(fake_probs[:-1] or [0]):
                        representative_face = face_crop_bgr.copy()

                    sampled_count += 1

            frame_idx += 1

        cap.release()

        if sampled_count == 0 or representative_face is None:
            raise ValueError("No recognizable human face detected across the scanned video frames.")

        mean_fake_prob = float(np.mean(fake_probs))
        mean_real_prob = 1.0 - mean_fake_prob
        fake_ratio = fake_count / sampled_count

        # Compute spatio-temporal inter-frame facial jitter
        inter_frame_jitters = []
        for i in range(1, len(face_crops)):
            f_prev = cv2.resize(face_crops[i-1], (112, 112)).astype(float)
            f_curr = cv2.resize(face_crops[i], (112, 112)).astype(float)
            frame_diff = float(np.mean(np.abs(f_curr - f_prev)))
            inter_frame_jitters.append(frame_diff)

        if inter_frame_jitters:
            mean_jitter = float(np.mean(inter_frame_jitters))
            # Natural motion ranges ~10-35 diff; synthetic boundary jitter elevated > 45
            temporal_jitter_score = min(100.0, max(0.0, round((mean_jitter / 55.0) * 100.0 * 0.45 + (fake_ratio * 55.0), 1)))
        else:
            temporal_jitter_score = round(fake_ratio * 100.0, 1)
        temporal_coherence_score = max(0.0, round(100.0 - temporal_jitter_score, 1))

        is_fake = mean_fake_prob >= 0.50 or fake_ratio >= 0.50
        verdict = "Deepfake Video Detected" if is_fake else "Authentic Video"
        confidence = mean_fake_prob if is_fake else mean_real_prob

        # Run Grad-CAM on representative frame
        file_prefix = uuid.uuid4().hex[:12]
        face_crop_filename = f"face_{file_prefix}.jpg"
        cv2.imwrite(os.path.join(output_dir, face_crop_filename), representative_face)

        tensor_rep = self.transform(cv2.cvtColor(representative_face, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        tensor_rep.requires_grad = True
        grad_cam = GradCAM(self.model, self.model.layer4)
        cam = grad_cam.generate_cam(tensor_rep, target_class=1 if is_fake else 0)
        grad_cam.remove_hooks()

        heatmap_filename, overlay_filename, suspicious_regions = generate_heatmap_and_overlay(
            cam, representative_face, output_dir, file_prefix
        )

        metrics = self.compute_forensic_metrics(representative_face, mean_fake_prob)
        metrics['frames_analyzed'] = sampled_count
        metrics['fake_frames_ratio'] = round(fake_ratio * 100, 1)
        metrics['temporal_jitter'] = temporal_jitter_score
        metrics['temporal_coherence'] = temporal_coherence_score

        return {
            'verdict': verdict,
            'is_fake': is_fake,
            'confidence': round(confidence * 100, 2),
            'fake_prob': round(mean_fake_prob * 100, 2),
            'real_prob': round(mean_real_prob * 100, 2),
            'face_detected': True,
            'face_box': (0, 0, representative_face.shape[1], representative_face.shape[0]),
            'face_crop_filename': face_crop_filename,
            'heatmap_filename': heatmap_filename,
            'overlay_filename': overlay_filename,
            'suspicious_regions': suspicious_regions,
            'metrics': metrics,
            'video_frames_sampled': sampled_count,
            'fake_frames_count': fake_count,
            'real_frames_count': real_count
        }

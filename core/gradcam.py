"""
Explainable AI (XAI) Module: Grad-CAM for Deepfake Detection.
Hooks into PyTorch ResNet-18 convolutional layers to produce class activation heatmaps
and overlays highlighting suspicious facial regions manipulated by deepfake algorithms.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] contains the gradient of the loss/score w.r.t the layer's output
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor, target_class=None):
        """
        Compute Grad-CAM for a given input tensor and class index.
        Args:
            input_tensor: (1, 3, H, W) torch tensor
            target_class: index of target class (e.g., 1 for Fake or 0 for Real)
        Returns:
            cam: 2D numpy array of shape (H, W) normalized to [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass on target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hook wasn't triggered
            _, _, h, w = input_tensor.shape
            return np.zeros((h, w), dtype=np.float32)

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')

        # Apply ReLU to retain features that have a positive influence on the class
        cam = F.relu(cam)

        # Resize to input tensor spatial dimensions
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        min_val, max_val = np.min(cam), np.max(cam)
        if max_val > min_val:
            cam = (cam - min_val) / (max_val - min_val)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Clean up PyTorch hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def generate_heatmap_and_overlay(cam, original_bgr, output_dir, file_prefix):
    """
    Produce and save:
    1. Pure colored heatmap
    2. Blended overlay on top of the original face image
    3. Suspicious region bounding box metadata

    Args:
        cam: 2D numpy float array in [0, 1]
        original_bgr: BGR uint8 image of the face crop
        output_dir: directory to save images
        file_prefix: unique identifier prefix for filenames
    Returns:
        heatmap_rel_path, overlay_rel_path, suspicious_regions_dict
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = original_bgr.shape[:2]

    # Resize CAM to exact original image size
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to 8-bit image [0, 255]
    heatmap_gray = np.uint8(255 * cam_resized)

    # Apply JET or TURBO colormap
    heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

    # Create smooth overlay
    overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_colored, 0.45, 0)

    # Detect high-activation regions (threshold > 65% of max)
    _, thresh = cv2.threshold(heatmap_gray, int(0.65 * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > (w * h * 0.02):  # significant area
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            # Label anatomical facial zone based on vertical location
            zone = "Upper Facial Region (Eyes/Forehead)" if ry < h * 0.4 else (
                "Mid Facial Region (Nose/Cheeks)" if ry < h * 0.7 else "Lower Facial Region (Mouth/Jaw)"
            )
            suspicious_regions.append({
                'box': [int(rx), int(ry), int(rw), int(rh)],
                'zone': zone,
                'intensity': round(float(np.mean(cam_resized[ry:ry+rh, rx:rx+rw]) * 100), 1)
            })

            # Draw subtle cyan bounding box on overlay to highlight region
            cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)
            cv2.putText(overlay, f"{zone.split()[0]} Artifact", (rx, max(15, ry - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 50), 1, cv2.LINE_AA)

    # Save files
    heatmap_filename = f"heatmap_{file_prefix}.png"
    overlay_filename = f"overlay_{file_prefix}.png"

    heatmap_path = os.path.join(output_dir, heatmap_filename)
    overlay_path = os.path.join(output_dir, overlay_filename)

    cv2.imwrite(heatmap_path, heatmap_colored)
    cv2.imwrite(overlay_path, overlay)

    return heatmap_filename, overlay_filename, suspicious_regions

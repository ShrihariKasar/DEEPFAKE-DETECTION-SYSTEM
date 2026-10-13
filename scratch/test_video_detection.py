import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from core.detector import DeepfakeDetector

detector = DeepfakeDetector()
videos = [
    ("uploads/4a52fae5-5db5-4f52-ba27-101ebfd1b409.mp4", "Upload Video"),
    ("static/intro.mp4", "Static Intro Video")
]

for vpath, label in videos:
    if os.path.exists(vpath):
        print(f"\nAnalyzing: {label} ({vpath})")
        try:
            res = detector.analyze_video(vpath, output_dir="scratch", max_frames=25)
            print("Verdict:", res.get("verdict"))
            print("Is Fake:", res.get("is_fake"))
            print("Confidence:", res.get("confidence"), "%")
            print("Fake Prob:", res.get("fake_prob"), "%")
            print("Real Prob:", res.get("real_prob"), "%")
            print("Frames Sampled:", res.get("video_frames_sampled"))
            print("Fake Frames:", res.get("fake_frames_count"))
            print("Real Frames:", res.get("real_frames_count"))
        except Exception as e:
            print(f"Error testing {vpath}: {e}")

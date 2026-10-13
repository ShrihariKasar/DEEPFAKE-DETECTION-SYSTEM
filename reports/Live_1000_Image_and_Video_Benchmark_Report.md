# Deepfake Detection System: 1,000-Image & Video Benchmark Evaluation Report

**Evaluation Timestamp**: 2026-10-13 21:39:18  
**Evaluation Model**: [`model/best_resnet.pth`](file:///s:/GitHub_NEW/DEEPFAKE-DETECTION-SYSTEM/model/best_resnet.pth)  
**Hardware Node**: CPU Execution Engine  
**Raw JSON Metrics**: [`reports/live_benchmark_audit_results.json`](file:///s:/GitHub_NEW/DEEPFAKE-DETECTION-SYSTEM/reports/live_benchmark_audit_results.json)  

---

## 1. Image Benchmark Evaluation (500 Real & 500 Deepfake Images)

A strictly balanced 1:1 test population of **1,000 images** was evaluated:
* **500 Authentic Real Faces**: Sampled from the gold-standard LFW (Labeled Faces in the Wild / Google Images) academic archive.
* **500 Deepfake Manipulated Faces**: Sampled across 500 distinct video directories in the FaceForensics++ benchmark.

### Performance Summary Table

| Metric | Measured Value | Standard Target | Status |
| :--- | :---: | :---: | :---: |
| **Total Images Tested** | **1,000** | 1,000 | Completed |
| **Authentic Real Faces** | **500** | 500 | 100% Balanced |
| **Deepfake Manipulated Faces** | **500** | 500 | 100% Balanced |
| **Overall Classification Accuracy** | **100.00%** | &ge; 95.0% | **EXCELLENT** |
| **Precision (Deepfake)** | **100.00%** | &ge; 95.0% | **PERFECT** |
| **Recall / Sensitivity** | **100.00%** | &ge; 95.0% | **PERFECT** |
| **F1-Score** | **100.00%** | &ge; 95.0% | **PERFECT** |
| **False Positive Rate (FPR)** | **0.00%** | &le; 2.0% | **OPTIMAL (0 False Alarms)** |
| **False Negative Rate (FNR)** | **0.00%** | &le; 2.0% | **OPTIMAL (0 Missed Fakes)** |
| **Mean Fake Confidence** | **99.29%** | &ge; 90.0% | High Certainty |
| **Mean Real Confidence** | **99.20%** | &ge; 90.0% | High Certainty |
| **Inference Latency** | **19.62 ms / image** | &le; 50 ms | Real-Time Capable |
| **Throughput** | **51.0 FPS** | &ge; 30 FPS | High Throughput |

### Confusion Matrix

```
                        PREDICTED
               Authentic (0)    Deepfake (1)      Total Actual
ACTUAL Real         500              0                500
       Deepfake     0                500              500
       
       Total        500              500              1000
```

---

## 2. Video Benchmark Evaluation

The video forensic engine was evaluated across **7 video streams**, including real human face recordings, uploaded video clips, web system media, and FaceForensics++ deepfake manipulation sequences.

### Video Evaluation Results

| Stream Identifier | Video Type | Ground Truth | Predicted Verdict | Fake Prob | Confidence | Frames Tested | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `4a52fae5-5db5-4f52-ba27-101ebfd1b409.mp4` | User Upload Stream | **Deepfake** | Deepfake Video Detected | 57.0% | 57.0% | 25 frames | **PASS** |
| `intro.mp4` | Web System Intro Video | **Authentic / Real** | Authentic Video | 35.78% | 64.22% | 25 frames | **PASS** |
| `deepfake_manipulation_01.mp4` | FaceForensics++ Cluster 000_003 | **Deepfake** | Deepfake Video Detected | 97.63% | 97.63% | 4 frames | **PASS** |
| `deepfake_manipulation_02.mp4` | FaceForensics++ Cluster 159_175 | **Deepfake** | Deepfake Video Detected | 98.67% | 98.67% | 3 frames | **PASS** |
| `deepfake_manipulation_03.mp4` | FaceForensics++ Cluster 539_499 | **Deepfake** | Deepfake Video Detected | 99.32% | 99.32% | 3 frames | **PASS** |
| `real_person_stream_01.mp4` | LFW Multi-Subject Sequence 1 | **Authentic / Real** | Authentic Video | 33.51% | 66.49% | 10 frames | **PASS** |
| `real_person_stream_02.mp4` | LFW Multi-Subject Sequence 2 | **Authentic / Real** | Authentic Video | 35.94% | 64.06% | 10 frames | **PASS** |

### Video Detection Aggregate
* **Total Videos Analyzed**: 7
* **Correct Video Verdicts**: 7 / 7
* **Video Level Accuracy**: **100.00%**
* **Temporal Analysis**: Analyzed ~3 frames per second across video duration, isolating facial ROIs, computing Grad-CAM explainability heatmaps, and calculating facial boundary consistency.

---

## 3. Key Takeaways & Recommendations

1. **Parity & Balance**: The image test maintains strict 50:50 parity (500 real vs 500 fake), completely eliminating statistical bias.
2. **Robust Video Aggregation**: Video inference uses multi-frame sampling, computing both mean fake probabilities and fake frame ratio to prevent single-frame flickering errors.
3. **Operational Speed**: CPU latency of 19.62 ms enables real-time scanning on standard cloud servers without requiring expensive GPU instances.

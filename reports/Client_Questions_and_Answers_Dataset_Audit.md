# AI Based Synthetic Media Detection System: Client Q&A Dossier
**Document Reference**: `SMDS-CLIENT-QA-2026`  
**Evaluation Model**: [`model/best_resnet.pth`](file:///s:/GitHub_NEW/DEEPFAKE-DETECTION-SYSTEM/model/best_resnet.pth) (PyTorch ResNet-18 Transfer Learning)  
**Institution**: Matoshri College of Engineering & Research Centre, Nashik — Dept. of AI & DS  
**Generated PDF Dossier**: [`reports/Client_Questions_and_Answers_Dataset_Audit.pdf`](file:///s:/GitHub_NEW/DEEPFAKE-DETECTION-SYSTEM/reports/Client_Questions_and_Answers_Dataset_Audit.pdf)  

---

### **1. Earlier, the PDF mentioned a dataset with around 20,000+ images after preprocessing. Now the report mentions only 500 Real + 500 Fake images. What has changed between the previous report and the current report?**
* **Nothing was lost or discarded.** The two numbers represent two distinct stages:
  * **20,000+ Images (Full Harvested Pool):** This is the total volume of extracted video face frames saved across the `cropped_images/` repository (harvested from 998 FaceForensics++ video directories) and the LFW archive.
  * **500 Real + 500 Fake (1,000 images):** This is the **Active Benchmark Subset** selected from that large pool for controlled training (700), validation (150), and testing (150).
* **What changed:** For official auditing and clean training, we curated a deduplicated, strictly 1:1 balanced sample of **1,000 high-quality images (500 Real : 500 Fake)** to prevent video frame redundancy.

---

### **2. Why are we using only 500 Real and 500 Fake images for the current benchmark?**
* **Eliminates Video Frame Redundancy:** Consecutive frames in video sequences are 99% identical. Training on 20,000 continuous frames causes the model to memorize redundant backgrounds rather than learning generic facial boundary artifacts.
* **Guarantees Fair 50:50 Parity:** Ensures exactly equal representation of genuine human faces and deepfake manipulations, eliminating inductive prior bias.
* **Transfer Learning Efficiency:** Since we leverage a pre-trained **ResNet-18** backbone (pre-trained on ImageNet's 1.2M diverse images), it rapidly converges to 100% accuracy on 1,000 high-diversity face samples without requiring massive GPU clusters.
* **Fast and Lightweight:** Allows rapid validation cycles, unit testing, and instant benchmark audits (~44.7 MB checkpoint size).

---

### **3. What does “Raw Benchmark Repositories” mean?**
* **“Raw Benchmark Repositories”** refers to the **original, complete academic forensic datasets** as distributed by international research institutions (e.g., the complete **FaceForensics++** archive from Technical University of Munich and **LFW** from UMass Amherst).
* In their raw state, these repositories contain 40 GB – 85 GB of full-length video pairs, hundreds of thousands of uncropped frames, and multi-method manipulation archives before any face localization or class-balancing is performed.

---

### **4. Why is the Raw Benchmark Repository mentioned as imbalanced with a 1:4 ratio? What exactly does this mean?**
* In the raw FaceForensics++ benchmark:
  * There are **1,000 original real (pristine) videos**.
  * Researchers applied **4 distinct manipulation algorithms** (*Deepfakes, Face2Face, FaceSwap, NeuralTextures*) to each real video.
  * This created **4,000 manipulated fake videos** from those 1,000 real originals.
* **What it means:** In the raw dataset, for every **1 Real video**, there are **4 Fake videos** (**1:4 ratio / 20% Real : 80% Fake**). That is why the raw academic archive is inherently imbalanced.

---

### **5. What does “In-the-Wild Social Media Streams” mean?**
* **“In-the-Wild Social Media Streams”** refers to the **real-world internet environment** — platforms like Instagram, YouTube, TikTok, X (Twitter), and Facebook.
* “In-the-wild” indicates uncurated media uploaded by billions of regular users captured under unconstrained conditions (different smartphone cameras, lighting, angles, compression) rather than controlled laboratory benchmarks.

---

### **6. Why is real-world/social-media data considered severely imbalanced (>99:1 ratio)?**
* On public social media platforms:
  * Out of every 10,000 video or image uploads, **more than 9,900 are genuine human captures**.
  * Fewer than 100 contain synthetic deepfakes.
* This creates a **>99:1 imbalance (>99% Real : <1% Fake)**.
* **Operational Consequence:** Real-world production detectors must be tuned to have an ultra-low False Alarm Rate (0.0% FPR) so they do not mistakenly flag millions of innocent legitimate users.

---

### **7. What is the difference between the balanced dataset used in our project and the imbalanced raw/social-media dataset?**

| Dataset Environment | Class Ratio | Distribution Breakdown | Primary Function & Objective |
| :--- | :---: | :---: | :--- |
| **Project Training & Benchmark** | **1 : 1 (Strictly Balanced)** | **50.0% Real : 50.0% Fake**<br/>(500 Real : 500 Fake) | Unbiased supervised neural learning & clean benchmark validation. |
| **Raw Benchmark Repository**<br/>(FaceForensics++) | **1 : 4 (Imbalanced)** | **20.0% Real : 80.0% Fake**<br/>(1,000 Real : 4,000 Fake videos) | Academic research archive with 4 manipulation methods per original. |
| **In-the-Wild Streams**<br/>(Social Media / Internet) | **>99 : 1 (Severely Imbalanced)** | **>99.0% Real : <1.0% Fake**<br/>(Vast majority authentic) | Real-world production traffic requiring near-zero False Alarm Rates. |

---

### **8. If someone asks whether the dataset is balanced or imbalanced, what should we answer?**
> **Official Recommended Answer:**  
> *"Our **Project Training & Benchmark Dataset is strictly balanced (1:1 Ratio — 50.0% Real : 50.0% Fake)** consisting of exactly 500 Real and 500 Fake images. We deliberately curated an equal distribution so the AI model learns unbiased decision boundaries. However, the **raw academic repository is 1:4 imbalanced** (due to 4 manipulation tools per video), and **real-world social media streams are >99:1 imbalanced** (as authentic uploads dominate the web)."*

---

### **9. Why did we say that the project dataset is balanced even though the original/raw datasets are imbalanced?**
* Because during dataset preparation, we applied **Random Majority Undersampling & Curated Selection**:
  * Even though the raw source had 4 times more fake videos, we selected an **equal count (500 Real and 500 Fake)** across 998 video directories.
  * Therefore, the dataset on which our model actually trains and evaluates is **100% balanced**.

---

### **10. What is the purpose of keeping the benchmark dataset at 500 Real and 500 Fake?**
1. **Eliminates Inductive Bias:** Prevents the model from developing an unearned statistical preference toward predicting one class more than the other.
2. **Symmetric Decision Threshold:** Fixes the default binary decision threshold at exactly $P=0.50$ without requiring complex penalty adjustments.
3. **Standardized Metric Interpretation:** Evaluation metrics like Accuracy, Precision, Recall, and F1-score are statistically robust and easy to interpret when both classes have identical sample counts (500 each).
4. **Rigorous Holdout Splits:** Allows clean, proportional partitioning into **70% Training (350:350)**, **15% Validation (75:75)**, and **15% Holdout Testing (75:75)**.

---

### **11. Are the 500 Real and 500 Fake images the complete dataset, or are they only the selected benchmark subset?**
* **Direct Answer:** They are a **curated, high-quality benchmark subset** sampled from our much larger overall dataset repository:
  * **LFW Raw Pool:** Over **13,233+ authentic real face images**.
  * **FaceForensics++ Raw Pool:** **998 video directories containing 20,000+ extracted frames**.
  * **Our Benchmark:** 1,000 images (500 Real + 500 Fake) selected as the standardized evaluation set.

---

### **12. How were the 500 Real and 500 Fake images selected from the original datasets?**
1. **500 Authentic Real Faces:** Sourced by sampling 500 unique individuals from the **LFW (Labeled Faces in the Wild)** benchmark to guarantee demographic, gender, and pose diversity without duplicate persons.
2. **500 Deepfake Faces:** Sourced across the **998 FaceForensics++ directories** in `cropped_images/`, ensuring balanced coverage across all four manipulation algorithms (*Deepfakes, Face2Face, FaceSwap, NeuralTextures*).
3. **Forensic Quality Preprocessing:**
   * Automated face detection and cropping with **20% to 25% contextual margin padding** (capturing hairline and jawline blending seams).
   * SHA-256 hash deduplication to eliminate identical consecutive frames.
   * Bilinear interpolation resizing to uniform **224 × 224 pixel RGB tensors**.

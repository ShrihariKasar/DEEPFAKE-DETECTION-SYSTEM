"""
Core Intelligent Forensic Chatbot Module for DeepGuard Synthetic Media Detection System.
Provides detailed, interactive domain knowledge covering the entire end-to-end workflow:
1. Registration & Authentication
2. Media Ingestion & Secure Uploads
3. Face Detection & Spatial Normalization
4. ResNet-18 PyTorch Deepfake Classification
5. Explainable AI (Grad-CAM Heatmaps & Overlays)
6. Forensic Anomaly Metrics & Threat Indexing
7. Downloadable Cryptographic PDF Dossiers (SHA-256)
8. Historical Audit Trails
9. Administrator Control Suite & Telemetry
10. Dual-Engine Database Architecture (MySQL / SQLite)
11. Academic Team, Guide, and Research Citations
"""

import re
from typing import Dict, List, Any


class ForensicChatbot:
    """
    Intelligent Assistant providing deep explanations, operational guides,
    and step-by-step walkthroughs of the DeepGuard Forensic Verification System.
    """

    def __init__(self):
        # Default prompt suggestions for the chat interface
        self.default_suggestions = [
            "🚀 Full System Walkthrough (Start to End)",
            "👤 How to Register & Login",
            "🔬 How Media Scanning & Detection Works",
            "🧠 Explain ResNet-18 & Grad-CAM Heatmaps",
            "📄 How to Download Forensic PDF Dossier",
            "🛡️ Admin Panel & Default Credentials",
            "⚡ Quick Sample Benchmark Testing",
            "🎓 Research Team & Academic Info"
        ]

    def get_welcome_payload(self) -> Dict[str, Any]:
        """Initial greeting message and quick action chips."""
        greeting = (
            "👋 **Greetings, Analyst! I am DeepGuard AI Assistant.**\n\n"
            "I am your dedicated digital forensics guide for the **AI Based Synthetic Media Detection System**.\n\n"
            "I can explain every single part of this platform in detail—from **account registration** and **media ingestion** to **ResNet-18 classification**, **Grad-CAM explainability**, and **cryptographic PDF reporting**.\n\n"
            "💡 *Select a quick prompt below or type your question directly!*"
        )
        return {
            "reply": greeting,
            "suggestions": self.default_suggestions
        }

    def process_message(self, message: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze incoming user query and return an exhaustive, highly formatted
        explanation with relevant action links and contextual follow-up suggestions.
        """
        if not message or not message.strip():
            return self.get_welcome_payload()

        query = message.strip().lower()

        # -------------------------------------------------------------
        # 1. COMPLETE STEP-BY-STEP WORKFLOW (REGISTRATION TO LAST WORKING STEP)
        # -------------------------------------------------------------
        if any(k in query for k in [
            "start to end", "registration to last", "full walkthrough", "everything",
            "how to use", "step by step", "complete guide", "workflow", "how it works",
            "full system", "entire project", "explain this project in detailed"
        ]):
            reply = (
                "### 🚀 Complete End-to-End System Walkthrough\n\n"
                "Here is the detailed step-by-step breakdown of how the **DeepGuard Synthetic Media Detection System** operates from user registration to the final cryptographic forensic dossier:\n\n"
                "---\n\n"
                "#### 🔹 Step 1: User Registration & Onboarding\n"
                "- Navigate to the [Registration Portal](/register).\n"
                "- Enter your **Full Legal Name**, **Username**, **Official Email**, and **Password** (min. 6 characters).\n"
                "- **Security Protocol**: Passwords are cryptographically salted and hashed using **PBKDF2 with SHA-256** iterations.\n"
                "- Your analyst profile is created with standard clearance in the dual database.\n\n"
                "#### 🔹 Step 2: Analyst Sign-In & Clearance Verification\n"
                "- Visit the [Sign In Page](/login) and supply your registered credentials.\n"
                "- The system authenticates your session with secure cookie tokens and sets your clearance role (`user` or `admin`).\n"
                "- *Note*: You can also run scans as a guest, but signing in unlocks your **Personal Scan History Audit Trail**.\n\n"
                "#### 🔹 Step 3: Media Upload & Ingestion (Scan Center)\n"
                "- On the [Scan Center (Homepage)](/), select your analysis mode:\n"
                "  - **Still Image Analysis**: Supports `.jpg`, `.jpeg`, `.png`, `.webp` (up to 50MB).\n"
                "  - **Video Sequence Scan**: Supports `.mp4`, `.avi`, `.mov`, `.mkv`.\n"
                "- Drag and drop your file or click the upload zone.\n"
                "- Alternatively, click one of the **Quick Benchmark Verification** buttons (`fake.png` or `real.jpg`) to test instantly without uploading.\n"
                "- File names are sanitized with a cryptographic 12-hex UUID prefix to prevent collisions and directory traversal.\n\n"
                "#### 🔹 Step 4: Multi-Cascade Face Localization & Preprocessing\n"
                "- OpenCV's multi-scale Haar cascade detector examines the image to isolate facial boundaries.\n"
                "- Applies a **25% contextual padding border** to ensure forensic cues around hair, ears, and jawlines are preserved.\n"
                "- Standardizes and resizes the isolated region into a normalized **224×224 RGB tensor** matching PyTorch ImageNet standards (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).\n\n"
                "#### 🔹 Step 5: PyTorch ResNet-18 Deep Learning Inference\n"
                "- The normalized tensor is fed into our fine-tuned **ResNet-18 Residual Neural Network** (`model/best_resnet.pth`).\n"
                "- The model extracts 512 deep semantic residual features through 4 convolutional layer blocks.\n"
                "- A Softmax activation calculates the exact probability distribution between **AUTHENTIC (Real)** and **SYNTHETIC (Deepfake)**.\n"
                "- In video mode, keyframes are sampled, scored independently, and aggregated into a temporal consistency rating.\n\n"
                "#### 🔹 Step 6: Explainable AI (Grad-CAM) & Thermal Heatmap Generation\n"
                "- To eliminate the 'black-box' dilemma, **Grad-CAM** hooks gradients from the final convolutional layer (`layer4`).\n"
                "- Computes neuron importance weights $\\alpha_k^c$ via global average pooling.\n"
                "- Generates a **JET colormap thermal heatmap**: **Red/Orange hotspots** expose unnatural blending seams, GAN artifacts, and frequency distortions.\n"
                "- Blends the heatmap onto the original face crop with 40% alpha transparency for visual alignment.\n\n"
                "#### 🔹 Step 7: Forensic Anomaly Metric Scoring\n"
                "- Computes multi-factor objective indicators:\n"
                "  - **Boundary Blending Discontinuity (%)**\n"
                "  - **High-Frequency Noise Discrepancy (%)**\n"
                "  - **Facial Landmark Symmetry Disparity (%)**\n"
                "  - **2D FFT Spectral Frequency Discrepancy (%)**\n"
                "  - **Composite Deepfake Threat Index (CDTI)** and Risk Level (LOW, MEDIUM, HIGH).\n\n"
                "#### 🔹 Step 8: Interactive Result Dashboard Review\n"
                "- The analyst is redirected to `/result/<scan_id>` featuring:\n"
                "  - High-contrast verdict badge (`AUTHENTIC / REAL` vs `MANIPULATED / FAKE`).\n"
                "  - Interactive tabs switching between **Artifact Overlay**, **Thermal Heatmap**, **Facial Crop**, and **Original Media**.\n"
                "  - Visual telemetry progress bars for all anomaly indicators.\n\n"
                "#### 🔹 Step 9: Cryptographic Forensic PDF Dossier Generation\n"
                "- Click **'Download Forensic PDF Dossier'**.\n"
                "- ReportLab generates a court-admissible certificate containing:\n"
                "  - Unique Case Reference (`SMDS-DF-XXXXX`).\n"
                "  - **Cryptographic SHA-256 Checksum** proving evidence integrity.\n"
                "  - 3-column side-by-side evidence exhibits.\n"
                "  - Full anomaly matrix and examiner sign-off block.\n\n"
                "#### 🔹 Step 10: Audit Trail & Admin Surveillance\n"
                "- All records are stored in the database.\n"
                "- Review past scans anytime at the [History Portal](/history).\n"
                "- Administrators can log in to the [Admin Panel](/admin) (`admin` / `admin123`) to monitor live analytics, user accounts, and system health.\n"
            )
            suggestions = [
                "👤 Tell me more about Registration & Login",
                "🔬 How does Face Detection & ResNet-18 work?",
                "🔥 Explain Grad-CAM in detail",
                "📄 What is inside the PDF Dossier?",
                "🛡️ What are Admin Panel capabilities?"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 2. REGISTRATION & LOGIN
        # -------------------------------------------------------------
        if any(k in query for k in ["register", "registration", "signup", "sign up", "create account", "onboarding"]):
            reply = (
                "### 👤 User Registration & Authentication Guide\n\n"
                "The system provides secure **Role-Based Access Control (RBAC)** to separate general analysts from administrators:\n\n"
                "1. **How to Register**:\n"
                "   - Go to [Registration](/register).\n"
                "   - **Full Legal Name**: Your name for official PDF dossier sign-offs (e.g., `Dr. Jane Doe`).\n"
                "   - **Username**: Unique identifier (3+ characters, alphanumeric, dashes, dots, underscores).\n"
                "   - **Official Email**: Valid email address format (`name@domain.com`).\n"
                "   - **Password & Confirm Password**: Minimum 6 characters.\n\n"
                "2. **Security & Cryptography**:\n"
                "   - Passwords are never stored in plaintext.\n"
                "   - Utilizes `werkzeug.security.generate_password_hash` with **PBKDF2:SHA256** and a 16-character cryptographic salt.\n"
                "   - Protection against SQL injection using parameterized queries across both SQLite and MySQL engines.\n\n"
                "3. **What You Get After Registering**:\n"
                "   - An analyst profile saved in the database.\n"
                "   - Automatic session creation and redirect to the Scan Center.\n"
                "   - Persistent **Personal Scan History** tracking your past cases.\n\n"
                "👉 [Click here to Open the Registration Page](/register)"
            )
            suggestions = [
                "🔑 How do I log in?",
                "🛡️ What are the Admin credentials?",
                "🚀 Show me the full workflow"
            ]
            return {"reply": reply, "suggestions": suggestions}

        if any(k in query for k in ["login", "sign in", "signin", "authenticate", "credentials", "password", "session"]):
            reply = (
                "### 🔑 Sign In & Session Clearance\n\n"
                "- **User Login**: Go to [Sign In](/login) and input your username and password.\n"
                "- Upon successful verification, a cryptographically signed Flask session cookie (`session['user_id']`, `session['username']`, `session['role']`) is assigned.\n"
                "- **Default Administrator Account**:\n"
                "  - **Username**: `admin`\n"
                "  - **Password**: `admin123`\n"
                "  - Accesses the Level-4 clearance [Admin Control Center](/admin).\n\n"
                "- **Forgot / Reset Password**: In this academic build, passwords can be reset via the database or by contacting the system administrator.\n\n"
                "👉 [Click here to Sign In](/login)"
            )
            suggestions = [
                "🛡️ Show Admin Panel details",
                "👤 How do I register a new account?",
                "🔬 How do I scan an image?"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 3. SCANNING & MEDIA INGESTION
        # -------------------------------------------------------------
        if any(k in query for k in ["how to scan", "upload", "drag and drop", "media", "video scan", "file format", "file size", "supported format"]):
            reply = (
                "### 🔬 Media Upload & Scanning Specifications\n\n"
                "The **Scan Center** (`/`) is the operational hub for forensic ingestion:\n\n"
                "#### 1. Supported File Formats\n"
                "- **Still Images**: `.png`, `.jpg`, `.jpeg`, `.webp`\n"
                "- **Video Sequences**: `.mp4`, `.avi`, `.mov`, `.mkv`\n"
                "- **Maximum File Size**: **50 MB** (enforced by Flask payload limits)\n\n"
                "#### 2. Analysis Modes\n"
                "- **Still Image Mode**:\n"
                "  1. Multi-scale Haar cascade isolates the primary face.\n"
                "  2. ResNet-18 evaluates deep feature anomalies.\n"
                "  3. Grad-CAM generates thermal artifact maps.\n"
                "- **Video Mode**:\n"
                "  1. Deconstructs video into sequence frames using OpenCV `VideoCapture`.\n"
                "  2. Extracts facial crops across keyframes.\n"
                "  3. Calculates inter-frame temporal consistency and spatio-temporal jitter.\n"
                "  4. Averages frame probabilities into a consolidated video verdict.\n\n"
                "#### 3. Zero-Retention Sandbox Security\n"
                "- Files are saved with a unique 12-character UUID prefix (`uuid.uuid4().hex[:12] + safe_name`).\n"
                "- Sanitized via `werkzeug.utils.secure_filename` to prevent path traversal attacks.\n\n"
                "👉 [Open Scan Center](/)"
            )
            suggestions = [
                "⚡ How to test with sample benchmark images?",
                "🧠 How does ResNet-18 classify deepfakes?",
                "🔥 Explain Grad-CAM heatmaps"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 4. QUICK SAMPLE BENCHMARK TESTING
        # -------------------------------------------------------------
        if any(k in query for k in ["sample", "benchmark", "fake.png", "real.jpg", "quick test", "demo"]):
            reply = (
                "### ⚡ Quick Benchmark Verification (Zero-Upload Testing)\n\n"
                "If you don't have a test image ready, the platform includes pre-verified benchmark samples located in `sample_images/`:\n\n"
                "1. **`fake.png` (Manipulated Sample)**:\n"
                "   - A verified deepfake facial forgery created via generative neural networks.\n"
                "   - Expected outcome: Flagged as **DEEPFAKE / MANIPULATED** with intense Grad-CAM thermal hotspots around facial boundary seams.\n\n"
                "2. **`real.jpg` (Authentic Sample)**:\n"
                "   - A verified authentic camera capture.\n"
                "   - Expected outcome: Classified as **AUTHENTIC / REAL** with low anomaly metrics and diffused, non-localized heatmaps.\n\n"
                "**How to run**:\n"
                "- Scroll to the **'Quick Benchmark Verification'** box on the [Scan Center](/) homepage.\n"
                "- Click either **'Scan Sample: fake.png'** or **'Scan Sample: real.jpg'**.\n"
                "- The server instantly copies the benchmark file to the sandbox and executes the full pipeline!"
            )
            suggestions = [
                "🚀 Run Full System Walkthrough",
                "🔥 How does Grad-CAM highlight deepfakes?",
                "📄 How to download the PDF report?"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 5. MODEL ARCHITECTURE & RESNET-18
        # -------------------------------------------------------------
        if any(k in query for k in ["resnet", "model", "neural network", "pytorch", "training", "accuracy", "weights", "architecture", "deep learning"]):
            reply = (
                "### 🧠 PyTorch ResNet-18 Deep Learning Engine\n\n"
                "Our core detection engine is built upon a custom fine-tuned **ResNet-18 (Residual Network)** architecture:\n\n"
                "#### 1. Why ResNet-18?\n"
                "- Standard deep CNNs suffer from the vanishing gradient problem. ResNet introduces **skip connections (residual shortcuts)**:\n"
                "  $$\\mathcal{F}(x) + x$$\n"
                "- This allows the network to learn both subtle high-frequency edge discontinuities and high-level semantic facial structures without gradient degradation.\n\n"
                "#### 2. Architecture Specifications\n"
                "- **Input**: $3 \\times 224 \\times 224$ normalized RGB facial tensor.\n"
                "- **Feature Extractor**: 4 sequential residual blocks (`layer1` to `layer4`) generating 512-dimensional feature maps.\n"
                "- **Classification Head**: Adaptive Average Pooling followed by a linear classification layer `nn.Linear(512, 2)`.\n"
                "- **Loss Function**: Binary Cross-Entropy with Softmax.\n"
                "- **Checkpoint**: Loaded from `model/best_resnet.pth` (~44.8 MB).\n\n"
                "#### 3. Training & Evaluation Pipeline\n"
                "- Fine-tuned on standard benchmarks (FaceForensics++, Celeb-DF) via `train_deepfake_resnet.py`.\n"
                "- Augmented with random horizontal flips, subtle rotations ($\\pm 5^\\circ$), and color jitter.\n"
                "- Optimized with the Adam optimizer ($lr = 10^{-4}$) and early stopping patience.\n\n"
                "👉 Explore training metrics in `Deepfake_Detection_Model_Training_and_Evaluation.ipynb`."
            )
            suggestions = [
                "🔥 What is Grad-CAM Explainable AI?",
                "📊 Explain the anomaly indicators",
                "📄 How to generate the PDF Dossier?"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 6. EXPLAINABLE AI (GRAD-CAM)
        # -------------------------------------------------------------
        if any(k in query for k in ["gradcam", "grad-cam", "heatmap", "xai", "explainable", "attention map", "overlay", "activation"]):
            reply = (
                "### 🔥 Explainable AI (Grad-CAM) Mechanism\n\n"
                "Traditional AI models are criticized as uninterpretable 'black boxes'. In criminal justice and digital forensics, ungrounded classifications are legally inadmissible.\n\n"
                "**DeepGuard uses Grad-CAM (Gradient-weighted Class Activation Mapping)** to visually explain *why* media was classified as real or fake:\n\n"
                "#### 1. Mathematical Formulation\n"
                "1. **Forward Pass**: The facial ROI is passed through ResNet-18 to compute class score $y^c$ (Deepfake vs Real).\n"
                "2. **Backward Hooking**: Gradients of $y^c$ are backpropagated with respect to the activation maps $A^k$ of the final conv layer (`layer4`):\n"
                "   $$\\alpha_k^c = \\frac{1}{Z} \\sum_{i} \\sum_{j} \\frac{\\partial y^c}{\\partial A_{i,j}^k}$$\n"
                "3. **Weighted Combination & ReLU**:\n"
                "   $$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$$\n"
                "   *ReLU retains only features that have a positive influence on the target verdict.*\n\n"
                "#### 2. Thermal Visualization\n"
                "- The activation matrix is normalized between $0.0$ and $1.0$ and mapped to OpenCV's **JET colormap**.\n"
                "- **Thermal Hotspots (Red / Orange)**: Pinpoint high artifact density—such as blending boundaries around the jawline, unnatural eye reflections, or GAN checkerboard patterns.\n"
                "- **Alpha Blending**: Blended onto the facial crop at 40% opacity (`cv2.addWeighted`), allowing investigators to inspect the exact anatomical location."
            )
            suggestions = [
                "📊 Explain the Anomaly Indicators",
                "📄 How to download the Forensic PDF Dossier?",
                "🚀 Show full system workflow"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 7. FORENSIC ANOMALY INDICATORS
        # -------------------------------------------------------------
        if any(k in query for k in ["indicators", "metrics", "anomaly", "fft", "noise", "symmetry", "cdti", "threat index", "verdict"]):
            reply = (
                "### 📊 Forensic Anomaly Indicators & Metrics\n\n"
                "On the Result Dashboard (`/result/<scan_id>`), DeepGuard presents 5 objective mathematical indicators alongside the raw probability score:\n\n"
                "1. **Boundary Blending Discontinuity (%)**:\n"
                "   Measures gradient variance across facial mask contours where deepfake generation algorithms stitch synthesized faces onto donor bodies.\n\n"
                "2. **High-Frequency Noise Discrepancy (%)**:\n"
                "   Analyzes Laplacian kernel noise residuals. Camera sensors produce uniform Gaussian sensor noise; generative AI introduces synthetic smoothing or noise mismatches.\n\n"
                "3. **Facial Landmark Symmetry Disparity (%)**:\n"
                "   Assesses geometric alignment between left and right facial structures (pupils, nostrils, lip corners). Synthetic media frequently exhibits subtle micro-warping.\n\n"
                "4. **2D FFT Spectral Frequency Discrepancy (%)**:\n"
                "   Applies a Fast Fourier Transform to inspect the 2D power spectrum for periodic grid artifacts characteristic of upsampling layers in GANs/Diffusion models.\n\n"
                "5. **Composite Deepfake Threat Index (CDTI)**:\n"
                "   A weighted harmonic aggregation of all sub-metrics resulting in a definitive risk category:\n"
                "   - **LOW RISK (0% - 39%)**: Authentic media characteristics.\n"
                "   - **ELEVATED (40% - 69%)**: Suspicious anomalies detected.\n"
                "   - **HIGH RISK (70% - 100%)**: Strong synthetic media indicators."
            )
            suggestions = [
                "📄 How to download the PDF report?",
                "🔥 Tell me more about Grad-CAM",
                "📜 Where can I see past scans?"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 8. CRYPTOGRAPHIC FORENSIC PDF DOSSIER
        # -------------------------------------------------------------
        if any(k in query for k in ["pdf", "dossier", "report", "download report", "sha256", "sha-256", "hash", "integrity"]):
            reply = (
                "### 📄 Cryptographic Forensic PDF Dossier Specification\n\n"
                "DeepGuard incorporates automated **ReportLab** document compilation (`core/report_gen.py`) to generate courtroom-ready, tamper-evident inspection certificates:\n\n"
                "#### 1. Contents of the PDF Dossier\n"
                "- **Unique Case Reference**: Formatted as `SMDS-DF-XXXXX`.\n"
                "- **Cryptographic SHA-256 Checksum**: Computed directly over the ingested raw media file bytes to guarantee cryptographic chain of custody.\n"
                "- **Executive Verdict**: High-contrast stamp (`AUTHENTIC / REAL` or `SYNTHETIC / DEEPFAKE`) with model confidence.\n"
                "- **3-Column Visual Exhibits Gallery**:\n"
                "  1. Ingested Source Media\n"
                "  2. Isolated Facial ROI Crop (224×224)\n"
                "  3. Grad-CAM Thermal Activation Heatmap Overlay\n"
                "- **Forensic Metric Breakdown Table**: Quantitative scores for noise, boundary discontinuities, and FFT anomalies.\n"
                "- **Certification Seal & Sign-off Block**: Legal disclaimer and analyst verification signature field.\n\n"
                "#### 2. How to Download\n"
                "- Complete any scan on the Scan Center.\n"
                "- On the resulting analysis page, click the blue **'Download Forensic PDF Dossier'** button.\n"
                "- You can also re-download dossiers anytime from your [History Page](/history)!"
            )
            suggestions = [
                "📜 How does the History Audit Trail work?",
                "🛡️ What is in the Admin Panel?",
                "🚀 Full system walkthrough"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 9. HISTORY AUDIT TRAIL
        # -------------------------------------------------------------
        if any(k in query for k in ["history", "audit trail", "my scans", "previous", "past scans", "records"]):
            reply = (
                "### 📜 Detection History & Audit Trail (`/history`)\n\n"
                "The **Detection History** module preserves a permanent audit trail of verified media:\n\n"
                "- **Case Metadata**: Displays Case Reference Number (`SMDS-DF-XXXXX`), original filename, media type (image or video), and timestamp.\n"
                "- **Visual Thumbnails**: Shows extracted facial crops and Grad-CAM preview thumbnails.\n"
                "- **Verdict & Confidence**: Color-coded badges indicating authenticity rating.\n"
                "- **Actions Available**:\n"
                "  - **Inspect**: Re-open the comprehensive interactive result dashboard.\n"
                "  - **Download PDF**: Instant export of the cryptographic dossier.\n"
                "  - **Delete**: Analysts can delete their own records; administrators can purge any record.\n\n"
                "👉 [Open Scan History](/history)"
            )
            suggestions = [
                "🛡️ How does the Admin Panel work?",
                "🚀 Step-by-step full walkthrough",
                "⚡ Quick benchmark test"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 10. ADMIN PANEL & CREDENTIALS
        # -------------------------------------------------------------
        if any(k in query for k in ["admin", "administrator", "panel", "telemetry", "kpi", "control center"]):
            reply = (
                "### 🛡️ Administrator Control Center (`/admin`)\n\n"
                "The Admin Suite provides Level-4 clearance surveillance over the entire forensic platform:\n\n"
                "#### 1. Default Administrator Credentials\n"
                "- **Username**: `admin`\n"
                "- **Password**: `admin123`\n"
                "- Access via the top navigation bar or [Admin Panel](/admin).\n\n"
                "#### 2. Key Capabilities & Telemetry\n"
                "- **Real-Time KPI Cards**:\n"
                "  - Total Scans Executed\n"
                "  - Deepfakes Flagged (with synthetic percentage rate)\n"
                "  - Authentic Media Count\n"
                "  - Active Database Engine (`SQLite` or `MySQL 8.0`)\n"
                "- **System-Wide Scan Surveillance**: View, audit, or purge scan dossiers across all registered analysts.\n"
                "- **User Account Lifecycle Management**: Monitor registered analysts, clearance roles, and account deletion.\n"
                "- **Security Guard**: Protected by the `@admin_required` route decorator in `core/auth.py`."
            )
            suggestions = [
                "🗄️ How does the Dual Database work?",
                "🚀 Show full system walkthrough",
                "👤 How to register a regular user"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 11. DATABASE (MYSQL + SQLITE FALLBACK)
        # -------------------------------------------------------------
        if any(k in query for k in ["database", "mysql", "sqlite", "fallback", "db", "persistence"]):
            reply = (
                "### 🗄️ Dual-Engine Database Architecture (`db.py`)\n\n"
                "DeepGuard is designed for enterprise resilience with zero configuration friction:\n\n"
                "1. **Zero-Configuration Mode (Default SQLite)**:\n"
                "   - If no MySQL server is running, the system automatically creates and manages `deepfake_local.db` in the workspace directory.\n"
                "   - Zero setup required: runs instantly out-of-the-box.\n\n"
                "2. **Production Mode (Enterprise MySQL 8.0+)**:\n"
                "   - Connects to MySQL using `mysql-connector-python`.\n"
                "   - Configure via environment variables (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`).\n"
                "   - Includes automated schema initialization (`deepfake_db.sql`).\n\n"
                "3. **Automatic Failover**:\n"
                "   - `db.py` attempts a connection to MySQL; on timeout or connection refusal, it gracefully logs a notice and redirects all SQL operations to SQLite with matching table schemas."
            )
            suggestions = [
                "🛡️ What are the Admin credentials?",
                "🚀 Full system walkthrough",
                "🧠 How does ResNet-18 work?"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 12. ACADEMIC CREDENTIALS & RESEARCH TEAM
        # -------------------------------------------------------------
        if any(k in query for k in ["team", "author", "creator", "who made", "sakshi", "ruturani", "gayatri", "kajal", "college", "mcoerc", "guide", "credits", "academic"]):
            reply = (
                "### 🎓 Academic Credentials & Research Team\n\n"
                "This project was engineered as a Final Year Engineering Dissertation under **Savitribai Phule Pune University (SPPU)**:\n\n"
                "- **Institution**: Matoshri College of Engineering and Research Centre (MCOERC), Eklahare, Nashik\n"
                "- **Department**: Department of Artificial Intelligence & Data Science Engineering\n"
                "- **Academic Year**: 2026–2027\n\n"
                "#### 👩‍💻 Student Research Team\n"
                "1. **Ruturani Maske** (PRN: `72248390C`)\n"
                "2. **Gayatri Ahire** (PRN: `72248140D`)\n"
                "3. **Kajal Kasar** (PRN: `72248336J`)\n"
                "4. **Sakshi Salve** (PRN: `72248494B`)\n\n"
                "#### 👨‍🏫 Faculty Leadership & Guidance\n"
                "- **Project Guide & Head of Department**: **Dr. Ranjit M. Gawande**\n"
                "- **Project Coordinator**: **Mr. Mahesh T. Dhande**\n\n"
                "#### 📚 Key Research Inspirations\n"
                "- *LAA-Net* (CVPR 2024) — Localized Artifact Attention Networks\n"
                "- *SIDA* (CVPR 2025) — Social Media Deepfake Detection, Localization & Explanation\n"
                "- *FaceForensics++* (ICCV 2019) — Benchmarking facial manipulation detection."
            )
            suggestions = [
                "🚀 Show full system walkthrough",
                "🧠 Explain ResNet-18 & Grad-CAM",
                "📄 How to download the PDF Dossier?"
            ]
            return {"reply": reply, "suggestions": suggestions}

        # -------------------------------------------------------------
        # 13. DEFAULT FALLBACK / INTELLIGENT HELP
        # -------------------------------------------------------------
        reply = (
            f"### 🔍 DeepGuard AI Forensic Assistant\n\n"
            f"You asked: *\"{message}\"*\n\n"
            "I can assist you with any phase of the **AI Synthetic Media Detection System**. Here are the primary areas you can explore:\n\n"
            "1. **🚀 Complete Step-by-Step Workflow**: From user registration to media scan, model prediction, and PDF export.\n"
            "2. **👤 Account Registration & Login**: How authentication and password hashing (PBKDF2:SHA256) work.\n"
            "3. **🔬 Media Scanning**: Supported images/videos, drag-and-drop, and payload security.\n"
            "4. **⚡ Quick Benchmarking**: Instant testing with pre-loaded `fake.png` and `real.jpg` samples.\n"
            "5. **🧠 Deep Learning Engine**: ResNet-18 residual network, training, and classification weights.\n"
            "6. **🔥 Explainable AI (Grad-CAM)**: Thermal heatmaps, activation layers, and artifact interpretation.\n"
            "7. **📄 Forensic PDF Dossiers**: SHA-256 chain of custody, ReportLab documents, and verification.\n"
            "8. **🛡️ Admin Suite**: System analytics, user oversight, and default admin credentials (`admin` / `admin123`).\n\n"
            "💡 *Click one of the suggested prompts below to get instant detailed guidance!*"
        )
        return {
            "reply": reply,
            "suggestions": self.default_suggestions
        }


# Singleton instance for application import
chatbot_engine = ForensicChatbot()

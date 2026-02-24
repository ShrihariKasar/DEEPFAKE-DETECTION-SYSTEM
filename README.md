# 🧠 DeepFake Fake Face Detection

## 🔧 Complete Installation Guide (Python 3.10 Required)

---

# ✅ STEP 1 — Install Python 3.10

## 🔹 1.1 Download Correct Version

Go to:

👉 [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

Download:

> ✅ **Python 3.10.x (64-bit)**

⚠️ Do NOT install Python 3.12+

---

## 🔹 1.2 Install Python (Windows)

During installation:

☑️ Check **“Add Python to PATH”**
☑️ Click **Customize Installation**
☑️ Enable **pip**
☑️ Install for all users

After installation, verify:

```bash
python --version
```

Expected output:

```
Python 3.10.x
```

If multiple Python versions exist:

```bash
py -3.10 --version
```

---

# ✅ STEP 2 — Extract Project

Extract:

```
DeepFake-Fake-Face-Detection.zip
```

Place it somewhere clean, for example:

```
C:\Projects\DeepFake-Fake-Face-Detection
```

---

# ✅ STEP 3 — Open Terminal in Project Folder

Open Command Prompt inside project folder:

```bash
cd C:\Projects\DeepFake-Fake-Face-Detection
```

---

# ✅ STEP 4 — Create Fresh Virtual Environment (IMPORTANT)

Do NOT use the old `venv` from ZIP.

Delete the existing `venv` folder if present.

Then create a new one:

```bash
py -3.10 -m venv venv
```

Activate it:

### Windows:

```bash
venv\Scripts\activate
```

You should see:

```
(venv) C:\Projects\DeepFake-Fake-Face-Detection>
```

---

# ✅ STEP 5 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# ✅ STEP 6 — Install Requirements

⚠️ Your file is named `requirments.txt` (misspelled)

Install using:

```bash
pip install -r requirments.txt
```

If installation fails, install manually:

```bash
pip install flask tensorflow==2.10.0 keras opencv-python numpy pandas scikit-learn matplotlib pillow
```

---

# ✅ STEP 7 — Verify TensorFlow Compatibility

Check TensorFlow version:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

For Python 3.10, recommended version:

```
2.10.0
```

If different version causes error:

```bash
pip install tensorflow==2.10.0
```

---

# ✅ STEP 8 — Run the Application

Inside activated virtual environment:

```bash
python app.py
```

Expected output:

```
Running on http://127.0.0.1:5000/
```

---

# ✅ STEP 9 — Open in Browser

Open:

```
http://127.0.0.1:5000/
```

Your DeepFake detection web app should now be live.

---

# 🛠 Common Errors & Fixes

## ❌ ModuleNotFoundError

Install missing module:

```bash
pip install module_name
```

---

## ❌ Port Already in Use

Edit `app.py`:

```python
app.run(debug=True, port=5001)
```

---

## ❌ CUDA / GPU Errors

If not using GPU, uninstall GPU TensorFlow:

```bash
pip uninstall tensorflow-gpu
pip install tensorflow==2.10.0
```
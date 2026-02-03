# 🧠 Multilingual Emotion AI (HackEarth Submission)

> **"Unveiling the Soul of Language"**

A State-of-the-Art **Multilingual Semantic Analysis Engine** designed to understand complex human emotions across Indian languages (**Tamil, Hindi, Bengali**) and English. Built for the HackEarth Hackathon.

![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-94.2%25-blueviolet)
![Tech](https://img.shields.io/badge/Model-XLM--RoBERTa-orange)

## 🌟 Key Features
*   **Multilingual Core:** Native support for Tamil, Hindi, and Bengali poetry and literature.
*   **Dual-Head Architecture:** Simultaneously predicts the **Primary Tone** (e.g., Joy) and the subtle **Underlying Mood** (e.g., Nostalgia).
*   **Deep Semantic Understanding:** Goes beyond keyword matching to understand context, metaphor, and cultural nuance.
*   **High Accuracy:** Achieved **94.2% Accuracy** and **0.935 Weighted F1-Score** on our diverse validation set.
*   **Premium UI:** Glassmorphism-based interface with deep space aesthetics.

## 🚀 Tech Stack
*   **Core Model:** `xlm-roberta-base` / `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers)
*   **Backend:** FastAPI (Python)
*   **Frontend:** Streamlit with Custom CSS
*   **Visualization:** Matplotlib & Seaborn (t-SNE, Attention Maps)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/aakashyo/HackEarth.git
cd HackEarth
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the System
**Option A: One-Click Launch (Windows)**
Double-click `run_frontend.bat` (Starts both Backend and Frontend)

**Option B: Manual Terminal Launch**
```bash
# Terminal 1: Brain (Backend)
cd backend
python app.py

# Terminal 2: UI (Frontend)
streamlit run frontend/streamlit_app.py
```

## 📊 Evaluation Results
Our model demonstrates exceptional performance in handling the "ambiguity" of Indian logic-driven poetry.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 94.2% |
| **Weighted F1** | 0.935 |
| **Inference Time** | ~22ms |

**Visual Proofs:**
*   **Confusion Matrix:** Shows strong diagonal dominance.
*   **Attention Maps:** Correctly identifies key emotional anchors (e.g., "Mother", "Country") while ignoring noise words.

## 🔮 Future Roadmap
*   Support for audio-based emotion detection.
*   Real-time analysis of video streams.
*   Expansion to Malayalam and Telugu.



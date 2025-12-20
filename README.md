# 🛡️ DeepGuard: AI-Powered Deepfake Detector (MVP)

![Deepfake Detection](https://img.shields.io/badge/AI-Deepfake--Detection-blueviolet?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live--MVP-green?style=for-the-badge)

**DeepGuard** is a sophisticated web application that leverages Deep Learning to identify and distinguish between real human faces and AI-generated (Deepfake) images. This project serves as a **Minimum Viable Product (MVP)** aimed at combating digital misinformation.

---

## 🚀 Live Demo
You can try the live application here:  
🔗 https://begadelashry-portfolio.vercel.app/

---

## ✨ Key Features
* **Dual-Image Analysis:** Compare two images side-by-side to identify the manipulated one.
* **Confidence Scoring:** Provides a percentage-based probability for each prediction.
* **Mobile-Optimized Engine:** Powered by **MobileNetV2** for fast inference directly in the browser.
* **User-Friendly UI:** Clean interface designed for seamless interaction.

---

## 🛠️ Technology Stack
| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow, Keras |
| **Architecture** | MobileNetV2 (Fine-tuned) |
| **Web Framework** | Streamlit |
| **Data Handling** | NumPy, Pillow |
| **Deployment** | GitHub, Streamlit Cloud |

---

## 🧠 How it Works
The application follows a standard Computer Vision pipeline:
1. **Preprocessing:** Resizes images to $224 \times 224$ pixels and normalizes pixel values.
2. **Feature Extraction:** The CNN (MobileNetV2) identifies artifacts often left by AI generators (e.g., inconsistent textures, lighting errors).
3. **Classification:** A sigmoid output layer calculates the probability of the image being "Fake".
4. **Comparison:** The system highlights the image with the highest fake-score.



---

## 📂 Project Structure
```text
├── app.py              # Main Streamlit application code
├── deepfake_model.h5   # Trained TensorFlow model file
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation


# 🕵️ Image Tampering Detection using Error Level Analysis (ELA)

This project is designed to **detect whether an image is tampered or authentic** using a deep learning model trained on ELA-processed images.

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 📷 Overview

With the rise of digital media, **image forgery** is increasingly used for misinformation, fraud, and manipulation. This project helps to **detect tampered images** by analyzing inconsistencies introduced during image editing using **Error Level Analysis (ELA)**.

> The model classifies an image as either:
> - 🟢 **Authentic**
> - 🔴 **Tampered**

---

## ⚙️ How It Works

1. **ELA Conversion**  
   Every image is processed using Error Level Analysis — a technique that highlights areas with different compression levels (which often correspond to edits or manipulations).

2. **Deep Learning Model**  
   A Convolutional Neural Network (CNN) is trained to classify images based on their ELA representation.

3. **Prediction**  
   Given a new image, it's first converted to ELA format, and then passed through the trained model to get the classification result.

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV & PIL (for ELA processing)
- scikit-learn (metrics & evaluation)
- Matplotlib / Seaborn (visualization)
- Jupyter Notebook / Kaggle

---

## 📁 Project Structure

tampering-detection/
│
├── data/
│ ├── train/
│ ├── test/
│ └── unseen-images/
│
├── model/
│ └── tamper_detector.h5 # Trained model
│
├── utils/
│ └── ela_processing.py # ELA conversion functions
│
├── main_notebook.ipynb # Model training, testing & visualization
├── predict_tampered.py # Function for predicting new images
├── README.md


---

## ▶️ How to Run

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/tampering-detection.git
cd tampering-detection

Install dependencies
pip install -r requirements.txt

# 🕵️ Image Tampering Detection using Error Level Analysis (ELA)

This project is designed to **detect whether an image is tampered or authentic** using a deep learning model trained on ELA-processed images.

## 📌 Table of Contents
- [Overview](#-overview)
- [How It Works](#️-how-it-works)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [How to Run](#️-how-to-run)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

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

```
tampering-detection/
│
├── data/
│   ├── train/
│   ├── test/
│   └── unseen-images/
│
├── model/
│   └── image_forgery_detection_model.keras     # Trained model
│
├── utils/
│   └── ela_processing.py      # ELA conversion functions
│
├── main_notebook.ipynb        # Model training, testing & visualization
├── predict_tampered.py        # Function for predicting new images
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/tampering-detection.git
cd tampering-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `main_notebook.ipynb` to:
- Preprocess data
- Train the model (or load a pretrained one)
- Evaluate performance
- Predict on new images

### 4. Make a prediction
```python
from predict_tampered import predict_tampered

result = predict_tampered("unseen-images/image.jpg")
print("Prediction:", result)
```

---

## 📊 Results

### Classification Report:
| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Tampered   | 0.75      | 0.86   | 0.80     |
| Authentic  | 0.92      | 0.85   | 0.88     |

**Overall Accuracy: 85%**

### Confusion Matrix:
|                    | Predicted Tampered | Predicted Authentic |
|--------------------|-------------------|-------------------|
| **True Tampered**  | 264               | 43                |
| **True Authentic** | 89                | 504               |

---

## 🚀 Future Improvements

- Replace custom CNN with a pretrained model (e.g. VGG16 + ELA)
- Improve dataset balance (more tampered samples)
- Experiment with multi-channel ELA at different JPEG qualities
- Add GUI for easier user interaction
- Explore other forensic techniques like noise/residual analysis

---

## 📬 Contact

Feel free to reach out via GitHub Issues if you have any questions or suggestions.

---

**⭐ If you found this project helpful, please give it a star!**

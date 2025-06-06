# 🕵️ Image Tampering Detection using Error Level Analysis (ELA)

This project is designed to **detect whether an image is tampered or authentic** using a deep learning model trained on ELA-processed images.

## 📌 Table of Contents
- [Overview](#-overview)
- [How It Works](#️-how-it-works)
- [Technologies Used](#-technologies-used)
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

## ▶️ How to Run

### 🚀 Recommended: Use Kaggle Notebook (Easiest Way)

1. **Open the Kaggle Notebook**
   - Click here: [Image Tampering Detection - Kaggle Notebook](https://www.kaggle.com/code/najmihassan101/image-forgery-detection-with-ela/edit)
   - Or upload the `main_notebook.ipynb` to your Kaggle account

2. **Run All Cells**
   - All dependencies are pre-installed on Kaggle
   - GPU/TPU acceleration is available for free
   - Datasets can be easily imported

3. **Upload Your Own Images**
   - Use Kaggle's file upload feature to test your own images
   - Results will be displayed directly in the notebook

### 💻 Alternative: Local Setup

If you prefer to run locally:

```bash
https://github.com/NajmiHassan/Image_Forgery_Detection_ELA.git

pip install -r requirements.txt
```

Then open `image-forgery-detection-with-ela.ipynb` in Jupyter Notebook or JupyterLab.

---

### 3. Run the notebook
Open `image-forgery-detection-with-ela.ipynb` to:
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

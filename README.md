# Face-Emotion-Detection

# 😊 Face Emotion Detection Using CNN

This project is a deep learning–based emotion recognition system that detects human facial emotions in real-time video streams or images. It uses **Convolutional Neural Networks (CNN)** trained on the **FER-2013 dataset** to classify emotions such as **Happy, Sad, Angry, Surprise, Neutral, Disgust, and Fear**.

---

## 🚀 Features
- Real-time emotion detection using webcam.
- Trained with FER-2013 dataset for high accuracy.
- Emotion classification into 7 classes.
- Simple and clean interface.
- Uses OpenCV for face detection and preprocessing.

---

## 🧠 Tech Stack
- **Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib  
- **Model:** Convolutional Neural Network (CNN)  
- **Dataset:** FER-2013 (Facial Expression Recognition Dataset)

---

## 🧩 Project Structure
   Face_Emotion_Detection/
│
├── train_emotion_model.py # Model training script
├── emotion_model.h5 # Trained CNN model (after training)
├── detection.py # Real-time emotion detection using webcam
├── haarcascade_frontalface_default.xml # Face detection model
├── dataset/ # (Optional) FER-2013 or other dataset
└── README.md # Project documentation


---

## ⚙️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/Face-Emotion-Detection-Using-CNN.git
   cd Face-Emotion-Detection-Using-CNN
   
2. **Install dependencies**
pip install -r requirements.txt

3. **Train the model**
python train_emotion_model.py

4. **Run real-time detection**
python detection.py

**Requirements**

Python 3.8+

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

**If you don’t have them installed, run:**

pip install tensorflow keras opencv-python numpy matplotlib


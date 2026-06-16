# 🌍 EcoConnect Vision: Real-Time Waste Classifier

An intelligent, real-time computer vision application built to classify physical waste as either **Biodegradable** or **Non-Biodegradable** using a live webcam feed. This project serves as the AI vision engine for the broader EcoConnect ecosystem.

## 🚀 Features
* **Real-Time Inference:** Captures live video feed and performs instant material classification.
* **Smart Object Tracking:** Utilizes OpenCV background subtraction (`MOG2`) and contour detection to dynamically draw custom bounding boxes around new objects introduced to the frame.
* **Optimized AI Architecture:** Built on a fine-tuned **MobileNetV2** backbone, optimized for high-speed, lightweight edge deployment without sacrificing accuracy.
* **Glassmorphism UI:** Features a clean, custom Heads-Up Display (HUD) overlay natively rendered through OpenCV.

## 🧠 Technical Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV (cv2)
* **Data Processing:** NumPy
* **Base Model:** MobileNetV2 (Transfer Learning via ImageNet weights)

## 📊 Model Performance
The model was trained from scratch on a custom dataset of 7,000+ images. 
* **Training Accuracy:** ~96%
* **Validation Accuracy:** ~90%

## 💻 How It Works
Instead of relying on heavy object-detection architectures (like YOLO), this project utilizes a highly efficient hybrid approach:
1. **Motion Isolation:** OpenCV detects the specific contours of a new object entering the camera frame.
2. **Dynamic Cropping:** The script isolates and crops *only* the region of interest (the physical object) to remove background noise.
3. **Classification:** The cropped region is passed to the MobileNetV2 brain, which outputs a binary confidence score.
4. **Visual Feedback:** The bounding box instantly shifts to **Green (Biodegradable)** or **Red (Non-Biodegradable)** based on the AI's verdict.

## 🛠️ Local Setup
*(Note: The trained `.keras` model and dataset are excluded from this repository due to file size constraints. You must train the model locally to use the scanner).*

1. Clone the repository.
2. Ensure you have the required dependencies: `pip install tensorflow opencv-python numpy`
3. Run the training notebook to generate the `waste_scanner_model.keras` file.
4. Launch the live scanner cell. Keep your webcam completely still for optimal background subtraction!

---
*Developed by Shaurya Malhotra*
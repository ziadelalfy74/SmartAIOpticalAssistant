# 👓 Smart AI Optical Assistant

A professional-grade, cyber-themed desktop application for Virtual Try-On built using an object-oriented Python architecture, CustomTkinter, OpenCV, and MediaPipe.

---

## 🚀 Features
- **Real-time Face Landmarking**: Powered by MediaPipe Tasks (`FaceLandmarker` in VIDEO mode).
- **5-Step Smart Calibration**: Precise user head calibration for accurate eyewear scaling.
- **Fashion Compatibility Logic**: Matches specific eyewear styles (Square, Round, Aviator, Cat-Eye) based on face shape analysis.
- **Interactive UI**: CustomTkinter-based modern dark-mode cyberpunk interface with live video feed and interactive thumbnail catalog navigation.

---

## 🛠️ Tech Stack
- **Python** (Object-Oriented Architecture)
- **CustomTkinter** (Modern UI Framework)
- **OpenCV & Pillow** (Image processing and manipulation)
- **MediaPipe Tasks** (Computer vision & face tracking)
- **PyTorch** (Underlying tensor engine)

---

## 📂 Project Structure
```text
SmartAIOpticalAssistant/
│
├── gui.py                 # CustomTkinter User Interface & App Loop
├── logic.py               # Optical Engine, MediaPipe, & Geometry Overlays
├── face_landmarker.task   # MediaPipe Face Landmark Model Asset
├── icon.ico               # Application Icon
├── square.png             # Eyewear asset
├── round.png              # Eyewear asset
├── aviator.png            # Eyewear asset
├── cateye.png             # Eyewear asset
└── README.md              # Project Documentation


🚀 Getting Started & Installation

Clone the repository:
git clone [https://github.com/ziadelalfy74/SmartAIOpticalAssistant.git](https://github.com/ziadelalfy74/SmartAIOpticalAssistant.git)
cd SmartAIOpticalAssistant

Install dependencies:
pip install opencv-python mediapipe customtkinter pillow torch


Run the application:
python gui.py
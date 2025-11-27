# 🧠 Realtime OpenCV Human Detection & Counting (YOLO + OpenCV)

This project performs **real-time human detection, tracking, and counting** using **YOLOv8 + OpenCV**.  
It detects all people in a video frame, draws bounding boxes, and shows the **total human count** at the top-left corner.

Perfect for:

- 🎥 Sports analysis (football, cricket, basketball)
- 📹 CCTV and security systems  
- 👥 Crowd monitoring  
- 🧠 Smart vision applications  

---

## 📸 Demo Screenshot

<img width="1786" height="1077" alt="Screenshot 2025-11-27 at 10 29 35" src="https://github.com/user-attachments/assets/06f79adf-bbf4-47c1-bf01-a5fe46f3086a" />

Or display it locally like this:

![Demo Output](assets/demo.png)

---

## 🚀 Features

- ✔ Realtime human detection using **YOLOv8**
- ✔ Displays **total human count**
- ✔ Green bounding boxes around each detected person
- ✔ Works on any video file or webcam
- ✔ Fast and optimized using OpenCV

---

## 🛠 Tech Stack

| Component | Technology |
|----------|------------|
| Detection | YOLOv8 (Ultralytics) |
| Vision Processing | OpenCV |
| Programming | Python |
| Model | pretrained `yolov8n.pt` |

---

## 📂 Project Structure

realtime-opencv-human-detection/
│── human_id.py
│── sort.py
│── requirements.txt
│── assets/
│ └── demo.png
└── README.md

---

## 📦 Installation

### 1️⃣ Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
2️⃣ Install dependencies
pip install -r requirements.txt
▶️ Run the Project
Run on a video
python3 human_id.py
Run using webcam
python3 human_id.py --webcam

🎯 How It Works
YOLOv8 detects humans frame-by-frame
OpenCV draws bounding boxes and labels
Frame is refreshed in real-time
Dynamic counter displays how many humans are present
SORT (optional) can be used for stable tracking

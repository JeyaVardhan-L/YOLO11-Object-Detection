
✅ README.me

markdown
# 🧠 YOLO11 Object Detection – Custom Model Training

This repository contains a complete training and inference pipeline for a custom object detection model using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), trained on a labeled dataset in YOLO format.

You can follow this project to **learn**, **reproduce**, or **build upon** a custom YOLO11-based model for your own use.



## 📁 Project Structure


yolo/
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── labels/          # YOLO format labels
│   └── validation/
│       ├── images/          # Validation images
│       └── labels/          # Validation labels
├── runs/                    # Training logs and weights
│   └── detect/
│       └── train/
│           ├── weights/     # Contains best.pt and last.pt
│           └── results.png  # Training metrics graph
├── data.yaml                # Dataset configuration file
├── train\_val\_split.py       # Script to split data into train/val
├── yolo\_detect.py           # Inference script
├── Training process ----.txt # Step-by-step training log
├── yolov11s.pt              # Pretrained model (optional)
└── testvidairtrack.mp4      # Optional test video (optional)



---

## 🛠️ Setup Instructions

> Make sure you have Python 3.12 and Anaconda installed.

bash
conda create --name yolo11-env python=3.12 -y
conda activate yolo11-env
pip install ultralytics
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


---

## 🧪 Training the Model

Make sure you're inside the `yolo` folder:

bash
cd C:\Users\YourUsername\Documents\yolo


Then run:

bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640


Trained weights will be saved in:


runs/detect/train/weights/


---

## 📷 Running Inference

### Download Inference Script

bash
curl -o yolo_detect.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/yolo_detect.py


### Run on a webcam

bash
python yolo_detect.py --model=runs/detect/train/weights/best.pt --source=usb0


### Run on a video file

bash
python yolo_detect.py --model=runs/detect/train/weights/best.pt --source=testvidairtrack.mp4


---

## ⚠️ Notes

* All data and code is formatted to match Ultralytics YOLOv8/YOLO11 standards
* You can replace the model with `yolov8s.pt`, `yolov5s.pt`, etc., and re-train.

---

## 📎 Credits

* Ultralytics (YOLO): [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* Training guidance and script inspiration: [EJTech Tutorials](https://www.ejtech.io)

---

## 🧠 License

This project is open-source for educational use. Please cite or link if you build from this.


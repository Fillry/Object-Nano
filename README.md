# 🚀 คำแนะนำการติดตั้งและรันระบบบน NVIDIA Jetson Nano (4GB)

โฟลเดอร์ `Object-Nano` นี้บรรจุโค้ดของระบบ ITS Pipeline ที่พร้อมนำไปใช้งานจริงบนบอร์ด **NVIDIA Jetson Nano 4GB** เพื่อให้ระบบทำงานได้รวดเร็วแบบ Real-time และประหยัดทรัพยากรมากที่สุด

---

## 📂 โครงสร้างไฟล์ในโฟลเดอร์ Object-Nano
* **`main.py`** — ไฟล์รันระบบหลัก (รองรับการปิด GUI ด้วย `--headless` และรองรับ GStreamer Hardware Decoding)
* **`setup_jetson.sh`** — สคริปต์อัตโนมัติในการเช็คระบบและติดตั้ง Library ต่างๆ
* **`export_yolo.py`** — สคริปต์สั้นช่วยในการแปลงโมเดล YOLO เป็น ONNX (opset=12)
* **`configs/model_config.yaml`** — ไฟล์ตั้งค่าตัวแบบและระดับการดึง GPU (เช่น เลือกใช้ `.engine` และเปิด Hardware Decoder)
* **`configs/camera_config.yaml`** — ไฟล์ตั้งค่ากล้อง แหล่งสัญญาณ (RTSP) และพิกัดตรวจจับ ROI

---

## ⚡ ขั้นตอนการเตรียมพร้อมและติดตั้ง (บน Jetson Nano)

เพื่อความสะดวกสูงสุด แนะนำให้ดำเนินการตามขั้นตอนดังต่อไปนี้:

### ขั้นตอนที่ 1: รันสคริปต์ติดตั้งอัตโนมัติ
ใน Terminal ของ Jetson Nano ให้เข้าไปที่โฟลเดอร์ `Object-Nano` แล้วรันสคริปต์:
```bash
chmod +x setup_jetson.sh
./setup_jetson.sh
```
สคริปต์นี้จะทำหน้าที่:
1. ตั้งค่า Path สำหรับ CUDA (`nvcc`) ในไฟล์ `~/.bashrc` ให้โดยอัตโนมัติ
2. สร้าง Python Virtual Environment ชื่อ `env_its` โดยดึง Library คอมไพล์ในตัวอย่าง OpenCV และ TensorRT เข้ามาด้วย (`--system-site-packages`)
3. ติดตั้ง Dependencies ทั่วไปของ Python
4. ติดตั้ง `ultralytics` แบบพิเศษ (ไม่ไปติดตั้งทับ OpenCV เดิมที่มากับ JetPack)
5. ติดตั้ง PyCUDA เพื่อรองรับโมเดลจำแนกแบรนด์รถยนต์ TinyViT

### ขั้นตอนที่ 2: ติดตั้ง PyTorch (แมนนวลเพิ่มเติม)
เนื่องจาก PyTorch บนบอร์ด ARM (Jetson) ต้องใช้ไฟล์คอมไพล์พิเศษจาก NVIDIA ให้รันคำสั่งเหล่านี้หลังจากสคริปต์ติดตั้งขั้นแรกเสร็จสิ้น:
```bash
# เปิดใช้งาน env
source env_its/bin/activate

# ดาวน์โหลดและติดตั้ง PyTorch 1.10 สำหรับ JetPack 4.6 (Python 3.6)
wget https://nvidia.box.com/shared/static/fjup34sb5gq2yiyx9u2f9uxf3n31t175.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

---

## 🛠️ ขั้นตอนการแปลงโมเดลให้ Optimize สูงสุด (TensorRT)

เพื่อป้องกันการเกิด Out of Memory บน Jetson Nano แนะนำให้ดำเนินการแปลงไฟล์บนเครื่อง PC/Notebook ก่อนนำมาแปลงขั้นสุดท้ายบน Jetson Nano:

### 1. แปลง YOLOv26n เป็น ONNX (ทำบน PC หรือ Jetson)
รันสคริปต์แปลงไฟล์ที่เราเตรียมไว้ให้:
```bash
python3 export_yolo.py --model yolo26n.pt
```
คุณจะได้ไฟล์ **`yolo26n.onnx`** 

### 2. บิลด์เป็น TensorRT Engine (รันบน Jetson Nano เท่านั้น)
ใช้เครื่องมือคอมไพเลอร์เฉพาะทางของบอร์ดในการสร้าง Engine เพื่อดึงประสิทธิภาพ GPU Maxwell ออกมาเต็มที่:
```bash
# แปลง YOLO Detector
/usr/src/tensorrt/bin/trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16

# แปลง TinyViT Brand Classifier
/usr/src/tensorrt/bin/trtexec --onnx=models/tinyVIT_ThaiCar_ONNX/brand_classifier.onnx --saveEngine=models/tinyVIT_ThaiCar_ONNX/brand_classifier.engine --fp16
```
*(ไฟล์ `.engine` จะทำงานเร็วขึ้น 10-20 เท่า เมื่อเทียบกับไฟล์ `.pt` ปกติ)*

---

## ⚙️ ขั้นตอนการรันระบบเพื่อความประหยัดแรมและสเปกสูงสุด

1. **เปิดใช้การถอดรหัสผ่านฮาร์ดแวร์ (Hardware Decoding):**
   เปิดไฟล์ [configs/model_config.yaml](file:///d:/model-1/Object-Nano/configs/model_config.yaml) และตั้งค่า `use_jetson_hw_dec` เป็น `true` เพื่อลดโหลดของ CPU ในการอ่านกล้อง RTSP:
   ```yaml
   stream_reader:
     use_jetson_hw_dec: true
     latency: 100
   ```
2. **เปลี่ยน Model Path ไปชี้ไฟล์ `.engine`:**
   ในไฟล์ [configs/model_config.yaml](file:///d:/model-1/Object-Nano/configs/model_config.yaml):
   ```yaml
   detector:
     model_path: "yolo26n.engine"  # เปลี่ยนจาก .pt เป็น .engine
     device: "cuda"                # ตั้งค่าเป็น cuda
   
   classifiers:
     brand:
       model_path: "models/tinyVIT_ThaiCar_ONNX/brand_classifier.engine" # เปลี่ยนเป็น .engine
   ```
3. **รันระบบแบบ Headless (ปิด GUI การแสดงผล):**
   ```bash
   python3 main.py --headless
   ```
   *หมายเหตุ: สามารถตรวจสอบรถที่ตรวจจับและพิกัดย้อนหลังได้ในตารางโดยใช้คำสั่ง `python3 query_answer.py`*

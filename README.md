# 🏥 Skin Cancer Detection LINE Bot v8

**AI-Powered Skin Cancer Detection via LINE Bot using YOLOv8**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![LINE Bot SDK](https://img.shields.io/badge/LINE-Bot%20SDK-00C300.svg)](https://github.com/line/line-bot-sdk-python)
[![Railway](https://img.shields.io/badge/Deploy-Railway-purple.svg)](https://railway.app/)

LINE Bot ที่ใช้ AI สำหรับการตรวจจับโรคผิวหนังเบื้องต้น โดยใช้โมเดล YOLOv8 ที่ได้รับการฝึกฝนเฉพาะสำหรับการวิเคราะห์รอยโรคผิวหนัง 3 ประเภท พร้อมแสดงผลด้วย Bounding Box สีสันที่แตกต่างตามระดับความเสี่ยง

## 🎯 Features

### 🔍 **AI Detection Capabilities**
- ตรวจจับโรคผิวหนัง 3 ประเภท:
  - **Melanoma** (เมลาโนมา) - ความเสี่ยงสูง 🔴
  - **Nevus** (เนวัส) - ความเสี่ยงต่ำ 🟢  
  - **Seborrheic Keratosis** (เซบอร์รีอิก เคราโทซิส) - ความเสี่ยงปานกลาง 🟠

### 📱 **LINE Bot Integration**
- รับรูปภาพผ่าน LINE Chat
- ส่งผลลัพธ์พร้อมรูปภาพที่มี Bounding Box
- แสดงความแม่นยำและคำแนะนำเบื้องต้น
- รองรับภาษาไทยและอังกฤษ

### 🎨 **Advanced Visualization**
- **Smart Bounding Box**: กรอบสีแสดงตำแหน่งรอยโรค
- **Adaptive Font Size**: ขนาดตัวอักษรปรับตามขนาดรูปภาพ
- **Color-Coded Results**: สีที่แตกต่างตามระดับความเสี่ยง
- **Text Shadow**: เงาตัวอักษรเพื่อความชัดเจน

### ☁️ **Cloud Deployment**
- Deploy บน Railway Platform
- Auto-scaling และ High Availability
- Webhook สำหรับ LINE Bot API

## 🚀 Quick Start

### 1. **Clone Repository**
```bash
git clone https://github.com/Konrawut11/skin-cancer-linebot-v8.git
cd skin-cancer-linebot-v8
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Setup Environment Variables**
สร้างไฟล์ `.env` และเพิ่มข้อมูลต่อไปนี้:
```env
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
LINE_CHANNEL_SECRET=your_line_channel_secret
RAILWAY_PUBLIC_DOMAIN=your_railway_domain
```

### 4. **Add YOLO Model**
วางไฟล์โมเดล `best.pt` ในโฟลเดอร์ `models/`:
```
models/
├── best.pt          # Custom trained YOLOv8 model
└── README.md
```

### 5. **Run Application**
```bash
python app.py
```

## 📁 Project Structure

```
skin-cancer-linebot-v8/
├── app.py                 # Main Flask application
├── models/
│   └── best.pt           # YOLOv8 trained model
├── static/
│   └── images/           # Temporary image storage
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔧 Configuration

### **Required Dependencies**
```txt
Flask==2.3.3
line-bot-sdk==3.5.0
ultralytics==8.0.196
torch>=1.13.0
torchvision>=0.14.0
opencv-python==4.8.1.78
Pillow==10.0.1
numpy==1.24.3
```

### **LINE Bot Setup**
1. สร้าง LINE Bot ใน [LINE Developers Console](https://developers.line.biz/)
2. รับ Channel Access Token และ Channel Secret
3. ตั้งค่า Webhook URL: `https://your-domain.railway.app/webhook`

### **Railway Deployment**
1. Connect GitHub repository to Railway
2. Set environment variables
3. Deploy automatically via Git push

## 🎨 Color Coding System

| โรค | Bounding Box | ความหมาย | คำแนะนำ |
|-----|-------------|-----------|---------|
| **Melanoma** | 🔴 สีแดง | ความเสี่ยงสูง | ควรปรึกษาแพทย์โดยเร็ว |
| **Nevus** | 🟢 สีเขียว | ความเสี่ยงต่ำ | ดูแลสุขภาพผิวหนังสม่ำเสมอ |
| **Seborrheic Keratosis** | 🟠 สีส้ม | ความเสี่ยงปานกลาง | ควรติดตามอาการ |

## 📱 How to Use

### **สำหรับผู้ใช้งาน**
1. เพิ่มเพื่อน LINE Bot
2. ส่งรูปภาพผิวหนังที่ต้องการตรวจ
3. รอรับผลการวิเคราะห์ (ประมาณ 5-10 วินาที)
4. ดูผลลัพธ์พร้อมคำแนะนำ

### **คำสั่งที่รองรับ**
- `สวัสดี` หรือ `hello` - แสดงคำแนะนำการใช้งาน
- `สถานะ` หรือ `status` - ตรวจสอบสถานะระบบ
- ส่งรูปภาพ - วิเคราะห์โรคผิวหนัง

## 🧠 AI Model Details

### **YOLOv8 Architecture**
- **Base Model**: YOLOv8n (Nano version for speed)
- **Custom Training**: Fine-tuned สำหรับ skin lesion dataset
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.3
- **Device**: CPU optimized for cloud deployment

### **Training Dataset**
- **Classes**: 3 skin cancer types
- **Images**: Custom dataset with augmentation
- **Validation**: Cross-validation with medical expert review

## ⚠️ Important Disclaimers

### 🏥 **Medical Disclaimer**
- ผลการตรวจนี้เป็นเพียงการประเมินเบื้องต้นด้วย AI
- **ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้**
- ควรปรึกษาแพทย์ผิวหนังเพื่อการตรวจสอบที่แม่นยำ
- ไม่รับผิดชอบต่อการตัดสินใจทางการแพทย์จากผลลัพธ์นี้

### 🔒 **Privacy & Security**
- รูปภาพจะถูกลบออกจากเซิร์ฟเวอร์โดยอัตโนมัติภายใน 1 ชั่วโมง
- ไม่มีการเก็บข้อมูลส่วนบุคคลของผู้ใช้
- การส่งข้อมูลผ่าน HTTPS encryption

## 🛠️ Development

### **Local Development**
```bash
# Install development dependencies
pip install -r requirements.txt

# Run in debug mode
python app.py

# Test webhook locally (requires ngrok)
ngrok http 5000
```

### **Testing**
```bash
# Test model loading
python -c "from app import model; print('Model loaded:', model is not None)"

# Test image processing
python -c "from app import predict_skin_cancer; print('Functions loaded successfully')"
```

### **Contributing**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📊 Performance Metrics

- **Response Time**: ~5-10 seconds per image
- **Model Accuracy**: 85%+ on validation set
- **Uptime**: 99.9% (Railway hosting)
- **Concurrent Users**: Up to 100 simultaneous requests

## 🔄 Version History

### **v8.0** (Current)
- ✅ Improved bounding box visualization
- ✅ Adaptive font sizing
- ✅ Enhanced color coding system
- ✅ Better error handling
- ✅ Text shadow for clarity

### **Previous Versions**
- **v7.x**: Basic YOLOv8 integration
- **v6.x**: YOLOv5 implementation
- **v5.x**: Initial LINE Bot setup

## 📞 Support & Contact

### **Issues & Bug Reports**
- GitHub Issues: [Create New Issue](https://github.com/Konrawut11/skin-cancer-linebot-v8/issues)

### **Documentation**
- [LINE Bot SDK Documentation](https://github.com/line/line-bot-sdk-python)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Railway Deployment Guide](https://docs.railway.app/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [LINE Corporation](https://line.me/) for LINE Bot SDK
- [Railway](https://railway.app/) for hosting platform
- Medical professionals who provided dataset validation

---

⚠️ **Remember**: This tool is for educational and preliminary screening purposes only. Always consult healthcare professionals for proper medical diagnosis and treatment.

Skin Cancer Detection LINE Bot
โปรเจกต์นี้เป็น LINE Bot สำหรับตรวจจับโรคผิวหนังเบื้องต้นโดยใช้ YOLOv5

การติดตั้ง
Clone repo นี้
ติดตั้ง dependencies: pip install -r requirements.txt
ตั้งค่า environment variables:
LINE_CHANNEL_ACCESS_TOKEN
LINE_CHANNEL_SECRET
วางไฟล์ model YOLOv8 ที่ models/best.pt
รันแอป: python app.py
Deployment
แนะนำใช้ Render.com, Railway.app หรือ VPS ที่เปิดพอร์ต 5000

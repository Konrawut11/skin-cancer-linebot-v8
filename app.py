import os
import io
import sys
import logging
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,
    ImageSendMessage, QuickReply, QuickReplyButton, MessageAction
)

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ตรวจสอบและ import โมดูลที่จำเป็น
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("NumPy imported successfully")
except ImportError as e:
    logger.error(f"NumPy not available: {e}")
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch imported successfully")
except ImportError as e:
    logger.error(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV imported successfully")
except ImportError as e:
    logger.error(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("PIL imported successfully")
except ImportError as e:
    logger.error(f"PIL not available: {e}")
    PIL_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("Ultralytics imported successfully")
except ImportError as e:
    logger.error(f"Ultralytics not available: {e}")
    ULTRALYTICS_AVAILABLE = False

app = Flask(__name__)

# ตั้งค่า LINE Bot
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    logger.error("LINE credentials not found in environment variables")
    raise ValueError("LINE credentials required")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# โหลด YOLO model (เฉพาะเมื่อมีโมดูลที่จำเป็น)
MODEL_PATH = 'models/best.pt'
model = None

if ULTRALYTICS_AVAILABLE and TORCH_AVAILABLE and NUMPY_AVAILABLE:
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info("Custom model loaded successfully")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}, using YOLOv8n")
            model = YOLO('yolov8n.pt')  # fallback model
            logger.info("Fallback model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
else:
    logger.warning("Required dependencies not available. Model not loaded.")

# คลาสโรคผิวหนัง
SKIN_CANCER_CLASSES = {
    0: "เมลาโนมา (Melanoma)",
    1: "เนวัส (Nevus)", 
    2: "เซบอร์รีอิก เคราโทซิส (Seborrheic Keratosis)"
}

RISK_LEVELS = {
    0: "ความเสี่ยงสูง - ควรปรึกษาแพทย์",
    1: "ความเสี่ยงต่ำ",
    2: "ความเสี่ยงปานกลาง"
}

def check_dependencies():
    """ตรวจสอบโมดูลที่จำเป็น"""
    missing_deps = []
    
    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    if not CV2_AVAILABLE:
        missing_deps.append("opencv-python")
    if not PIL_AVAILABLE:
        missing_deps.append("Pillow")
    if not ULTRALYTICS_AVAILABLE:
        missing_deps.append("ultralytics")
    
    return missing_deps

def download_image_from_line(message_id):
    """ดาวน์โหลดรูปภาพจาก LINE"""
    if not PIL_AVAILABLE:
        logger.error("PIL not available for image processing")
        return None
        
    try:
        message_content = line_bot_api.get_message_content(message_id)
        image_data = io.BytesIO()
        for chunk in message_content.iter_content():
            image_data.write(chunk)
        image_data.seek(0)
        return Image.open(image_data)
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None

def predict_skin_cancer(image):
    """ทำนายโรคผิวหนังจากรูปภาพ"""
    # ตรวจสอบว่ามีโมดูลที่จำเป็นหรือไม่
    missing_deps = check_dependencies()
    if missing_deps:
        error_msg = f"ขาดโมดูลที่จำเป็น: {', '.join(missing_deps)}"
        logger.error(error_msg)
        return None, error_msg
    
    if model is None:
        return None, "โมเดลไม่พร้อมใช้งาน - กรุณาตรวจสอบการติดตั้งโมดูล"
    
    try:
        # แปลง PIL Image เป็น numpy array
        img_array = np.array(image)
        logger.info(f"Image shape: {img_array.shape}")
        
        # ทำการทำนาย
        results = model(img_array)
        
        # ดึงผลลัพธ์
        if len(results) > 0 and len(results[0].boxes) > 0:
            # หา detection ที่มี confidence สูงสุด
            best_detection = results[0].boxes[0]
            class_id = int(best_detection.cls.item())
            confidence = float(best_detection.conf.item())
            
            logger.info(f"Detection result - Class: {class_id}, Confidence: {confidence}")
            
            return {
                'class_id': class_id,
                'class_name': SKIN_CANCER_CLASSES.get(class_id, "Unknown"),
                'confidence': confidence,
                'risk_level': RISK_LEVELS.get(class_id, "ไม่ทราบ")
            }, None
        else:
            return None, "ไม่พบรอยโรคผิวหนังในรูปภาพ"
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

def create_result_message(prediction_result):
    """สร้างข้อความผลลัพธ์"""
    if prediction_result is None:
        return "ไม่สามารถวิเคราะห์รูปภาพได้"
    
    message = f"""🏥 ผลการวิเคราะห์ภาพผิวหนัง

🔍 ผลการตรวจพบ: {prediction_result['class_name']}
📊 ความแม่นยำ: {prediction_result['confidence']:.2%}
⚠️ ระดับความเสี่ยง: {prediction_result['risk_level']}

⚕️ คำแนะนำ:"""
    
    if prediction_result['class_id'] == 0:  # เมลาโนมา
        message += "\n• ควรปรึกษาแพทย์ผิวหนังโดยเร็ว\n• อาจต้องการการตรวจเพิ่มเติม"
    elif prediction_result['class_id'] == 2:  # เซบอร์รีอิก เคราโทซิส
        message += "\n• ควรติดตามอาการ\n• หากมีการเปลี่ยนแปลง ควรพบแพทย์"
    else:  # เนวัส
        message += "\n• ดูแลสุขภาพผิวหนังอย่างสม่ำเสมอ\n• หลีกเลี่ยงแสงแดดจัด"
    
    message += "\n\n⚠️ หมายเหตุ: ผลนี้เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"
    
    return message

def create_dependency_error_message():
    """สร้างข้อความแจ้งเตือนเมื่อขาดโมดูล"""
    missing_deps = check_dependencies()
    
    if not missing_deps:
        return None
    
    message = """❌ ระบบไม่พร้อมใช้งาน

🔧 ขาดโมดูลที่จำเป็น:"""
    
    for dep in missing_deps:
        message += f"\n• {dep}"
    
    message += f"""

📝 วิธีแก้ไข:
pip install {' '.join(missing_deps)}

หรือติดตั้งทั้งหมด:
pip install numpy torch opencv-python Pillow ultralytics

🔄 กรุณาติดตั้งโมดูลและรีสตาร์ทระบบ"""
    
    return message

@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)

    return 'OK', 200

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """จัดการข้อความข้อความ"""
    text = event.message.text.lower()
    
    if 'สวัสดี' in text or 'hello' in text.lower():
        reply_text = """สวัสดีครับ! 👋

ผมเป็นบอทช่วยตรวจโรคผิวหนังเบื้องต้น

📸 วิธีใช้งาน:
1. ส่งรูปภาพผิวหนังที่ต้องการตรวจ
2. รอผลการวิเคราะห์
3. ได้รับคำแนะนำเบื้องต้น

⚠️ สำคัญ: ผลการตรวจเป็นเพียงข้อมูลเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"""
        
    elif 'ช่วยเหลือ' in text or 'help' in text.lower():
        reply_text = """🔧 วิธีใช้งานบอท:

📷 ส่งรูปภาพ:
- ถ่ายรูปผิวหนังที่ชัดเจน
- แสงสว่างเพียงพอ
- ไม่มีสิ่งบดบัง

🔍 การวิเคราะห์:
- ระบบจะตรวจหาความผิดปกติ
- แสดงระดับความเสี่ยง
- ให้คำแนะนำเบื้องต้น

❓ คำถามเพิ่มเติม พิมพ์ "ช่วยเหลือ" """

    elif 'สถานะ' in text or 'status' in text.lower():
        # ตรวจสอบสถานะระบบ
        missing_deps = check_dependencies()
        if missing_deps:
            reply_text = create_dependency_error_message()
        else:
            reply_text = f"""✅ สถานะระบบ: พร้อมใช้งาน

🤖 โมเดล: {'✅ พร้อมใช้งาน' if model is not None else '❌ ไม่พร้อม'}
📦 NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}
🔥 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}
🖼️ OpenCV: {'✅' if CV2_AVAILABLE else '❌'}
🎨 PIL: {'✅' if PIL_AVAILABLE else '❌'}
🚀 Ultralytics: {'✅' if ULTRALYTICS_AVAILABLE else '❌'}

ระบบพร้อมรับรูปภาพเพื่อวิเคราะห์"""
        
    else:
        reply_text = """กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจ 📸

คำสั่งที่ใช้ได้:
• "ช่วยเหลือ" - ดูวิธีใช้งาน
• "สถานะ" - ตรวจสอบสถานะระบบ"""
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """จัดการรูปภาพ"""
    try:
        # ตรวจสอบโมดูลที่จำเป็นก่อน
        missing_deps = check_dependencies()
        if missing_deps:
            error_message = create_dependency_error_message()
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=error_message)
            )
            return
        
        # ส่งข้อความแจ้งว่ากำลังประมวลผล
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🔍 กำลังวิเคราะห์รูปภาพ กรุณารอสักครู่...")
        )
        
        # ดาวน์โหลดรูปภาพ
        image = download_image_from_line(event.message.id)
        if image is None:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text="ไม่สามารถดาวน์โหลดรูปภาพได้ กรุณาลองใหม่")
            )
            return
        
        # ทำการทำนาย
        prediction, error = predict_skin_cancer(image)
        
        if error:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=f"เกิดข้อผิดพลาด: {error}")
            )
            return
        
        # สร้างข้อความผลลัพธ์
        result_message = create_result_message(prediction)
        
        # ส่งผลลัพธ์
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=result_message)
        )
        
    except Exception as e:
        logger.error(f"Error handling image: {e}")
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text="เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")
        )

@app.route("/", methods=['GET'])
def health_check():
    """Health check endpoint"""
    missing_deps = check_dependencies()
    
    return {
        "status": "ok" if not missing_deps else "missing_dependencies",
        "message": "Skin Cancer Detection LINE Bot is running",
        "model_loaded": model is not None,
        "dependencies": {
            "numpy": NUMPY_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "opencv": CV2_AVAILABLE,
            "pil": PIL_AVAILABLE,
            "ultralytics": ULTRALYTICS_AVAILABLE
        },
        "missing_dependencies": missing_deps
    }

@app.route("/install-guide", methods=['GET'])
def install_guide():
    """แสดงคำแนะนำการติดตั้ง"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Installation Guide</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>🔧 คำแนะนำการติดตั้งโมดูล</h1>
        
        <h2>📦 โมดูลที่จำเป็น:</h2>
        <ul>
            <li>numpy - สำหรับการประมวลผลข้อมูล</li>
            <li>torch - PyTorch framework</li>
            <li>opencv-python - การประมวลผลภาพ</li>
            <li>Pillow - การจัดการรูปภาพ</li>
            <li>ultralytics - YOLO model</li>
        </ul>
        
        <h2>⚡ วิธีติดตั้ง:</h2>
        <pre><code>pip install numpy torch opencv-python Pillow ultralytics flask line-bot-sdk</code></pre>
        
        <h2>🐳 หรือใช้ Docker:</h2>
        <pre><code>FROM python:3.9-slim

RUN pip install numpy torch opencv-python Pillow ultralytics flask line-bot-sdk

COPY . /app
WORKDIR /app

CMD ["python", "app.py"]</code></pre>
        
        <h2>📋 requirements.txt:</h2>
        <pre><code>numpy>=1.21.0
torch>=1.9.0
opencv-python>=4.5.0
Pillow>=8.3.0
ultralytics>=8.0.0
flask>=2.0.0
line-bot-sdk>=2.0.0</code></pre>
    </body>
    </html>
    """

if __name__ == "__main__":
    # ตรวจสอบโมดูลที่จำเป็นตอนเริ่มต้น
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"⚠️ Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("📝 Run: pip install " + " ".join(missing_deps))
    else:
        print("✅ All dependencies available")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

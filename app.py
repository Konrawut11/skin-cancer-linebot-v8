import os
import io
import torch
import cv2
import numpy as np
from flask import Flask, request, abort
from PIL import Image
import logging
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,
    ImageSendMessage, QuickReply, QuickReplyButton, MessageAction
)
import requests
from ultralytics import YOLO

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ตั้งค่า LINE Bot
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    logger.error("LINE credentials not found in environment variables")
    raise ValueError("LINE credentials required")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# โหลด YOLOv5 model
MODEL_PATH = 'models/best.pt'
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}, using YOLOv5s")
        model = YOLO('yolov5s.pt')  # fallback model
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

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

def download_image_from_line(message_id):
    """ดาวน์โหลดรูปภาพจาก LINE"""
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
    if model is None:
        return None, "Model not available"
    
    try:
        # แปลง PIL Image เป็น numpy array
        img_array = np.array(image)
        
        # ทำการทำนาย
        results = model(img_array)
        
        # ดึงผลลัพธ์
        if len(results) > 0 and len(results[0].boxes) > 0:
            # หา detection ที่มี confidence สูงสุด
            best_detection = results[0].boxes[0]
            class_id = int(best_detection.cls.item())
            confidence = float(best_detection.conf.item())
            
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

@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)

    return 'OK', 200  # <<< ตรงนี้สำคัญ

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
        
    else:
        reply_text = """กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจ 📸

หรือพิมพ์ "ช่วยเหลือ" เพื่อดูวิธีใช้งาน"""
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """จัดการรูปภาพ"""
    try:
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
    return {
        "status": "ok",
        "message": "Skin Cancer Detection LINE Bot is running",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

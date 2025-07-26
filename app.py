import os
import io
import sys
import logging
from flask import Flask, request, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,
    ImageSendMessage, QuickReply, QuickReplyButton, MessageAction
)
import tempfile
import base64
import time
import random

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ตรวจสอบและ import โมดูลที่จำเป็น
try:
    import numpy as np
    test_array = np.array([1, 2, 3])
    NUMPY_AVAILABLE = True
    logger.info(f"NumPy imported successfully - version: {np.__version__}")
except Exception as e:
    logger.error(f"NumPy not available or not working: {e}")
    NUMPY_AVAILABLE = False

try:
    import torch
    if NUMPY_AVAILABLE:
        test_tensor = torch.tensor([1, 2, 3])
        test_numpy = test_tensor.cpu().numpy()
        logger.info(f"PyTorch-NumPy integration working")
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch imported successfully - version: {torch.__version__}")
except Exception as e:
    logger.error(f"PyTorch not available or NumPy integration failed: {e}")
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV imported successfully")
except ImportError as e:
    logger.error(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
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

# ตั้งค่า LINE Bot จาก Railway Environment Variables
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

# ตั้งค่า BASE_URL อัตโนมัติสำหรับ Railway
RAILWAY_STATIC_URL = os.getenv('RAILWAY_STATIC_URL')
if RAILWAY_STATIC_URL:
    BASE_URL = RAILWAY_STATIC_URL
else:
    # Fallback สำหรับ Railway
    BASE_URL = f"https://{os.getenv('RAILWAY_SERVICE_NAME', 'your-app')}.up.railway.app"

logger.info(f"BASE_URL set to: {BASE_URL}")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    logger.error("LINE credentials not found in environment variables")
    raise ValueError("LINE credentials required")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# โหลด YOLO model
MODEL_PATH = 'models/best.pt'
model = None

if ULTRALYTICS_AVAILABLE and TORCH_AVAILABLE and NUMPY_AVAILABLE:
    try:
        # ตั้งค่า device เป็น CPU เพื่อหลีกเลี่ยงปัญหา
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            model.to('cpu')
            logger.info("Custom model loaded successfully on CPU")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}, using YOLOv8n")
            model = YOLO('yolov8n.pt')
            model.to('cpu')
            logger.info("Fallback model loaded successfully on CPU")
            
        # ทดสอบโมเดล
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_results = model(test_img, device='cpu', verbose=False)
            logger.info("Model test prediction successful")
        except Exception as test_error:
            logger.warning(f"Model test failed: {test_error}")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
else:
    missing_modules = []
    if not ULTRALYTICS_AVAILABLE:
        missing_modules.append("ultralytics")
    if not TORCH_AVAILABLE:
        missing_modules.append("torch")
    if not NUMPY_AVAILABLE:
        missing_modules.append("numpy")
    logger.warning(f"Required dependencies not available: {missing_modules}. Model not loaded.")

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

CLASS_COLORS = {
    0: (255, 0, 0),    # แดง
    1: (0, 255, 0),    # เขียว
    2: (255, 165, 0)   # ส้ม
}

def save_image_temporarily(image, filename):
    """บันทึกรูปภาพชั่วคราวสำหรับ Railway"""
    try:
        # สร้างโฟลเดอร์ temp ถ้ายังไม่มี
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # บันทึกรูปภาพ
        file_path = os.path.join(temp_dir, filename)
        
        # แปลงเป็น RGB ก่อนบันทึกเป็น JPEG
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # บันทึกด้วยคุณภาพที่เหมาะสม
        image.save(file_path, 'JPEG', quality=85, optimize=True)
        
        # ตรวจสอบว่าไฟล์ถูกสร้างแล้ว
        if not os.path.exists(file_path):
            raise Exception("ไม่สามารถสร้างไฟล์รูปภาพได้")
        
        # สร้าง URL สำหรับ Railway
        image_url = f"{BASE_URL}/temp_images/{filename}"
        
        logger.info(f"Image saved: {file_path}, URL: {image_url}")
        return image_url, file_path
        
    except Exception as e:
        logger.error(f"Error saving image temporarily: {e}")
        return None, None

def cleanup_old_images():
    """ลบไฟล์รูปภาพเก่า"""
    try:
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            return
        
        current_time = time.time()
        max_age = 3600  # 1 hour
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > max_age:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        logger.error(f"Error removing file {filename}: {e}")
                        
    except Exception as e:
        logger.error(f"Error in cleanup_old_images: {e}")

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

def draw_bounding_boxes(image, results):
    """วาด bounding boxes บนรูปภาพ"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                
                color = CLASS_COLORS.get(class_id, (255, 255, 0))
                
                # วาด bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                class_name = SKIN_CANCER_CLASSES.get(class_id, "Unknown")
                label = f"{class_name}\n{confidence:.2%}"
                
                if font:
                    draw.rectangle([x1, y1-30, x2, y1], fill=color)
                    draw.text((x1+5, y1-25), label, fill=(255, 255, 255), font=font)
        
        return img_with_boxes
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {e}")
        return image

def predict_skin_cancer(image):
    """ทำนายโรคผิวหนังจากรูปภาพ"""
    if model is None:
        return None, None, "โมเดลไม่พร้อมใช้งาน"
    
    try:
        # ทดสอบ NumPy
        try:
            test_array = np.array([1, 2, 3])
            logger.info("NumPy test passed")
        except Exception as np_error:
            logger.error(f"NumPy test failed: {np_error}")
            return None, None, f"NumPy ไม่ทำงานอย่างถูกต้อง: {str(np_error)}"
        
        # แปลงรูปภาพ
        try:
            img_array = np.array(image)
            logger.info(f"Image converted to array successfully - shape: {img_array.shape}")
        except Exception as img_error:
            logger.error(f"Failed to convert image to array: {img_error}")
            return None, None, f"ไม่สามารถแปลงรูปภาพ: {str(img_error)}"
        
        # ทำการทำนาย
        try:
            if hasattr(model, 'to'):
                model.to('cpu')
            
            results = model(img_array, device='cpu', verbose=False)
            logger.info("Model prediction completed")
            
        except Exception as model_error:
            logger.error(f"Model prediction failed: {model_error}")
            return None, None, f"การทำนายล้มเหลว: {str(model_error)}"
        
        # วาด bounding boxes
        img_with_boxes = draw_bounding_boxes(image, results)
        
        # ดึงผลลัพธ์
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = 0
            best_conf = 0
            
            for i, box in enumerate(boxes):
                conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i
            
            best_detection = boxes[best_idx]
            class_id = int(best_detection.cls.item()) if hasattr(best_detection.cls, 'item') else int(best_detection.cls)
            confidence = float(best_detection.conf.item()) if hasattr(best_detection.conf, 'item') else float(best_detection.conf)
            
            prediction_result = {
                'class_id': class_id,
                'class_name': SKIN_CANCER_CLASSES.get(class_id, "Unknown"),
                'confidence': confidence,
                'risk_level': RISK_LEVELS.get(class_id, "ไม่ทราบ"),
                'total_detections': len(boxes)
            }
            
            return prediction_result, img_with_boxes, None
        else:
            return None, img_with_boxes, "ไม่พบรอยโรคผิวหนังในรูปภาพ"
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None, f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

def create_result_message(prediction_result):
    """สร้างข้อความผลลัพธ์"""
    if prediction_result is None:
        return "ไม่สามารถวิเคราะห์รูปภาพได้"
    
    message = f"""🏥 ผลการวิเคราะห์ภาพผิวหนัง

🔍 ผลการตรวจพบ: {prediction_result['class_name']}
📊 ความแม่นยำ: {prediction_result['confidence']:.2%}
⚠️ ระดับความเสี่ยง: {prediction_result['risk_level']}
📍 จำนวนจุดที่ตรวจพบ: {prediction_result.get('total_detections', 1)} จุด

⚕️ คำแนะนำ:"""
    
    if prediction_result['class_id'] == 0:  # เมลาโนมา
        message += "\n• ควรปรึกษาแพทย์ผิวหนังโดยเร็ว\n• อาจต้องการการตรวจเพิ่มเติม"
    elif prediction_result['class_id'] == 2:  # เซบอร์รีอิก เคราโทซิส
        message += "\n• ควรติดตามอาการ\n• หากมีการเปลี่ยนแปลง ควรพบแพทย์"
    else:  # เนวัส
        message += "\n• ดูแลสุขภาพผิวหนังอย่างสม่ำเสมอ\n• หลีกเลี่ยงแสงแดดจัด"
    
    message += "\n\n🎯 กรอบสีในรูปภาพ:"
    message += "\n🔴 แดง = ความเสี่ยงสูง (เมลาโนมา)"
    message += "\n🟢 เขียว = ความเสี่ยงต่ำ (เนวัส)"
    message += "\n🟠 ส้ม = ความเสี่ยงปานกลาง (เซบอร์รีอิก เคราโทซิส)"
    
    message += "\n\n⚠️ หมายเหตุ: ผลนี้เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"
    
    return message

# Routes
@app.route("/")
def home():
    return """
    <h1>LINE Bot Skin Cancer Detection</h1>
    <p>Status: Active</p>
    <p>Model: """ + ("Loaded" if model is not None else "Not Loaded") + """</p>
    <p>BASE_URL: """ + BASE_URL + """</p>
    """

@app.route("/temp_images/<filename>")
def serve_temp_image(filename):
    """ให้บริการรูปภาพชั่วคราว"""
    try:
        return send_from_directory('temp_images', filename)
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        abort(404)

@app.route("/health")
def health_check():
    """ตรวจสอบสถานะระบบ"""
    try:
        status = {
            "status": "ok",
            "model_loaded": model is not None,
            "numpy_available": NUMPY_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "cv2_available": CV2_AVAILABLE,
            "pil_available": PIL_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "base_url": BASE_URL
        }
        return status, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.before_request
def before_request():
    """ทำความสะอาดไฟล์เก่าก่อนประมวลผล request"""
    if random.randint(1, 10) == 1:
        cleanup_old_images()

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
3. ได้รับรูปภาพพร้อม bounding box
4. ได้รับคำแนะนำเบื้องต้น

🎯 สีของกรอบ:
🔴 แดง = ความเสี่ยงสูง
🟢 เขียว = ความเสี่ยงต่ำ  
🟠 ส้ม = ความเสี่ยงปานกลาง

⚠️ สำคัญ: ผลการตรวจเป็นเพียงข้อมูลเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"""
        
    elif 'สถานะ' in text or 'status' in text.lower():
        reply_text = f"""✅ สถานะระบบ: พร้อมใช้งาน

🤖 โมเดล: {'✅ พร้อมใช้งาน' if model is not None else '❌ ไม่พร้อม'}
📦 NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}
🔥 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}
🖼️ OpenCV: {'✅' if CV2_AVAILABLE else '❌'}
🎨 PIL: {'✅' if PIL_AVAILABLE else '❌'}
🚀 Ultralytics: {'✅' if ULTRALYTICS_AVAILABLE else '❌'}
🌐 Base URL: {BASE_URL}

🎯 ฟีเจอร์ Bounding Box: ✅ พร้อมใช้งาน

ระบบพร้อมรับรูปภาพเพื่อวิเคราะห์และแสดงผลด้วย bounding box"""
        
    else:
        reply_text = """กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจ 📸

คำสั่งที่ใช้ได้:
• "สถานะ" - ตรวจสอบสถานะระบบ

🎯 ระบบจะส่งรูปภาพกลับพร้อมกรอบสีแสดงผลการตรวจ"""
    
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
            TextSendMessage(text="🔍 กำลังวิเคราะห์รูปภาพและสร้าง bounding box กรุณารอสักครู่...")
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
        prediction, img_with_boxes, error = predict_skin_cancer(image)
        
        if error:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=f"เกิดข้อผิดพลาด: {error}")
            )
            return
        
        # สร้างข้อความผลลัพธ์
        result_message = create_result_message(prediction)
        
        # บันทึกรูปภาพที่มี bounding box และส่งกลับ
        if img_with_boxes is not None:
            try:
                # สร้างชื่อไฟล์ unique
                filename = f"result_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
                
                # บันทึกรูปภาพชั่วคราว
                image_url, file_path = save_image_temporarily(img_with_boxes, filename)
                
                if image_url:
                    # ส่งรูปภาพและข้อความผลลัพธ์
                    messages = [
                        ImageSendMessage(
                            original_content_url=image_url,
                            preview_image_url=image_url
                        ),
                        TextSendMessage(text=result_message)
                    ]
                    
                    line_bot_api.push_message(event.source.user_id, messages)
                    logger.info(f"Image sent successfully: {image_url}")
                else:
                    # ถ้าส่งรูปภาพไม่ได้ ส่งแค่ข้อความ
                    line_bot_api.push_message(
                        event.source.user_id,
                        TextSendMessage(text=f"{result_message}\n\n⚠️ ไม่สามารถส่งรูปภาพที่วิเคราะห์แล้วได้")
                    )
                    
            except Exception as img_error:
                logger.error(f"Error sending image: {img_error}")
                # ส่งแค่ข้อความผลลัพธ์
                line_bot_api.push_message(
                    event.source.user_id,
                    TextSendMessage(text=f"{result_message}\n\n⚠️ ส่งรูปภาพไม่ได้: {str(img_error)}")
                )
        else:
            # ไม่มีรูปภาพ ส่งแค่ข้อความ
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=result_message)
            )
            
    except Exception as e:
        logger.error(f"Error in handle_image_message: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
        )

if __name__ == "__main__":
    print("🚀 Starting LINE Bot Server on Railway...")
    print(f"📡 BASE_URL: {BASE_URL}")
    print(f"🤖 Model Status: {'✅ Loaded' if model is not None else '❌ Not Loaded'}")
    
    # สร้างโฟลเดอร์ temp_images
    if not os.path.exists("temp_images"):
        os.makedirs("temp_images")
    
    # รันแอป
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

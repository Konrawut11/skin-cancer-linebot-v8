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
import tempfile
import base64

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ตรวจสอบและ import โมดูลที่จำเป็น
try:
    import numpy as np
    # ทดสอบ NumPy functionality
    test_array = np.array([1, 2, 3])
    NUMPY_AVAILABLE = True
    logger.info(f"NumPy imported successfully - version: {np.__version__}")
except Exception as e:
    logger.error(f"NumPy not available or not working: {e}")
    NUMPY_AVAILABLE = False

try:
    import torch
    # ทดสอบ PyTorch-NumPy integration
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
        # ตั้งค่า device เป็น CPU เพื่อหลีกเลี่ยงปัญหา
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # บังคับใช้ CPU
        
        # ปิด warning ที่ไม่จำเป็น
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            model.to('cpu')  # บังคับใช้ CPU
            logger.info("Custom model loaded successfully on CPU")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}, using YOLOv8n")
            model = YOLO('yolov8n.pt')  # fallback model
            model.to('cpu')  # บังคับใช้ CPU
            logger.info("Fallback model loaded successfully on CPU")
            
        # ทดสอบโมเดลด้วยรูปภาพเล็กๆ
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

# สีสำหรับ bounding box แต่ละคลาส
CLASS_COLORS = {
    0: (255, 0, 0),    # แดง สำหรับเมลาโนมา (ความเสี่ยงสูง)
    1: (0, 255, 0),    # เขียว สำหรับเนวัส (ความเสี่ยงต่ำ)
    2: (255, 165, 0)   # ส้ม สำหรับเซบอร์รีอิก เคราโทซิส (ความเสี่ยงปานกลาง)
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

def draw_bounding_boxes(image, results):
    """วาด bounding boxes บนรูปภาพ"""
    try:
        # แปลง PIL Image เป็น RGB หากจำเป็น
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # สร้าง copy ของรูปภาพ
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # พยายามโหลดฟอนต์ถ้าเป็นไปได้
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # ดึงพิกัด bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # ดึงคลาสและ confidence
                class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                
                # เลือกสี
                color = CLASS_COLORS.get(class_id, (255, 255, 0))
                
                # วาด bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # เตรียมข้อความ
                class_name = SKIN_CANCER_CLASSES.get(class_id, "Unknown")
                label = f"{class_name}\n{confidence:.2%}"
                
                # วาดพื้นหลังสำหรับข้อความ
                if font:
                    bbox = draw.textbbox((x1, y1-40), label, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x1, y1-40), label, fill=(255, 255, 255), font=font)
                else:
                    # ใช้ฟอนต์เริ่มต้นถ้าโหลดฟอนต์ไม่ได้
                    draw.rectangle([x1, y1-30, x2, y1], fill=color)
                    draw.text((x1+5, y1-25), label, fill=(255, 255, 255))
        
        return img_with_boxes
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {e}")
        return image

def upload_image_to_line(image):
    """อัปโหลดรูปภาพไปยัง LINE (ใช้วิธี temporary file)"""
    try:
        # สร้างไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # บันทึกรูปภาพเป็น JPEG
            image.save(temp_file.name, 'JPEG', quality=90)
            temp_file_path = temp_file.name
        
        # อ่านไฟล์และแปลงเป็น base64
        with open(temp_file_path, 'rb') as f:
            image_data = f.read()
        
        # ลบไฟล์ชั่วคราว
        os.unlink(temp_file_path)
        
        # สร้าง URL สำหรับรูปภาพ (ในที่นี้เราจะใช้วิธีอื่น)
        # เนื่องจาก LINE ต้องการ URL ที่เข้าถึงได้จากภายนอก
        # คุณอาจต้องอัปโหลดไปยัง cloud storage เช่น AWS S3, Google Cloud Storage
        
        return image_data
        
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return None

def save_image_temporarily(image, filename):
    """บันทึกรูปภาพชั่วคราวและคืนค่า URL"""
    try:
        # สร้างโฟลเดอร์ temp ถ้ายังไม่มี
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # บันทึกรูปภาพ
        file_path = os.path.join(temp_dir, filename)
        image.save(file_path, 'JPEG', quality=90)
        
        # คืนค่า URL (คุณต้องแทนที่ด้วย URL จริงของเซิร์ฟเวอร์)
        # ตัวอย่าง: https://your-server.com/temp_images/filename
        base_url = os.getenv('BASE_URL', 'https://your-server.com')
        image_url = f"{base_url}/temp_images/{filename}"
        
        return image_url, file_path
        
    except Exception as e:
        logger.error(f"Error saving image temporarily: {e}")
        return None, None

def predict_skin_cancer(image):
    """ทำนายโรคผิวหนังจากรูปภาพ"""
    # ตรวจสอบว่ามีโมดูลที่จำเป็นหรือไม่
    missing_deps = check_dependencies()
    if missing_deps:
        error_msg = f"ขาดโมดูลที่จำเป็น: {', '.join(missing_deps)}"
        logger.error(error_msg)
        return None, None, error_msg
    
    if model is None:
        return None, None, "โมเดลไม่พร้อมใช้งาน - กรุณาตรวจสอบการติดตั้งโมดูล"
    
    try:
        # ทดสอบ NumPy functionality ก่อนใช้งาน
        try:
            test_array = np.array([1, 2, 3])
            logger.info("NumPy test passed")
        except Exception as np_error:
            logger.error(f"NumPy test failed: {np_error}")
            return None, None, f"NumPy ไม่ทำงานอย่างถูกต้อง: {str(np_error)}"
        
        # ลองใช้ image อย่างปลอดภัย
        try:
            # แปลง PIL Image เป็น numpy array
            img_array = np.array(image)
            logger.info(f"Image converted to array successfully - shape: {img_array.shape}")
        except Exception as img_error:
            logger.error(f"Failed to convert image to array: {img_error}")
            return None, None, f"ไม่สามารถแปลงรูปภาพเป็น array: {str(img_error)}"
        
        # ลองทำการทำนายด้วย error handling ที่ดีขึ้น
        try:
            # กำหนด device เป็น CPU เพื่อหลีกเลี่ยงปัญหา CUDA
            if hasattr(model, 'to'):
                model.to('cpu')
            
            # ทำการทำนาย
            results = model(img_array, device='cpu', verbose=False)
            logger.info("Model prediction completed")
            
        except Exception as model_error:
            logger.error(f"Model prediction failed: {model_error}")
            # ลองใช้วิธีอื่น
            try:
                # ลองแปลงเป็น PIL Image ก่อน
                if hasattr(image, 'convert'):
                    rgb_image = image.convert('RGB')
                    results = model(rgb_image, device='cpu', verbose=False)
                    logger.info("Model prediction with PIL image successful")
                else:
                    raise Exception("Cannot convert image format")
            except Exception as fallback_error:
                logger.error(f"Fallback prediction failed: {fallback_error}")
                return None, None, f"การทำนายล้มเหลว: {str(fallback_error)}"
        
        # วาด bounding boxes บนรูปภาพ
        img_with_boxes = draw_bounding_boxes(image, results)
        
        # ดึงผลลัพธ์
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            # หา detection ที่มี confidence สูงสุด
            boxes = results[0].boxes
            best_idx = 0
            best_conf = 0
            
            # หา box ที่มี confidence สูงสุด
            for i, box in enumerate(boxes):
                conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i
            
            best_detection = boxes[best_idx]
            class_id = int(best_detection.cls.item()) if hasattr(best_detection.cls, 'item') else int(best_detection.cls)
            confidence = float(best_detection.conf.item()) if hasattr(best_detection.conf, 'item') else float(best_detection.conf)
            
            logger.info(f"Detection result - Class: {class_id}, Confidence: {confidence}")
            
            prediction_result = {
                'class_id': class_id,
                'class_name': SKIN_CANCER_CLASSES.get(class_id, "Unknown"),
                'confidence': confidence,
                'risk_level': RISK_LEVELS.get(class_id, "ไม่ทราบ"),
                'total_detections': len(boxes)
            }
            
            return prediction_result, img_with_boxes, None
        else:
            logger.info("No detections found")
            return None, img_with_boxes, "ไม่พบรอยโรคผิวหนังในรูปภาพ หรือความชัดของรูปภาพไม่เพียงพอ"
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
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

# เพิ่ม route สำหรับให้บริการรูปภาพ
@app.route("/temp_images/<filename>")
def serve_temp_image(filename):
    """ให้บริการรูปภาพชั่วคราว"""
    try:
        from flask import send_from_directory
        return send_from_directory('temp_images', filename)
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        abort(404)

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
        
    elif 'ช่วยเหลือ' in text or 'help' in text.lower():
        reply_text = """🔧 วิธีใช้งานบอท:

📷 ส่งรูปภาพ:
- ถ่ายรูปผิวหนังที่ชัดเจน
- แสงสว่างเพียงพอ
- ไม่มีสิ่งบดบัง

🔍 การวิเคราะห์:
- ระบบจะตรวจหาความผิดปกติ
- วาดกรอบสีแสดงผลลัพธ์
- แสดงระดับความเสี่ยง
- ให้คำแนะนำเบื้องต้น

🎯 ความหมายสีกรอบ:
🔴 แดง = เมลาโนมา (ความเสี่ยงสูง)
🟢 เขียว = เนวัส (ความเสี่ยงต่ำ)
🟠 ส้ม = เซบอร์รีอิก เคราโทซิส (ปานกลาง)

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

🎯 ฟีเจอร์ Bounding Box: ✅ พร้อมใช้งาน

ระบบพร้อมรับรูปภาพเพื่อวิเคราะห์และแสดงผลด้วย bounding box"""
        
    else:
        reply_text = """กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจ 📸

คำสั่งที่ใช้ได้:
• "ช่วยเหลือ" - ดูวิธีใช้งาน
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
        
        # บันทึกรูปภาพที่มี bounding box ชั่วคราว
        import uuid
        filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        image_url, file_path = save_image_temporarily(img_with_boxes, filename)
        
        if image_url and file_path:
            # ส่งรูปภาพที่มี bounding box
            try:
                line_bot_api.push_message(
                    event.source.user_id,
                    ImageSendMessage(
                        original_content_url=image_url,
                        preview_image_url=image_url
                    )
                )
                
                # รอสักครู่แล้วลบไฟล์ (ทำในพื้นหลัง)
                import threading
                def cleanup_file():
                    import time
                    time.sleep(300)  # รอ 5 นาที
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up temp file: {file_path}")
                
                cleanup_thread = threading.Thread(target=cleanup_file)
                cleanup_thread.daemon = True
                cleanup_thread.start()
                
            except Exception as img_error:
                logger.error(f"Error sending image: {img_error}")
                line_bot_api.push_message(
                    event.source.user_id,
                    TextSendMessage(text="ไม่สามารถส่งรูปภาพผลลัพธ์ได้ กรุณาตรวจสอบการตั้งค่าเซิร์ฟเวอร์")
                )
        
        # สร้างและส่งข้อความผลลัพธ์
        result_message = create_result_message(prediction)
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
        "message": "Skin Cancer Detection LINE Bot with Bounding Box is running",
        "model_loaded": model is not None,
        "features": {
            "bounding_box": True,
            "image_analysis": True,
            "risk_assessment": True
        },
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
        <title>Installation Guide - Bounding Box Version</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .feature { background: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .command { background: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; }
            .color-demo { display: inline-block; width: 20px; height: 20px; margin-right: 10px; }
        </style>
    </head>
    <body>
        <h1>🔧 คำแนะนำการติดตั้งโมดูล - Bounding Box Version</h1>
        
        <div class="feature">
            <h2>✨ ฟีเจอร์ใหม่</h2>
            <ul>
                <li>🎯 แสดงผลด้วย Bounding Box บนรูปภาพ</li>
                <li>🎨 สีกรอบตามระดับความเสี่ยง</li>
                <li>📊 แสดงจำนวนจุดที่ตรวจพบ</li>
                <li>🖼️ ส่งรูปภาพผลลัพธ์กลับไปยัง LINE</li>
            </ul>
        </div>
        
        <h2>🎨 ความหมายสีกรอบ:</h2>
        <p><span class="color-demo" style="background: red;"></span>🔴 แดง = เมลาโนมา (ความเสี่ยงสูง)</p>
        <p><span class="color-demo" style="background: green;"></span>🟢 เขียว = เนวัส (ความเสี่ยงต่ำ)</p>
        <p><span class="color-demo" style="background: orange;"></span>🟠 ส้ม = เซบอร์รีอิก เคราโทซิส (ความเสี่ยงปานกลาง)</p>
        
        <h2>📦 โมดูลที่จำเป็น:</h2>
        <ul>
            <li>numpy - สำหรับการประมวลผลข้อมูล</li>
            <li>torch - PyTorch framework</li>
            <li>opencv-python - การประมวลผลภาพ</li>
            <li>Pillow - การจัดการรูปภาพและวาดกรอบ</li>
            <li>ultralytics - YOLO model</li>
            <li>flask - Web framework</li>
            <li>line-bot-sdk - LINE Bot SDK</li>
        </ul>
        
        <h2>⚡ วิธีติดตั้ง:</h2>
        <div class="command">
pip install numpy torch opencv-python Pillow ultralytics flask line-bot-sdk
        </div>
        
        <h2>🌐 การตั้งค่าเซิร์ฟเวอร์:</h2>
        <p>ตั้งค่าตัวแปร environment:</p>
        <div class="command">
export BASE_URL="https://your-server.com"<br>
export LINE_CHANNEL_ACCESS_TOKEN="your_access_token"<br>
export LINE_CHANNEL_SECRET="your_channel_secret"
        </div>
        
        <h2>📁 โครงสร้างไดเรกทอรี:</h2>
        <div class="command">
project/<br>
├── app.py<br>
├── models/<br>
│   └── best.pt<br>
└── temp_images/ (จะถูกสร้างอัตโนมัติ)
        </div>
        
        <h2>🐳 Docker Configuration:</h2>
        <div class="command">
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy torch opencv-python Pillow ultralytics flask line-bot-sdk

COPY . /app
WORKDIR /app

RUN mkdir -p temp_images

EXPOSE 5000

CMD ["python", "app.py"]
        </div>
        
        <h2>📋 requirements.txt:</h2>
        <div class="command">
numpy>=1.21.0<br>
torch>=1.9.0<br>
opencv-python>=4.5.0<br>
Pillow>=8.3.0<br>
ultralytics>=8.0.0<br>
flask>=2.0.0<br>
line-bot-sdk>=2.0.0
        </div>
        
        <div class="feature">
            <h2>⚠️ สำคัญ:</h2>
            <ul>
                <li>ตั้งค่า BASE_URL ให้ถูกต้องเพื่อให้ LINE เข้าถึงรูปภาพได้</li>
                <li>สร้างโฟลเดอร์ temp_images สำหรับเก็บรูปภาพชั่วคราว</li>
                <li>ตรวจสอบให้แน่ใจว่าเซิร์ฟเวอร์สามารถให้บริการไฟล์ static ได้</li>
            </ul>
        </div>
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
    
    # สร้างโฟลเดอร์ temp_images ถ้ายังไม่มี
    temp_dir = "temp_images"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"📁 Created directory: {temp_dir}")
    
    print("🎯 Bounding Box feature enabled")
    print("🎨 Color coding: Red=High Risk, Green=Low Risk, Orange=Medium Risk")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

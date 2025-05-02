
# في main.py
from fastapi import FastAPI, File, UploadFile, Depends  # تصحيح VoloadFile إلى UploadFile
import joblib  # تصحيح joslib إلى joblib
import os
import tempfile
import traceback
from fastapi.middleware.cors import CORSMiddleware  # تصحيح COSSMiddleware إلى CORSMiddleware
from auth import router as auth_router, load_users  # تصحيح app.math إلى auth
from audio_processing import extract_features
# ✅ تحميل المستخدمين عند بدء التشغيل
load_users()

# تحميل الموديل
model_path = "KNN_xbestx_model.pkl"
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {str(e)}")

app = FastAPI()

# ✅ إعداد CORS علشان تربط بالواجهة الأمامية
  app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # استخدام قائمة بدلاً من سلسلة نصية
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ تضمين مسارات المصادقة في التطبيق
app.include_router(auth_router)

@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # حفظ الملف مؤقتًا
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            temp_audio.write(await file.read())

        # التحقق من تحميل النموذج
        if model is None:
            return {"error": "❌ Model not loaded"}

        # استخراج الميزات
        features = extract_features(temp_audio_path)

        # التنبؤ باستخدام النموذج
        prediction = model.predict(features)[0]

        # إرجاع النتيجة
        result = "Correct" if prediction == 1 else "Incorrect"
        return {"result": result}

    except ValueError as ve:
        traceback.print_exc()
        return {"error": str(ve)}
    except Exception as e:
        traceback.print_exc()  # ✅ يطبع الخطأ الكامل في الكونسول
        return {"error": f"⚠️ Error analyzing audio: {repr(e)}"}
    finally:
        # حذف الملف المؤقت
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

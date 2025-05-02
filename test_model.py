import joblib
from app.audio_processing import extract_features

# تحميل النموذج
model = joblib.load("models/KNN_xbestx_model.pkl")

# تحديد مسار الملف الصوتي اللي انت عايز تختبره
audio_path = "C:/Users/AAAA/Downloads/test_audio.wav"  # غيّر المسار هنا لملفك الصوتي

print(f"Testing with audio file: {audio_path}")

# استخراج الميزات
features = extract_features(audio_path)
print(f"Extracted features: {features}")

# التنبؤ
prediction = model.predict(features)[0]
print(f"Prediction: {prediction}")

# طباعة النتيجة
if prediction == 1:
    print("✅ Correct")
else:
    print("❌ Incorrect")

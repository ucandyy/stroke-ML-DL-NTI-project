from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import joblib
import pandas as pd
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


# إعداد Flask
app = Flask(__name__)
CORS(app)
def clip_outliers(x):
    return x   # أو الكود الأصلي اللي كنتي عاملاه

model = joblib.load("logistic_f1_pipeline.pkl")

# رفع الصور
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تحميل الموديلات
logistic_model = joblib.load("logistic_f1_pipeline.pkl")
deep_model = load_model("deep.h5")

# --- Page1: Welcome ---
@app.route('/')
def home():
    return render_template("page1.html")


# --- Page2: إدخال البيانات + Logistic Regression ---
@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        data = {
            "gender": request.form.get("gender"),
            "age": request.form.get("age"),
            "hypertension": request.form.get("hypertension"),
            "heart_disease": request.form.get("heart_disease"),
            "ever_married": request.form.get("ever_married"),
            "work_type": request.form.get("work_type"),
            "Residence_type": request.form.get("residence_type"),
            "avg_glucose_level": request.form.get("avg_glucose_level"),
            "bmi": request.form.get("bmi"),
            "smoking_status": request.form.get("smoking_status"),
        }

        # تحويل البيانات DataFrame
        input_df = pd.DataFrame([data])

        # التنبؤ
        pred = logistic_model.predict(input_df)[0]
        result = "⚠️ Stroke Risk" if pred == 1 else "✅ No Stroke Risk"

        return render_template("page4.html", result=result)

    return render_template("page2.html")


# --- Page3: رفع صورة + Deep Learning ---
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == "":
            return "❌ No file selected"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # تجهيز الصورة للتنبؤ
        img = image.load_img(filepath, target_size=(224, 224))  # غيري المقاس حسب تدريب الموديل
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # التنبؤ
        prediction = deep_model.predict(img_array)

        # أسماء الكلاسات (غيريها حسب تدريب الموديل)
        class_labels = ["Normal", "Haemorrhagic", " Ischemic"]

        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_index]
        confidence = float(np.max(prediction)) * 100

        result = f"✅ Prediction: {predicted_label} ({confidence:.2f}% confidence)"

        return render_template("page4.html", result=result)

    return render_template("page3.html")



if __name__ == '__main__':
    app.run(debug=True)

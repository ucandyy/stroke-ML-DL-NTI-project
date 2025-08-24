from flask import Flask, render_template, request
from flask_cors import CORS
import joblib
import pandas as pd
import pickle

# --------- لازم تعيدي تعريف أي function custom اتعملت في التدريب ---------
def clip_outliers(x):
    return x  # 👈 حطي نفس الكود اللي كنتي كاتباه في التدريب
# -------------------------------------------------------------------------

# إنشاء Flask App
app = Flask(__name__)
CORS(app)  # يفتح CORS عشان أي فرونت يتصل بالسيرفر

# تحميل الموديل

import joblib
model=joblib.load("logistic_f1_pipeline.pkl")


@app.route('/')
def home():
    return render_template("page1.html")

@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        # استقبل الداتا من الفورم
        data = {
    "gender": request.form.get("gender"),
    "age": request.form.get("age"),
    "hypertension": request.form.get("hypertension"),
    "heart_disease": request.form.get("heart_disease"),
    "ever_married": request.form.get("ever_married"),
    "work_type": request.form.get("work_type"),
    "Residence_type": request.form.get("residence_type"),        # غيّرت الاسم
    "avg_glucose_level": request.form.get("avg_glucose"),        # غيّرت الاسم
    "bmi": request.form.get("bmi"),
    "smoking_status": request.form.get("smoking"),               # غيّرت الاسم
}


        # حوّلي البيانات لـ DataFrame بنفس أسماء الأعمدة اللي الموديل متعود عليها
        input_df = pd.DataFrame([data])

        # اعملي prediction
        pred = model.predict(input_df)[0]
        result = "⚠️ Stroke Risk" if pred == 1 else "✅ No Stroke Risk"

        return f"Prediction: {result}"

    return render_template("page2.html")

if __name__ == '__main__':
    app.run(debug=True)

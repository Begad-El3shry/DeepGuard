import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# تصميم واجهة المستخدم
st.set_page_config(page_title="Deepfake Detector", page_icon="🔍")
st.title("🛡️ Deepfake Comparison Tool")
st.markdown("### Upload two images to find the fake one")

# تحميل الموديل (تأكد من وجود ملف deepfake_model.h5 في المجلد الرئيسي للـ Colab)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('deepfake_model.h5')

model = load_model()

def process_and_predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return float(prediction[0][0])

# تقسيم الشاشة لرفع صورتين
col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("Upload Image A", type=['jpg', 'jpeg', 'png'])
    if file1:
        st.image(file1, caption="Image A", use_container_width=True)

with col2:
    file2 = st.file_uploader("Upload Image B", type=['jpg', 'jpeg', 'png'])
    if file2:
        st.image(file2, caption="Image B", use_container_width=True)

# زر التحليل المطور
if file1 and file2:
    if st.button("🚀 Start Deep Analysis"):
        with st.spinner('Analyzing facial patterns...'):
            img1 = Image.open(file1).convert('RGB')
            img2 = Image.open(file2).convert('RGB')
            
            # الحصول على النتائج
            score1 = process_and_predict(img1)
            score2 = process_and_predict(img2)
            
            st.divider()
            
            # حساب الفرق بين الصورتين (Margin)
            diff = abs(score1 - score2)
            
            # حالة 1: لو الصورتين قريبين جداً من بعض (نتيجة غير حاسمة)
            if diff < 0.10: 
                st.warning(f"⚠️ **Inconclusive Result:** Both images have very similar patterns (Diff: {diff:.2%}). It's hard to distinguish which one is manipulated.")
            
            # حالة 2: مقارنة واضحة
            
            col_res1, col_res2 = st.columns(2)
            
            if score1 > score2:
                with col_res1:
                    st.error(f"🚨 **IMAGE A: FAKE**")
                    st.metric(label="Fake Probability", value=f"{score1:.2%}", delta="High Risk")
                with col_res2:
                    st.success(f"✅ **IMAGE B: REAL**")
                    st.metric(label="Fake Probability", value=f"{score2:.2%}", delta="-Low Risk", delta_color="normal")
            elif score2 == score1:
                st.info("ℹ️ **Both images have identical fake probabilities. they might be the same image or equally manipulated.**")
            else:
                with col_res1:
                    st.success(f"✅ **IMAGE A: REAL**")
                    st.metric(label="Fake Probability", value=f"{score1:.2%}", delta="-Low Risk", delta_color="normal")
                with col_res2:
                    st.error(f"🚨 **IMAGE B: FAKE**")
                    st.metric(label="Fake Probability", value=f"{score2:.2%}", delta="High Risk")

            # نصيحة تقنية للمستخدم
            st.info("💡 **AI Insight:** The model focuses on skin texture inconsistencies and eye-light reflection patterns.")
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

# زر التحليل
if file1 and file2:
    if st.button("Detect Fake Image"):
        img1 = Image.open(file1).convert('RGB')
        img2 = Image.open(file2).convert('RGB')
        
        score1 = process_and_predict(img1)
        score2 = process_and_predict(img2)
        
        st.divider()
        if score1 > score2:
            st.error(f"⚠️ **Result:** Image A is likely the FAKE (Score: {score1:.2%})")
            st.success(f"✅ **Result:** Image B is likely REAL (Score: {score2:.2%})")
        else:
            st.error(f"⚠️ **Result:** Image B is likely the FAKE (Score: {score2:.2%})")
            st.success(f"✅ **Result:** Image A is likely REAL (Score: {score1:.2%})")

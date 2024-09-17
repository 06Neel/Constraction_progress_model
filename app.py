import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Load the pre-trained model
def load_model():
    model = tf.keras.models.load_model('path_to_your_model.h5')
    return model

model = load_model()

# Streamlit app
st.title('Construction Activity Assessment')

st.sidebar.header('Upload Images')
uploaded_img1 = st.sidebar.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
uploaded_img2 = st.sidebar.file_uploader("Upload Image 2 (for comparison)", type=["png", "jpg", "jpeg"])

if uploaded_img1 and uploaded_img2:
    def classify_image(image):
        img = load_img(image, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        prediction = model.predict(img_array)[0][0]
        class_name = "Developed" if prediction > 0.5 else "Underdeveloped"
        confidence = abs(prediction - 0.5) * 200
        return class_name, confidence, prediction
    
    # Load and classify images
    img1_class, img1_conf, img1_pred = classify_image(uploaded_img1)
    img2_class, img2_conf, img2_pred = classify_image(uploaded_img2)

    # Display images and classifications
    st.image(uploaded_img1, caption=f"Image 1: {img1_class} (Confidence: {img1_conf:.2f}%)", use_column_width=True)
    st.image(uploaded_img2, caption=f"Image 2: {img2_class} (Confidence: {img2_conf:.2f}%)", use_column_width=True)

    # Compare construction progress
    img1_array = img_to_array(load_img(uploaded_img1, target_size=(224, 224)))
    img2_array = img_to_array(load_img(uploaded_img2, target_size=(224, 224)))

    img1_gray = cv2.cvtColor(img1_array.astype('uint8'), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_array.astype('uint8'), cv2.COLOR_RGB2GRAY)

    data_range = 255

    similarity_index, _ = ssim(img1_gray, img2_gray, data_range=data_range, full=True)
    
    if img1_class == img2_class:
        st.write(f"Structural Similarity Index: {similarity_index:.2f}")
        if similarity_index > 0.8:
            st.write("The images are very similar, suggesting little to no progress between them.")
        else:
            st.write("The images show some differences, suggesting some progress or changes.")
    else:
        developed_confidence = img1_conf if img1_class == "Developed" else img2_conf
        underdeveloped_confidence = img2_conf if img1_class == "Developed" else img1_conf
        progress = (underdeveloped_confidence / developed_confidence) * 100
        underdeveloped_image = "Image 2" if img1_class == "Developed" else "Image 1"
        st.write(f"{underdeveloped_image} (Underdeveloped) is approximately {progress:.2f}% complete compared to the Developed image.")
    
    # Calculate overall progress based on model predictions
    overall_progress = (img1_pred + img2_pred) / 2 * 100
    st.write(f"Overall estimated progress of the construction site: {overall_progress:.2f}%")

    # Plot comparison results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img1_array / 255.0)
    ax1.set_title(f"Image 1: {img1_class}\nConfidence: {img1_conf:.2f}%")
    ax1.axis('off')
    
    ax2.imshow(img2_array / 255.0)
    ax2.set_title(f"Image 2: {img2_class}\nConfidence: {img2_conf:.2f}%")
    ax2.axis('off')
    
    plt.suptitle(f"Construction Progress Comparison: {overall_progress:.2f}%\n", fontsize=16)
    st.pyplot(fig)

elif uploaded_img1 or uploaded_img2:
    st.error("Please upload both images for comparison.")

else:
    st.warning("Upload images to proceed.")

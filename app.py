import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image

# Model loading function
@st.cache(allow_output_mutation=True)
def load_model():
    # Model architecture using a pre-trained model (Transfer Learning)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    # Load trained weights (replace 'model.h5' with your trained model path if needed)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# For inference on a new image
def classify_image(image, model):
    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    prediction = model.predict(img_array)[0][0]
    class_name = "Developed" if prediction > 0.5 else "Underdeveloped"
    confidence = abs(prediction - 0.5) * 200
    return class_name, confidence, prediction

# New function to compare construction progress with visual output
def compare_construction_progress(image1, image2, model):
    # Classify both images
    class1, confidence1, prediction1 = classify_image(image1, model)
    class2, confidence2, prediction2 = classify_image(image2, model)

    st.write("Comparison results:")
    if class1 == class2:
        st.write(f"Both images are classified as {class1}")
        # Calculate structural similarity
        img1_gray = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
        similarity_index, _ = ssim(img1_gray, img2_gray, full=True)
        st.write(f"Structural Similarity Index: {similarity_index:.2f}")
        if similarity_index > 0.8:
            st.write("The images are very similar, suggesting little to no progress between them.")
        else:
            st.write("The images show some differences, suggesting some progress or changes.")
    else:
        st.write("The images are classified differently.")
        developed_confidence = confidence1 if class1 == "Developed" else confidence2
        underdeveloped_confidence = confidence2 if class1 == "Developed" else confidence1
        progress = (underdeveloped_confidence / developed_confidence) * 100
        underdeveloped_image = "Image 2" if class1 == "Developed" else "Image 1"
        st.write(f"{underdeveloped_image} (Underdeveloped) is approximately {progress:.2f}% complete compared to the Developed image.")
    
    # Calculate overall progress based on model predictions
    overall_progress = (prediction1 + prediction2) / 2 * 100
    st.write(f"\nOverall estimated progress of the construction site: {overall_progress:.2f}%")

    # Display images with classification results
    #st.image(image1, caption=f"Image 1: {class1} ({confidence1:.2f}% confidence)", use_column_width=True)
    #st.image(image2, caption=f"Image 2: {class2} ({confidence2:.2f}% confidence)", use_column_width=True)

# Streamlit application starts here
st.title("Construction Site Development Classifier")

# Load the model
model = load_model()

st.write("Upload two images to compare construction progress")

# Allow upload of multiple file types (png, jpg, jpeg)
uploaded_file1 = st.file_uploader("Choose the first image...", type=["png", "jpg", "jpeg"])
uploaded_file2 = st.file_uploader("Choose the second image...", type=["png", "jpg", "jpeg"])

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Convert uploaded files to PIL Image
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    # Compare construction progress between the two images
    compare_construction_progress(image1, image2, model)
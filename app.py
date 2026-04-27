import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("disease_model.h5")

classes = [
    "Brain Tumor - No",
    "Brain Tumor - Yes",
    "Normal Chest",
    "Pneumonia",
    "Skin Cancer"
]

def predict_image(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)

    disease = classes[index]
    confidence = float(np.max(prediction))*100

    return f"{disease} | Confidence: {confidence:.2f}%"

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Medical Image Disease Detection",
    description="Upload a medical image to detect disease using Deep Learning"
)

interface.launch()
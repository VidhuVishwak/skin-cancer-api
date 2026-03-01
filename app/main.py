import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

# Load model once at startup
model = tf.keras.models.load_model("models/skin_cancer_binary_model.h5")

IMG_SIZE = (224, 224)

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "Skin Cancer Detection API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed = preprocess_image(image)

    prediction = model.predict(processed)[0][0]

    if prediction >= 0.5:
        label = "malignant"
    else:
        label = "benign"

    return {
        "prediction": label,
        "probability": float(prediction),
        "disclaimer": "This AI prediction is not a medical diagnosis. Please consult a dermatologist."
    }
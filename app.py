from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import numpy as np
import cv2 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os

app = FastAPI()

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model

model = unet((256,256,3))
model.load_weights('model.h5')

W, H = 256, 256


def predict_tumor(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))  ## [H, w, 3]
    x = image / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    pred = np.squeeze(pred, axis=-1)
    pred = pred >= 0.5
    pred = pred.astype(np.int32)
    return pred


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("index.html")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    image_path = "temp_image.png"
    image.save(image_path)
    
    tumor_prediction = predict_tumor(image_path)
    
    # Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    axes[1].imshow(tumor_prediction, cmap="gray")
    axes[1].axis('off')
    axes[1].set_title('Tumor Prediction')
    
    # Save the combined image
    combined_image_path = "combined_image.png"
    plt.savefig(combined_image_path)
    plt.close()  # Close the figure to release memory
    
    return FileResponse(combined_image_path)
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps

app = Flask(__name__)


model = tf.keras.models.load_model("model.h5")

def preprocess_image(image_data):
    try:
        
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = image.convert("L")  
        image = ImageOps.invert(image)  
        image = image.resize((28, 28))  

        image = np.array(image) / 255.0  
        image = image.reshape(1, 28, 28, 1)  
        return image
    except Exception as e:
        print("Error processing image:", str(e))
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]  
        processed_image = preprocess_image(data)

        if processed_image is None:
            return jsonify({"error": "Invalid image data"})

        prediction = np.argmax(model.predict(processed_image))

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

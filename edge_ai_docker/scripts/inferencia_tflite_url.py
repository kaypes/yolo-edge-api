import numpy as np
import tensorflow as tf
import requests
import time
from PIL import Image
from io import BytesIO
import sys

modelo = sys.argv[1]

print("Modelo:", modelo)

# URL
url = input("Digite a URL da imagem: ")

# Download
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Resize
img = img.resize((224, 224))

img_array = np.array(img)

# Normalização
img_array = img_array.astype(np.float32)
img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# Interpreter
interpreter = tf.lite.Interpreter(model_path=modelo)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inferência
inicio = time.time()

interpreter.set_tensor(input_details[0]["index"], img_array)
interpreter.invoke()

pred = interpreter.get_tensor(output_details[0]["index"])

fim = time.time()

# Decodifica
decoded = tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=3)

print("\nResultado:")
for item in decoded[0]:
    print(f"{item[1]} - {item[2]*100:.2f}%")

print(f"\nTempo: {fim - inicio:.4f} s")

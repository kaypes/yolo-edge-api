import tensorflow as tf
import numpy as np
import requests
import time
from PIL import Image
from io import BytesIO

# Solicita URL
url = input("Digite a URL da imagem: ")

# Download da imagem
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Redimensiona
img = img.resize((224, 224))

# Converte para array
img_array = np.array(img)

# Pré-processamento
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Batch
img_array = np.expand_dims(img_array, axis=0)

# Modelo
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Tempo
inicio = time.time()

# Inferência
pred = model.predict(img_array)

fim = time.time()

# Decodifica
decoded = tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=3)

print("\nResultado:")
for item in decoded[0]:
    print(f"{item[1]} - {item[2]*100:.2f}%")

print(f"\nTempo: {fim - inicio:.4f} s")

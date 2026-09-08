import tensorflow as tf

print("Carregando modelo...")

# Carrega MobileNet
model = tf.keras.applications.MobileNetV2(weights="imagenet")

print("Convertendo para TFLite FP32...")

# Cria conversor
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Converte
tflite_model = converter.convert()

# Salva
with open("modelo/mobilenet_fp32.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo FP32 salvo em modelo/mobilenet_fp32.tflite")

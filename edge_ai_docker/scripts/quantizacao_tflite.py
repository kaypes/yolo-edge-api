import tensorflow as tf

print("Carregando modelo...")

model = tf.keras.applications.MobileNetV2(weights="imagenet")

print("Aplicando quantização...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Otimização padrão
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant = converter.convert()

with open("modelo/mobilenet_int8.tflite", "wb") as f:
    f.write(tflite_quant)

print("Modelo INT8 salvo em modelo/mobilenet_int8.tflite")

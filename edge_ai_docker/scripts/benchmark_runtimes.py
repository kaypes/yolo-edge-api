"""
edge_ai_docker/scripts/benchmark_runtimes.py
Aula 8 - Atividade 3: benchmark rigoroso dos 3 runtimes (TF base, TFLite FP32,
TFLite INT8) para preencher a tabela comparativa com médias, não uma única
medição ruidosa (mesmo padrão do monitoring/bench_imgsz.py da Aula 7).

Baixa a imagem UMA vez (mesma URL para os três runtimes, controle experimental
pedido pelo próprio PDF) e roda N repetições por runtime, descartando as
--warmup primeiras (custo de carregar o modelo/alocar tensores na primeira
chamada não deve contaminar a média de inferência).

Modo --preproc:
  tf    (padrão) -- reproduz o script do PDF: TF base usa preprocess_input
        ([-1,1]), TFLite usa /255.0 ([0,1]). É o que vale nota, mas mistura
        o efeito da quantização com o efeito de pré-processamentos diferentes.
  fixed -- aplica preprocess_input também no TFLite, isolando o efeito real
        da quantização (usado para responder com rigor a pergunta 3 da análise).
"""
import argparse
import statistics
import time
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--fp32", default="modelo/mobilenet_fp32.tflite")
    p.add_argument("--int8", default="modelo/mobilenet_int8.tflite")
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--preproc", choices=["tf", "fixed"], default="tf")
    return p.parse_args()


def load_image(url):
    response = requests.get(url, timeout=15)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    return np.array(img)


def prep_tf_style(img_array):
    """Como inferencia_url.py: preprocess_input, escala [-1, 1]."""
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    return np.expand_dims(arr, axis=0)


def prep_tflite_style(img_array):
    """Como inferencia_tflite_url.py: /255.0, escala [0, 1] (o bug documentado)."""
    arr = img_array.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def bench_keras(model, img_array, runs, warmup):
    batch = prep_tf_style(img_array)
    times = []
    decoded_last = None
    for i in range(runs):
        t0 = time.time()
        pred = model.predict(batch, verbose=0)
        t1 = time.time()
        if i >= warmup:
            times.append(t1 - t0)
        decoded_last = tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=3)[0]
    return times, decoded_last


def bench_tflite(model_path, img_array, runs, warmup, preproc):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    batch = prep_tf_style(img_array) if preproc == "fixed" else prep_tflite_style(img_array)
    batch = batch.astype(np.float32)

    times = []
    decoded_last = None
    for i in range(runs):
        t0 = time.time()
        interpreter.set_tensor(input_details[0]["index"], batch)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])
        t1 = time.time()
        if i >= warmup:
            times.append(t1 - t0)
        decoded_last = tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=3)[0]
    return times, decoded_last


def report(name, times, decoded):
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"\n=== {name} ===")
    print(f"Tempo médio: {mean*1000:.2f} ms  (desvio: {stdev*1000:.2f} ms, n={len(times)})")
    print("Top-3:")
    for _, label, conf in decoded:
        print(f"  {label} - {conf*100:.2f}%")
    return mean, stdev


def main():
    args = parse_args()
    print(f"Baixando imagem: {args.url}")
    img_array = load_image(args.url)

    print(f"\n--- pré-processamento: {args.preproc} ---")

    print("\nCarregando TensorFlow (Keras) MobileNetV2...")
    keras_model = tf.keras.applications.MobileNetV2(weights="imagenet")
    t_keras, d_keras = bench_keras(keras_model, img_array, args.runs, args.warmup)
    m_keras, _ = report("TensorFlow Base", t_keras, d_keras)

    print(f"\nCarregando TFLite FP32 ({args.fp32})...")
    t_fp32, d_fp32 = bench_tflite(args.fp32, img_array, args.runs, args.warmup, args.preproc)
    m_fp32, _ = report("TFLite FP32", t_fp32, d_fp32)

    print(f"\nCarregando TFLite INT8 ({args.int8})...")
    t_int8, d_int8 = bench_tflite(args.int8, img_array, args.runs, args.warmup, args.preproc)
    m_int8, _ = report("TFLite INT8", t_int8, d_int8)

    print("\n=== Resumo (variação % vs TensorFlow Base) ===")
    print(f"TensorFlow Base: {m_keras*1000:.2f} ms  (referência)")
    print(f"TFLite FP32:     {m_fp32*1000:.2f} ms  ({(m_fp32-m_keras)/m_keras*100:+.1f}%)")
    print(f"TFLite INT8:     {m_int8*1000:.2f} ms  ({(m_int8-m_keras)/m_keras*100:+.1f}%)")


if __name__ == "__main__":
    main()

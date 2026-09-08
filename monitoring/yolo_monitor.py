"""
monitoring/yolo_monitor.py
Aula 7 - Monitoramento da inferência YOLO.

Roda o modelo YOLO em loop contínuo sobre um vídeo (simulando um stream) e
expõe o tempo de cada inferência como métrica Prometheus na porta indicada,
para que o Grafana Alloy colete e envie ao Grafana Cloud.

Baseado no roteiro do PDF, com duas diferenças justificadas:
  - --imgsz e --port como argumentos de linha de comando, em vez de editar
    a constante IMG_SIZE no arquivo entre uma medição e outra (o PDF manda
    editar o arquivo à mão -- aqui vira só um --imgsz diferente na chamada).
  - Sem cv2.imshow()/destroyAllWindows(): a Raspberry Pi roda headless
    (só SSH, sem monitor). O próprio PDF documenta essa remoção como
    esperada em ambiente headless.
"""
import argparse
import time

import cv2
import torch
from prometheus_client import Gauge, start_http_server
from ultralytics import YOLO

# PyTorch 2.6+ mudou o padrao de weights_only para True, o que bloqueia o
# carregamento de checkpoints do Ultralytics. Mesmo patch usado em
# app/model.py, scripts/validate_model.py e preprocessing/utils/evaluate.py.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

INFERENCE_TIME = Gauge(
    "yolo_inference_time_seconds",
    "Tempo de inferência do YOLO em segundos",
)


def parse_args():
    p = argparse.ArgumentParser(description="Monitoramento de inferência YOLO com métricas Prometheus")
    p.add_argument("--video", default="monitoring/videos/transito.mp4")
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640, help="Resolução de inferência (testar 640 e 320)")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--classes", type=int, nargs="+", default=[2], help="2 = carro (COCO)")
    p.add_argument("--port", type=int, default=8000, help="Porta do servidor de métricas Prometheus")
    return p.parse_args()


def main():
    args = parse_args()

    start_http_server(args.port)
    print(f"[yolo_monitor] Métricas Prometheus em http://localhost:{args.port}/metrics")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {args.video}")

    print(f"[yolo_monitor] imgsz={args.imgsz} modelo={args.model} video={args.video}")

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        start = time.time()
        model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=args.classes,
            verbose=False,
        )
        end = time.time()

        inference_time = end - start
        INFERENCE_TIME.set(inference_time)


if __name__ == "__main__":
    main()

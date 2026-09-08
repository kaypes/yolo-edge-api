#!/usr/bin/env python3
"""Roda um modelo .eim da Edge Impulse sobre uma imagem parada e desenha as
bounding boxes detectadas. Usado como fallback quando o runner CLI não
consegue abrir o stream de câmera via GStreamer/libcamera."""
import sys

import cv2
from edge_impulse_linux.image import ImageImpulseRunner

model_path = sys.argv[1]
image_path = sys.argv[2]
output_path = sys.argv[3]

with ImageImpulseRunner(model_path) as runner:
    model_info = runner.init()
    labels = model_info["model_parameters"]["labels"]
    print(f"Modelo carregado: {model_info['project']['name']} — classes {labels}")

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    features, cropped = runner.get_features_from_image(img_rgb)
    res = runner.classify(features)

    print(f"Tempo de inferência (DSP+classificação): {res['timing']['dsp'] + res['timing']['classification']} ms")

    detections = res["result"].get("bounding_boxes", [])
    print(f"{len(detections)} objeto(s) detectado(s):")
    for bb in detections:
        print(f"  {bb['label']} ({bb['value']*100:.1f}%) em x={bb['x']} y={bb['y']} w={bb['width']} h={bb['height']}")

    out_img = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    for bb in detections:
        x, y, w, h = bb["x"], bb["y"], bb["width"], bb["height"]
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out_img,
            f"{bb['label']} {bb['value']*100:.0f}%",
            (x, max(y - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    out_img = cv2.resize(out_img, (out_img.shape[1] * 4, out_img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, out_img)
    print(f"Salvo em {output_path}")

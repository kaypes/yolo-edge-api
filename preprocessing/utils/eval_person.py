"""
preprocessing/utils/eval_person.py

Avaliação complementar com sinal estatístico real.

Por que existe: o yolov8n.pt é pré-treinado no COCO (80 classes genéricas) e o
epi-v1 tem ['Capacete', 'Colete', 'Pessoa'] (índices 0, 1, 2). Os índices não
correspondem entre si, então avaliar o modelo COCO direto contra o epi-v1 produz
mAP ≈ 0.009 (ruído) e os deltas dos experimentos E1-E4 ficam sem significado.

A única classe com correspondência semântica real é Pessoa ↔ person (índice 0 do
COCO). Este módulo constrói um dataset temporário mantendo apenas as anotações de
Pessoa, remapeadas para o índice 0, e restringe a validação a essa classe
(classes=[0]). O resultado é um mAP não-degenerado, onde os deltas de
pré-processamento podem ser lidos de verdade.

O data.yaml temporário declara os 80 nomes do COCO (não apenas 'person') para que
nc do dataset case com nc do modelo -- só os labels é que ficam restritos a person.
"""
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

DATASET_YAML = "dataset/exports/epi-v1/data.yaml"
MODEL_PATH = "models/yolov8n.pt"

PESSOA_CLASS_ID = 2   # índice de 'Pessoa' no epi-v1
COCO_PERSON_ID = 0    # índice de 'person' no COCO


def evaluate_person_only(
    preprocess_fn: Optional[Callable] = None,
    label: str = "baseline",
    split: str = "val",
    verbose: bool = False,
) -> dict:
    """
    Avalia o mAP@0.5 considerando apenas a classe Pessoa (remapeada para person).

    Args:
        preprocess_fn: função que recebe frame BGR e devolve o frame transformado.
                       Se None, as imagens originais são usadas sem alteração.
        label:         identificador do experimento no log.

    Returns:
        dict com map50, map50_95 e o tempo médio de pré-processamento.
    """
    model = YOLO(MODEL_PATH)

    split_dirname = {"val": "valid", "test": "test", "train": "train"}.get(split, split)
    dataset_dir = Path(DATASET_YAML).parent
    src_images_dir = dataset_dir / split_dirname / "images"
    src_labels_dir = dataset_dir / split_dirname / "labels"
    images = sorted(src_images_dir.glob("*.jpg")) + sorted(src_images_dir.glob("*.png"))

    safe_label = "".join(c if c.isalnum() else "_" for c in label)
    tmp_root = Path("preprocessing/outputs/_tmp_eval_person") / safe_label
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_images_dir = tmp_root / "images"
    tmp_labels_dir = tmp_root / "labels"
    tmp_images_dir.mkdir(parents=True, exist_ok=True)
    tmp_labels_dir.mkdir(parents=True, exist_ok=True)

    preproc_times = []
    for img_path in images:
        frame = cv2.imread(str(img_path))

        if preprocess_fn is not None:
            t0 = time.perf_counter()
            frame_out = preprocess_fn(frame)
            preproc_times.append((time.perf_counter() - t0) * 1000)
        else:
            frame_out = frame

        cv2.imwrite(str(tmp_images_dir / img_path.name), frame_out)

        # Mantém só as linhas de Pessoa e remapeia o índice para person (0)
        label_src = src_labels_dir / f"{img_path.stem}.txt"
        kept_lines = []
        if label_src.exists():
            with open(label_src) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    if int(parts[0]) == PESSOA_CLASS_ID:
                        parts[0] = str(COCO_PERSON_ID)
                        kept_lines.append(" ".join(parts))
        # Arquivo de label sempre é escrito, mesmo vazio (imagem sem pessoa =
        # imagem negativa, o que é informação válida para a métrica)
        with open(tmp_labels_dir / f"{img_path.stem}.txt", "w") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")

    # data.yaml com os 80 nomes do COCO para casar nc do modelo
    coco_names = [model.names[i] for i in sorted(model.names)]
    tmp_yaml_cfg = {
        "path": str(tmp_root.resolve()),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": coco_names,
    }
    tmp_yaml = tmp_root / "data.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(tmp_yaml_cfg, f)

    metrics = model.val(data=str(tmp_yaml), split="val", classes=[COCO_PERSON_ID],
                        verbose=verbose)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    preproc_ms = float(np.mean(preproc_times)) if preproc_times else 0.0
    print(f"[PESSOA] [{label:32s}] mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}  "
          f"preproc={preproc_ms:.1f}ms")
    return {"label": label, "map50": map50, "map50_95": map50_95,
            "preproc_ms": preproc_ms}

"""
Experimento E4: equalização de histograma vs CLAHE.
Usa o dataset epi-v1-dark (imagens escurecidas por gamma) para
simular condições de iluminação adversa.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, '.')
import preprocessing.utils.eval_person as evp_module
import preprocessing.utils.evaluate as ev_module
from preprocessing.utils.eval_person import evaluate_person_only
from preprocessing.utils.evaluate import evaluate_pipeline

DATASET_DARK = "dataset/exports/epi-v1-dark/data.yaml"


def equalize_hist_hsv(frame: np.ndarray) -> np.ndarray:
    """
    Equalização global no canal V (Value) do espaço HSV.
    Preserva as cores (H, S) e age apenas na luminância.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)


def equalize_hist_lab(frame: np.ndarray) -> np.ndarray:
    """
    Equalização global no canal L* do espaço LAB.
    Espaço LAB é perceptualmente uniforme — L* é a luminância real.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def clahe_hsv(frame: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    CLAHE no canal V do HSV.
    clipLimit controla a amplificação máxima de contraste.
    tileGridSize divide a imagem em blocos para equalização local.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_cl = clahe.apply(v)
    hsv_cl = cv2.merge([h, s, v_cl])
    return cv2.cvtColor(hsv_cl, cv2.COLOR_HSV2RGB)


def clahe_lab(frame: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE no canal L* do LAB — geralmente superior ao HSV para detecção."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_cl = clahe.apply(l)
    lab_cl = cv2.merge([l_cl, a, b])
    return cv2.cvtColor(lab_cl, cv2.COLOR_LAB2RGB)


def rgb_only(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    print("=" * 65)
    print(" E4 — Equalização de Contraste em Imagens Subexpostas")
    print("=" * 65)

    # Substitui o dataset nos DOIS avaliadores para usar a versão escurecida
    original_ds = ev_module.DATASET_YAML
    original_ds_person = evp_module.DATASET_YAML
    ev_module.DATASET_YAML = DATASET_DARK
    evp_module.DATASET_YAML = DATASET_DARK

    variants = [
        (rgb_only, "E4-A: RGB apenas (ilum. ruim)"),
        (equalize_hist_hsv, "E4-B: equalizeHist (canal V-HSV)"),
        (equalize_hist_lab, "E4-C: equalizeHist (canal L-LAB)"),
        (clahe_hsv, "E4-D: CLAHE clip=2 tile=8 (HSV)"),
        (clahe_lab, "E4-E: CLAHE clip=2 tile=8 (LAB)"),
        (lambda f: clahe_lab(f, clip=4.0, tile=8), "E4-F: CLAHE clip=4 tile=8 (LAB)"),
    ]

    results, results_person = [], []
    for fn, label in variants:
        results.append(evaluate_pipeline(fn, label))
        results_person.append(evaluate_person_only(fn, label))

    # Restaura dataset original
    ev_module.DATASET_YAML = original_ds
    evp_module.DATASET_YAML = original_ds_person

    print("\n--- Resumo E4 (dataset escurecido, 3 classes) ---")
    b = results[0]['map50']
    for r in results[1:]:
        print(f"  {r['label']:38s} mAP={r['map50']:.4f}  delta={r['map50']-b:+.4f}")

    print("\n--- Resumo E4 (dataset escurecido, apenas Pessoa — sinal real) ---")
    bp = results_person[0]['map50']
    for r in results_person[1:]:
        print(f"  {r['label']:38s} mAP={r['map50']:.4f}  delta={r['map50']-bp:+.4f}")

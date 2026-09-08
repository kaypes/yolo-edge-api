"""
Experimento E1: impacto da conversão de espaço de cor BGR → RGB.
Compara três variantes de pré-processamento de cor.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, '.')
from preprocessing.utils.eval_person import evaluate_person_only
from preprocessing.utils.evaluate import evaluate_pipeline


# ── Variante A: sem conversão (passa BGR puro ao modelo) ─────────
def preproc_bgr_raw(frame: np.ndarray) -> np.ndarray:
    """Não faz nenhuma conversão — intencionalmente incorreto."""
    return frame    # BGR — o modelo espera RGB


# ── Variante B: conversão BGR→RGB (correto) ──────────────────────
def preproc_rgb_correct(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── Variante C: inversão manual de canais (equivalente ao B) ─────
def preproc_rgb_flip(frame: np.ndarray) -> np.ndarray:
    """Equivalente a cvtColor, mas usando indexação NumPy."""
    return frame[:, :, ::-1]   # inverte os canais: B,G,R → R,G,B


if __name__ == "__main__":
    print("=" * 65)
    print(" E1 — Impacto da Conversão de Espaço de Cor")
    print("=" * 65)

    variants = [
        (None, "E1-baseline (Ultralytics padrão)"),
        (preproc_bgr_raw, "E1-A: BGR sem conversão"),
        (preproc_rgb_correct, "E1-B: BGR→RGB (cvtColor)"),
        (preproc_rgb_flip, "E1-C: BGR→RGB (NumPy flip)"),
    ]

    results, results_person = [], []
    for fn, label in variants:
        results.append(evaluate_pipeline(fn, label))
        results_person.append(evaluate_person_only(fn, label))

    print("\n--- Resumo E1 (dataset completo, 3 classes) ---")
    baseline_map = results[0]['map50']
    for r in results[1:]:
        delta = r['map50'] - baseline_map
        print(f"  {r['label']:35s} mAP@0.5={r['map50']:.4f}  delta={delta:+.4f}")

    print("\n--- Resumo E1 (apenas Pessoa — sinal real) ---")
    baseline_person = results_person[0]['map50']
    for r in results_person[1:]:
        delta = r['map50'] - baseline_person
        print(f"  {r['label']:35s} mAP@0.5={r['map50']:.4f}  delta={delta:+.4f}")

    print("\nNOTA: o evaluate.py regrava as imagens em disco e o model.val() reaplica")
    print("sua própria conversão BGR→RGB ao ler o arquivo. Por isso o E1 sai com")
    print("delta ≈ 0 por construção — a verificação real do E1 é visual (e1_visualize.py).")

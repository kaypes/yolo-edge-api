import sys

sys.path.insert(0, '.')
from preprocessing.utils.eval_person import evaluate_person_only
from preprocessing.utils.evaluate import evaluate_pipeline

baseline = evaluate_pipeline(preprocess_fn=None, label="baseline (sem preproc)")
print(f"\nBaseline mAP@0.5 = {baseline['map50']:.4f}")
print("Anote este valor — ele é a referência de todos os experimentos.")

print("\n" + "=" * 65)
print(" Baseline complementar — apenas a classe Pessoa (COCO person)")
print("=" * 65)
print("O yolov8n.pt é COCO e não conhece Capacete/Colete, então o mAP acima é")
print("degenerado (~0.00x). A avaliação abaixo restringe a métrica à única classe")
print("com correspondência semântica real, produzindo deltas com significado.")
baseline_person = evaluate_person_only(preprocess_fn=None, label="baseline (sem preproc)")
print(f"\nBaseline Pessoa-only mAP@0.5 = {baseline_person['map50']:.4f}")

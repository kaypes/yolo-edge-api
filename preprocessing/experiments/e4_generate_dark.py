"""
Gera versões escuras das imagens de validação para testar equalização.
Salva em dataset/exports/epi-v1-dark/ mantendo os labels originais.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

SRC = Path("dataset/exports/epi-v1/valid")
DEST = Path("dataset/exports/epi-v1-dark/valid")
(DEST / "images").mkdir(parents=True, exist_ok=True)
(DEST / "labels").mkdir(parents=True, exist_ok=True)

# Copia labels sem alteração — a posição dos objetos não muda
for lbl in (SRC / "labels").glob("*.txt"):
    shutil.copy(lbl, DEST / "labels" / lbl.name)

# Gera imagens escurecidas (gamma > 1 escurece, < 1 clareia)
gamma = 2.2   # simula subexposição severa
table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)

for img_path in (SRC / "images").glob("*.jpg"):
    img = cv2.imread(str(img_path))
    dark = cv2.LUT(img, table)     # aplica a curva de gamma
    cv2.imwrite(str(DEST / "images" / img_path.name), dark)

n = len(list((DEST / 'images').glob('*.jpg')))
print(f"Geradas {n} imagens escurecidas (gamma={gamma})")

# Gera o data.yaml do dataset escurecido automaticamente -- o epi-v1-dark só
# tem a pasta valid/ (train/val/test apontam todos pra ela, o que é suficiente
# já que só rodamos avaliação, nunca treino, nesse dataset)
with open('dataset/exports/epi-v1/data.yaml') as f:
    base = yaml.safe_load(f)
dark_cfg = {
    'path': str(Path('dataset/exports/epi-v1-dark').resolve()),
    'train': 'valid/images',
    'val': 'valid/images',
    'test': 'valid/images',
    'names': base['names'],
}
with open('dataset/exports/epi-v1-dark/data.yaml', 'w') as f:
    yaml.safe_dump(dark_cfg, f)
print('data.yaml criado em dataset/exports/epi-v1-dark/')

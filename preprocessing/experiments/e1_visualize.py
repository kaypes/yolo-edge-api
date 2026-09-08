"""Gera comparativo visual BGR vs RGB para inspeção."""
from pathlib import Path

import cv2

img_path = sorted(Path("dataset/exports/epi-v1/valid/images").glob("*.jpg"))[0]
frame = cv2.imread(str(img_path))

# O cv2.imwrite() sempre interpreta o array recebido como BGR ao gravar.
#
#  - Gravar `frame` direto  -> arquivo correto (o array já é BGR).
#  - Gravar o array RGB     -> arquivo com R e B trocados, que é exatamente
#                              o que o modelo "enxerga" quando recebe BGR
#                              esperando RGB. É essa a imagem do erro.
#
# (O script do PDF gravava os dois com o mesmo nome de arquivo, sobrescrevendo
#  o comparativo — aqui cada variante tem o seu próprio nome.)
rgb_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

Path("preprocessing/outputs").mkdir(parents=True, exist_ok=True)
cv2.imwrite("preprocessing/outputs/e1_rgb_correto.jpg", frame)
cv2.imwrite("preprocessing/outputs/e1_bgr_incorreto.jpg", rgb_array)

# Comparativo lado a lado, com rótulo em cada metade
lado_a_lado = cv2.hconcat([frame, rgb_array])
cv2.putText(lado_a_lado, "CORRETO (RGB)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(lado_a_lado, "ERRADO (canais trocados)", (frame.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imwrite("preprocessing/outputs/e1_comparativo.jpg", lado_a_lado)

print(f"Imagem de referência: {img_path.name}")
print("Imagens salvas em preprocessing/outputs/:")
print("  e1_rgb_correto.jpg    — cores corretas")
print("  e1_bgr_incorreto.jpg  — canais R e B trocados (o que o modelo veria)")
print("  e1_comparativo.jpg    — as duas lado a lado")
print("\nDo seu computador, substitua <IP_DO_PI> e rode:")
print("IP_DO_PI=<seu-IP>")
print("scp kayque@$IP_DO_PI:~/yolo-edge-api/preprocessing/outputs/e1_*.jpg .")

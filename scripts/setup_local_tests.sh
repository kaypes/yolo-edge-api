#!/bin/bash
# scripts/setup_local_tests.sh
# Prepara o ambiente para rodar `pytest tests/ -v` fora do container,
# direto no Raspberry Pi. Idempotente: pode rodar quantas vezes quiser.
#
# Uso:  bash scripts/setup_local_tests.sh
#
# NAO rode com sudo. O script chama sudo apenas nas duas linhas que
# realmente precisam (criar /app e o link simbolico). Rodar tudo como
# root faria $HOME virar /root e instalaria os pacotes no lugar errado.

set -uo pipefail

# ── Recusa execucao como root ───────────────────────────────
if [ "$(id -u)" -eq 0 ]; then
    TARGET_USER="${SUDO_USER:-root}"
    if [ "$TARGET_USER" != "root" ]; then
        echo "[ERRO] Nao rode este script com sudo."
        echo "       Como root, \$HOME vira /root e o projeto/os pacotes"
        echo "       iriam parar no lugar errado."
        echo ""
        echo "       Rode assim, a partir da raiz do projeto:"
        echo "         cd ~/yolo-edge-api && bash scripts/setup_local_tests.sh"
        exit 1
    fi
fi

# ── Diretorio do projeto: onde este script mora, um nivel acima ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"

cd "$PROJECT_DIR" || { echo "[ERRO] Diretorio nao encontrado: $PROJECT_DIR"; exit 1; }
echo "Projeto: $PROJECT_DIR"
echo ""

echo "== [1/5] Imagem de referencia dos integration tests =="
mkdir -p tests/assets images output models
if [ ! -s tests/assets/zidane.jpg ]; then
    wget -q -O tests/assets/zidane.jpg https://ultralytics.com/images/zidane.jpg \
        && echo "  zidane.jpg baixado" \
        || echo "  [AVISO] download falhou - baixe manualmente"
else
    echo "  zidane.jpg ja existe"
fi
ls -lh tests/assets/zidane.jpg 2>/dev/null
# a mesma imagem serve como entrada do yolo-client
[ -s images/test.jpg ] || cp tests/assets/zidane.jpg images/test.jpg 2>/dev/null

echo ""
echo "== [2/5] Dependencias Python =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages -q
pip install -r app/requirements.txt --break-system-packages -q
pip install pytest ruff --break-system-packages -q
echo "  dependencias instaladas"

echo ""
echo "== [3/5] Link /app/models (mesmo mapeamento do docker-compose) =="
sudo mkdir -p /app
sudo ln -sfn "$PROJECT_DIR/models" /app/models
echo "  /app/models -> $PROJECT_DIR/models"

echo ""
echo "== [4/5] Pesos do modelo =="
if [ ! -s models/yolov8n.pt ]; then
    FOUND=$(find "$HOME" -name "yolov8n.pt" -size +1M -not -path "$PROJECT_DIR/*" 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        cp "$FOUND" models/yolov8n.pt
        echo "  copiado de: $FOUND"
    else
        echo "  baixando dos assets oficiais do Ultralytics..."
        wget -q -O models/yolov8n.pt \
            https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
            && echo "  yolov8n.pt baixado" \
            || echo "  [ERRO] download falhou - baixe manualmente para models/"
    fi
else
    echo "  models/yolov8n.pt ja existe"
fi
ls -lh /app/models/yolov8n.pt 2>/dev/null

echo ""
echo "== [5/5] Lint + testes =="
ruff check app/ --fix >/dev/null 2>&1
ruff check app/ && echo "  ruff limpo"
echo ""
pytest tests/ -v

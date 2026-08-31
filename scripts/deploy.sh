#!/bin/bash
# scripts/deploy.sh
# Executa no Raspberry Pi via SSH pelo pipeline de CI/CD.
# Faz pull da nova imagem, reinicia o servico e valida o health check.
# Em caso de falha, reverte para a imagem anterior automaticamente.

set -uo pipefail   # sem "-e": o rollback precisa ser alcancavel

DEPLOY_PATH="${DEPLOY_PATH:-$HOME/yolo-edge-api}"
HEALTH_URL="http://localhost:8000/health"
HEALTH_RETRIES=6
HEALTH_WAIT=10

echo "========================================"
echo " Deploy - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

cd "$DEPLOY_PATH" || exit 1

IMAGE_REF=$(docker compose config --images 2>/dev/null | head -1)
echo "[INFO] Imagem alvo: ${IMAGE_REF:-desconhecida}"

# ── Guarda o ID da imagem atual para possivel rollback ───────
# Usa .Image (ID sha256), nao .Config.Image: a tag :latest se move
# no pull e deixaria de apontar para a versao anterior.
PREVIOUS_ID=$(docker inspect yolo-api --format '{{.Image}}' 2>/dev/null || echo "")
if [ -n "$PREVIOUS_ID" ]; then
    echo "[INFO] Versao atual (rollback): ${PREVIOUS_ID:0:19}"
else
    echo "[INFO] Nenhum container anterior - primeiro deploy."
fi

# ── Baixa a nova imagem ──────────────────────────────────────
echo "[1/4] Baixando nova imagem..."
if ! docker compose pull; then
    echo "[ERRO] Falha ao baixar a imagem. Servico atual preservado."
    exit 1
fi

# ── Libera a porta antes de subir ────────────────────────────
echo "[2/4] Encerrando versao anterior..."
docker compose down --remove-orphans

PORT_HOLDER=$(docker ps --filter "publish=8000" --format '{{.Names}}' | head -1)
if [ -n "$PORT_HOLDER" ]; then
    echo "[AVISO] Porta 8000 ocupada pelo container '$PORT_HOLDER'"
    echo "        (provavelmente outro projeto Compose). Encerrando..."
    docker stop "$PORT_HOLDER" >/dev/null 2>&1
fi

# ── Sobe a nova versao ───────────────────────────────────────
echo "[3/4] Iniciando nova versao..."
UP_OK=true
docker compose up -d || UP_OK=false

# ── Aguarda o servico estabilizar ────────────────────────────
SUCCESS=false
if [ "$UP_OK" = true ]; then
    echo "[4/4] Aguardando health check ($((HEALTH_RETRIES * HEALTH_WAIT))s max)..."
    for i in $(seq 1 $HEALTH_RETRIES); do
        sleep $HEALTH_WAIT
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            SUCCESS=true
            break
        fi
        echo "  Tentativa $i/$HEALTH_RETRIES falhou, aguardando..."
    done
else
    echo "[ERRO] 'docker compose up' falhou. Pulando health check."
fi

# ── Avalia o resultado ───────────────────────────────────────
if [ "$SUCCESS" = true ]; then
    NEW_ID=$(docker inspect yolo-api --format '{{.Image}}' 2>/dev/null)
    echo ""
    echo "[OK] Deploy bem-sucedido: ${NEW_ID:0:19}"
    docker compose ps
    exit 0
fi

echo "[ERRO] Servico nao ficou saudavel."
echo "----- ultimas linhas do log do container -----"
docker compose logs --tail 30 yolo-api 2>&1 || true
echo "----------------------------------------------"

if [ -n "$PREVIOUS_ID" ] && [ -n "$IMAGE_REF" ]; then
    echo "[ROLLBACK] Revertendo para ${PREVIOUS_ID:0:19}"
    docker compose down --remove-orphans
    # Reaponta a tag para a imagem anterior; o Compose sobe essa versao.
    docker tag "$PREVIOUS_ID" "$IMAGE_REF"
    if docker compose up -d; then
        sleep 20
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            echo "[ROLLBACK] Concluido. Servico restaurado e saudavel."
        else
            echo "[ROLLBACK] Container subiu, mas o health check nao respondeu."
        fi
    else
        echo "[ROLLBACK] Falhou ao subir a versao anterior."
    fi
else
    echo "[AVISO] Sem imagem anterior para rollback."
fi

exit 1

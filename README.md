# yolo-edge-api

Pipeline completo de visão computacional em Edge AI: detecção de EPIs (capacete, colete, pessoa)
em tempo real numa **Raspberry Pi 5**, com API REST, streaming ao vivo, versionamento de dados/modelo
e CI/CD com deploy automático. Projeto desenvolvido ao longo do curso **Intensivo IA Cariri (PNAAT)**.

## Hardware

- Raspberry Pi 5 (aarch64, Cortex-A76), Raspberry Pi OS Trixie
- Câmera CSI (sensor ov5647)
- Treinamento de modelos feito fora do dispositivo (Google Colab, GPU T4) — a Pi só executa inferência

## Arquitetura

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│  yolo-api   │◄────►│  preprocessing│      │   yolo-stream    │
│  (FastAPI)  │      │  (letterbox,  │      │  (MJPEG + YOLO   │
│  /predict   │      │   CLAHE, ...) │      │  em tempo real)  │
└─────────────┘      └──────────────┘      └─────────────────┘
       ▲                                            ▲
       │                    modelo (DVC)             │
       └────────────────── models/yolo-epi.pt ───────┘
```

- **`app/`** — API REST em FastAPI (`/predict`, `/predict/image`, `/predict/batch`, `/health`,
  `/metrics`), servida via Uvicorn.
- **`stream/`** — pipeline de streaming em tempo real (evolução v1 → v2 threaded → v3 optimized)
  e servidor MJPEG (`mjpeg_server.py`) para visualização ao vivo no navegador.
- **`preprocessing/`** — módulo reutilizável de pré-processamento (letterbox, conversão de espaço
  de cor, CLAHE, filtros), com três configurações prontas (`CONFIG_DEFAULT`, `CONFIG_LOW_LIGHT`,
  `CONFIG_HIGH_QUALITY`) e a bateria de experimentos que embasou as decisões (`preprocessing/experiments/`).
- **`models/`** e **`dataset/`** — pesos e dataset versionados via **DVC**, com remoto SSH
  hospedado na própria Raspberry Pi (via Tailscale).
- **`.github/workflows/`** — pipeline de CI/CD com 4 jobs: lint & testes → build multi-arch
  (ARM64) e push para o GHCR → quality gate do modelo → deploy via SSH na Pi com rollback automático.
- **`entregaveis/`** — evidências de cada atividade do curso (Aulas 2–6).

## Rodando localmente

```bash
docker compose up -d --build
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://ultralytics.com/images/zidane.jpg", "confidence": 0.3}'
```

O stream ao vivo fica disponível em `http://<host>:5000`.

## Dataset — `epi-v1`

541 imagens anotadas (Roboflow), 3 classes: `Capacete`, `Colete`, `Pessoa`. Dividido em
474 treino / 34 validação / 33 teste. Versionado com DVC em `dataset/exports/epi-v1/`.

## Modelo — `yolo-epi.pt`

YOLOv8n treinado nas próprias classes do `epi-v1` (Colab, GPU T4, 50 épocas) — ver
`entregaveis/aula6/atividade-2/`. Substitui o `yolov8n.pt` genérico do COCO usado desde a Aula 2
e é o modelo validado pelo quality gate do CI/CD antes de qualquer deploy.

## Testes e qualidade

```bash
pytest tests/ -v      # suíte de testes (API + módulo de pré-processamento)
ruff check app/        # lint
```

## Índice de atividades do curso

| Aula | Tema | Evidências |
|---|---|---|
| 2 | Containerização (Docker Compose, API REST) | `entregaveis/aula2/` |
| 3 | Testes automatizados, DVC, CI/CD | `entregaveis/aula3/` |
| 4 | Streaming em tempo real, dataset epi-v1 | `entregaveis/aula4/` |
| 5 | Pré-processamento de imagens | `entregaveis/aula5/` |
| 6 | Arquiteturas de visão computacional, treinamento YOLOv8n/MobileNetV2 | `entregaveis/aula6/` |

## Autor

João Kayque Pereira de Souza — [github.com/kaypes](https://github.com/kaypes)

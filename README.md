# yolo-edge-api

Pipeline de MLOps e CI/CD para uma API de inferência YOLOv8 rodando em Raspberry Pi 5.

Um push na branch `main` dispara automaticamente: lint → testes → build da imagem
ARM64 → quality gate do modelo → deploy no dispositivo de borda com rollback automático.

## Stack

| Camada | Tecnologia |
| --- | --- |
| Inferência | Ultralytics YOLOv8 (`yolov8n.pt`) |
| API | FastAPI + Uvicorn |
| Container | Docker + Docker Compose (linux/arm64) |
| Versionamento de modelo | DVC (remote via SSH) |
| CI/CD | GitHub Actions + GHCR |
| Rede CI ↔ Pi | Tailscale |

## Estrutura do projeto

```
yolo-edge-api/
├── .github/
│   └── workflows/
│       └── edge-deploy.yml      ← pipeline CI/CD completo
├── .dvc/
│   └── config                   ← configuração do remote DVC
├── app/
│   ├── main.py                  ← FastAPI + logs estruturados
│   ├── model.py                 ← cache de modelos YOLO
│   ├── schemas.py               ← modelos Pydantic
│   ├── __init__.py
│   └── requirements.txt
├── client/
│   ├── client.py
│   └── requirements.txt
├── images/                      ← imagens de entrada do yolo-client
├── models/
│   └── yolov8n.pt.dvc           ← ponteiro DVC (binário no storage)
├── scripts/
│   ├── deploy.sh                ← deploy + health check + rollback
│   ├── setup_local_tests.sh     ← prepara o ambiente de testes na Pi
│   └── validate_model.py        ← quality gate mAP@0.5
├── tests/
│   ├── __init__.py
│   ├── assets/
│   │   └── zidane.jpg           ← imagem de referência para testes
│   └── test_api.py              ← 14 testes automatizados
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.client
└── ruff.toml
```

## Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| GET | `/health` | Smoke test — status, `model_loaded`, `model_name` |
| GET | `/metrics` | `total_requests`, `successful_requests`, `avg_inference_ms` |
| POST | `/predict` | Inferência em uma imagem (base64 ou URL) |
| POST | `/predict/image` | Inferência devolvendo a imagem anotada (JPEG) |
| POST | `/predict/batch` | Inferência em várias imagens |

## Rodando os testes localmente

Os testes resolvem os pesos em `/app/models` — o mesmo caminho que o
`docker-compose.yml` monta em produção. Fora do container, replique com
um link simbólico:

```bash
sudo mkdir -p /app
sudo ln -sfn ~/yolo-edge-api/models /app/models
```

Depois:

```bash
wget -O tests/assets/zidane.jpg https://ultralytics.com/images/zidane.jpg
pytest tests/ -v
```

Ou, para fazer tudo de uma vez:

```bash
bash scripts/setup_local_tests.sh
```

Saída esperada: `14 passed`.

## Cobertura dos testes

| Classe | Testes | Camada |
| --- | --- | --- |
| `TestSmoke` | 3 | O serviço sobe e responde |
| `TestDecodeImage` | 4 | Unit — decodificação base64 → `ndarray` |
| `TestPredictEndpoint` | 5 | Integration — inferência ponta a ponta |
| `TestBatchEndpoint` | 2 | Integration — lote de imagens |
| **Total** | **14** | |

## Secrets necessários no GitHub

`RPI_HOST`, `RPI_USER`, `RPI_SSH_KEY`, `RPI_DEPLOY_PATH`, `TAILSCALE_AUTHKEY`.

## Deploy e rollback

O `scripts/deploy.sh` guarda a imagem em execução, sobe a nova versão e
consulta `/health` por até 60s. Se o health check falhar, reverte
automaticamente para a imagem anterior.

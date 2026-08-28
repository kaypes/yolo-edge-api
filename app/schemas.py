"""
app/schemas.py
Contrato de dados da YOLO Inference API (Aula 2).

Unica diferenca em relacao ao original: protected_namespaces=().
Os campos model_name / model_used / model_loaded colidem com o
namespace protegido "model_" do Pydantic v2 e gerariam UserWarning
em todo `pytest -v`, poluindo a evidencia da entrega.
"""

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_base64: str | None = Field(
        None, description="Imagem PNG/JPG codificada em base64"
    )
    image_url: str | None = Field(
        None, description="URL publica acessivel a partir do container"
    )
    confidence: float = Field(
        0.25, ge=0.0, le=1.0, description="Limiar minimo de confianca (0-1)"
    )
    model_name: str = Field(
        "yolov8n.pt", description="Nome do arquivo de pesos dentro de /app/models/"
    )


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] em pixels


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    detections: list[Detection]
    inference_ms: float
    model_used: str
    image_width: int
    image_height: int


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    images_base64: list[str]
    confidence: float = 0.25
    model_name: str = "yolov8n.pt"


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_inference_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    avg_inference_ms: float

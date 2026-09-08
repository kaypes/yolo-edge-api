
from model import get_default_model_name
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str | None = Field(
        None,
        description="Imagem PNG/JPG codificada em base64"
    )
    image_url: str | None = Field(
        None,
        description="URL pública acessível a partir do container"
    )
    confidence: float = Field(0.25, ge=0.0, le=1.0,
        description="Limiar mínimo de confiança (0–1)")
    # default_factory (nao um valor fixo): le a env var MODEL_NAME no momento em
    # que o processo sobe, senao um cliente que nao informar model_name sempre
    # cairia no yolov8n.pt mesmo com outro modelo ativo em producao (era o que
    # acontecia antes -- /health reportava o modelo certo, /predict usava outro).
    model_name: str = Field(default_factory=get_default_model_name,
        description="Nome do arquivo de pesos dentro de /app/models/")


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]   # [x1, y1, x2, y2] em pixels


class PredictResponse(BaseModel):
    detections: list[Detection]
    inference_ms: float
    model_used: str
    image_width: int
    image_height: int


class BatchPredictRequest(BaseModel):
    images_base64: list[str]
    confidence: float = 0.25
    model_name: str = Field(default_factory=get_default_model_name)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    avg_inference_ms: float

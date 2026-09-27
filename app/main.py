from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import MODEL_DIR, DATABASE_PATH, MODEL_VERSION
from app.db.database import PredictionRepository
from app.services.classification_service import ClassificationService
from app.api.endpoints import router, set_classification_service



@asynccontextmanager
async def lifespan(app: FastAPI):

    repository = PredictionRepository(DATABASE_PATH)

    # ---------------------------------------- 
    # Classification service # # The service is responsible for: 
    # 1. Checking for existing model weights 
    # 2. Training on first run if weights # do not exist 
    # 3. Loading existing weights on # subsequent runs # ----------------------------------------

    classification_service = (
        ClassificationService(
            model_dir= MODEL_DIR,
            repository= repository,
            model_version = MODEL_VERSION
        )
    )

    # Make service available to API endpoints
    set_classification_service(classification_service)

    yield


app = FastAPI(
    title= "TerraClass",
    description= (
        "Offline satellite tile "
        "classification service"
    ),
    version= "1.0.0",
    lifespan= lifespan,
)


app.include_router(router)



@app.get("/health")
def health():
    return {"status": "ok"}
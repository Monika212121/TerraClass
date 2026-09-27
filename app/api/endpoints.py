from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    Query,
)

from app.schemas.prediction import load_image


router = APIRouter()

classification_service = None


def set_classification_service(service):
    global classification_service

    classification_service = service



@router.get("/predictions")
def get_predictions(limit: int = Query(default=100, ge=1, le=500)):
    return classification_service.list_predictions(limit=limit)



@router.post("/predict")
async def predict_tile(
    tile: UploadFile = File(...)
):

    if classification_service is None:
        raise HTTPException(status_code= 503, detail= "Classification service is not ready")

    if not tile.filename:
        raise HTTPException(status_code= 400, detail= "Tile filename is required")

    if (tile.content_type and not tile.content_type.startswith("image/")):
        raise HTTPException(status_code= 400, detail= "Uploaded file must be an image")


    data = await tile.read()

    if not data:
        raise HTTPException(status_code= 400, detail= "Uploaded tile is empty")


    try:
        image = load_image(data)

    except ValueError as exc:
        raise HTTPException(status_code= 400, detail= str(exc)) from exc


    result = classification_service.classify(image, data)

    record = classification_service.repository.save(
        tile_name= tile.filename,
        tile_sha256= result["tile_sha256"],
        predicted_label= result["predicted_label"],
        confidence= result["confidence"],
        model_version= classification_service.model_version,
        inference_time_ms= result[
            "inference_time_ms"
        ],
    )

    return {**record, "class_probs": result["class_probs"]}
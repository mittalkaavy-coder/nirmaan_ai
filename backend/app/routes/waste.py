from backend.app.services.waste_detector import detect_waste
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO

from backend.app.models import (
    WasteAnalysisRequest,
    WasteAnalysisResponse,
    ImageWasteAnalysisResponse,
)

from backend.app.services.waste_classifier import predict_waste
from backend.app.services.waste_intelligence import analyze_waste_type


router = APIRouter(
    prefix="/api/v1/waste",
    tags=["Waste Intelligence"]
)


# --------------------------------------------------
# Text-based waste analysis
# --------------------------------------------------

@router.post(
    "/analyze",
    response_model=WasteAnalysisResponse
)
def analyze_waste(data: WasteAnalysisRequest):

    result = analyze_waste_type(
        data.waste_type
    )

    return WasteAnalysisResponse(
        waste_type=data.waste_type,
        category=result["category"],
        recyclable=result["recyclable"],
        disposal_method=result["disposal_method"],
        environmental_impact=result["environmental_impact"],
        confidence=result["confidence"],
    )


# --------------------------------------------------
# Image-based waste analysis
# --------------------------------------------------

@router.post(
    "/analyze-image",
    response_model=ImageWasteAnalysisResponse
)
async def analyze_waste_image(
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        )

        prediction = predict_waste(image)

        intelligence = analyze_waste_type(
            prediction["class_name"],
            prediction["confidence"]
        )

        return {
            "filename": file.filename,
            "predicted_class": prediction["class_name"],
            "confidence": prediction["confidence"],
            "category": intelligence["category"],
            "recyclable": intelligence["recyclable"],
            "disposal_method": intelligence["disposal_method"],
            "environmental_impact": intelligence["environmental_impact"],
            "probabilities": prediction["probabilities"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(e)}"
        )


@router.post("/detect-image")
async def detect_waste_image(file: UploadFile = File(...)):
    """Detect individual waste objects in an uploaded image."""
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        try:
            image = Image.open(BytesIO(contents))
            image.verify()
            image = Image.open(BytesIO(contents))
            image.load()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported image file",
            )

        return detect_waste(image)

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Waste detection model checkpoint is missing",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to detect waste objects",
        )
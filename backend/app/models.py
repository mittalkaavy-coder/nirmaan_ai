from pydantic import BaseModel, Field


class WasteAnalysisRequest(BaseModel):
    waste_type: str = Field(
        ...,
        min_length=1,
        description="Type or description of the waste"
    )


class WasteAnalysisResponse(BaseModel):
    waste_type: str
    category: str
    recyclable: bool
    disposal_method: str
    environmental_impact: str
    confidence: float


class ImageWasteAnalysisResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    category: str
    recyclable: bool
    disposal_method: str
    environmental_impact: str
    probabilities: dict[str, float]
def analyze_waste_type(waste_type: str, confidence: float = 0.85) -> dict:
    """
    Analyze a waste type and return disposal intelligence.
    """

    waste = waste_type.lower()

    if any(word in waste for word in ["plastic", "bottle", "container"]):
        category = "Plastic"
        recyclable = True
        disposal_method = "Send to a plastic recycling facility"
        environmental_impact = (
            "Plastic waste can persist for hundreds of years"
        )

    elif any(word in waste for word in ["paper", "cardboard", "newspaper"]):
        category = "Paper"
        recyclable = True
        disposal_method = "Send to a paper recycling facility"
        environmental_impact = (
            "Paper recycling reduces the need for virgin materials"
        )

    elif any(word in waste for word in [
        "food",
        "vegetable",
        "fruit",
        "organic"
    ]):
        category = "Organic"
        recyclable = False
        disposal_method = (
            "Send to composting or organic waste processing"
        )
        environmental_impact = (
            "Organic waste can generate methane when dumped in landfills"
        )

    elif any(word in waste for word in [
        "battery",
        "chemical",
        "electronic",
        "e-waste"
    ]):
        category = "Hazardous / E-Waste"
        recyclable = True
        disposal_method = (
            "Send to an authorized e-waste collection facility"
        )
        environmental_impact = (
            "Improper disposal can release toxic substances"
        )

    else:
        category = "General Waste"
        recyclable = False
        disposal_method = (
            "Dispose through the appropriate municipal waste stream"
        )
        environmental_impact = (
            "Improper waste disposal can contribute to pollution"
        )

    return {
        "category": category,
        "recyclable": recyclable,
        "disposal_method": disposal_method,
        "environmental_impact": environmental_impact,
        "confidence": confidence,
    }
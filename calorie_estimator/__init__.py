"""Calorie Estimator — Agent Tool for food photo nutrition analysis.

Usage:
    from calorie_estimator import CalorieEstimator

    estimator = CalorieEstimator()
    result = await estimator.estimate(image_bytes, "grilled chicken with rice")

    print(result.format_summary())
    print(result.total.calories)
"""

from .estimator import CalorieEstimator, EstimatorTransientError
from .models import (
    FoodCategory,
    FoodItem,
    HiddenCalorieEstimate,
    ItemEstimate,
    MealEstimate,
    NutrientProfile,
    NutrientRange,
    OFFContribution,
    OpenFoodFactsProduct,
    USDACandidate,
    VisualAnalysis,
)
from .openfoodfacts import OpenFoodFactsClient

__all__ = [
    "CalorieEstimator",
    "EstimatorTransientError",
    "FoodCategory",
    "FoodItem",
    "HiddenCalorieEstimate",
    "ItemEstimate",
    "MealEstimate",
    "NutrientProfile",
    "NutrientRange",
    "OFFContribution",
    "OpenFoodFactsClient",
    "OpenFoodFactsProduct",
    "USDACandidate",
    "VisualAnalysis",
]

__version__ = "0.1.0"

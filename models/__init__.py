# models/__init__.py
from .advanced_models import (
    MultiScaleCNN,
    SelfAttention,
    SensorGNN,
    AdvancedPredictionModel,
    TORCH_GEOMETRIC_AVAILABLE,
)

__all__ = [
    'MultiScaleCNN',
    'SelfAttention',
    'SensorGNN',
    'AdvancedPredictionModel',
    'TORCH_GEOMETRIC_AVAILABLE',
]
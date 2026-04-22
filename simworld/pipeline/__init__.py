"""Pipeline package — SIMWORLD data pipeline.

Provides classification, logging, routing, dataset building,
training management, and inference routing for the simulation platform.
"""

from pipeline.model_registry import MODEL_REGISTRY, get_registry, get_all_categories
from pipeline.classifier import SituationClassifier
from pipeline.log_collector import LogCollector
from pipeline.log_router import LogRouter
from pipeline.dataset_builder import DatasetBuilder
from pipeline.training_manager import TrainingManager
from pipeline.inference_router import InferenceRouter

__all__ = [
    "MODEL_REGISTRY",
    "get_registry",
    "get_all_categories",
    "SituationClassifier",
    "LogCollector",
    "LogRouter",
    "DatasetBuilder",
    "TrainingManager",
    "InferenceRouter",
]

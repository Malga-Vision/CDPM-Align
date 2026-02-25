"""
Landmark detection downstream task.
"""

from .landmarks_train_test import (
    train_and_validate as train_landmarks,
    evaluate as evaluate_landmarks
)

__all__ = [
    'train_landmarks',
    'evaluate_landmarks',
]

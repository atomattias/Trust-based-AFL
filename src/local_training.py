"""
Local model training module for federated clients.

This module handles training machine learning models at each honeypot client.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, Tuple, Any, Optional
import pandas as pd


def train_local_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    random_state: int = 42,
    **model_kwargs
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a local model on client data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('random_forest' or 'logistic_regression')
        random_state: Random seed
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Tuple of (trained_model, training_metrics_dict)
    """
    if model_type == 'random_forest':
        n_estimators = model_kwargs.get('n_estimators', 100)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'logistic_regression':
        model = SGDClassifier(
            loss='log_loss',
            max_iter=model_kwargs.get('max_iter', 1000),
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest' or 'logistic_regression'")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    training_metrics = {
        'train_accuracy': train_acc,
        'train_f1': train_f1
    }
    
    return model, training_metrics


def evaluate_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate a model on validation data.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
    }
    
    # Calculate false positive rate
    cm = confusion_matrix(y_val, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        metrics['false_positive_rate'] = 0.0
    
    return metrics


def get_model_parameters(model: Any, model_type: str) -> Dict[str, np.ndarray]:
    """
    Extract model parameters for federated aggregation.
    
    Args:
        model: Trained model
        model_type: Type of model ('random_forest' or 'logistic_regression')
        
    Returns:
        Dictionary with model parameters
    """
    params = {}
    
    if model_type == 'random_forest':
        # For Random Forest, we aggregate feature importances
        params['feature_importances'] = model.feature_importances_
        params['n_features'] = len(model.feature_importances_)
        params['classes'] = model.classes_
    elif model_type == 'logistic_regression':
        # For Logistic Regression, we aggregate coefficients and intercept
        params['coef'] = model.coef_
        params['intercept'] = model.intercept_
        params['classes'] = model.classes_
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return params

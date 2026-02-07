"""
Evaluation module for computing metrics and comparing approaches.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, List, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }
    
    # Calculate false positive rate
    # Use labels parameter to ensure 2x2 matrix even if only one class present
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    except ValueError:
        # Fallback if labels don't match
        cm = confusion_matrix(y_true, y_pred)
        # Ensure 2x2 shape
        if cm.shape == (1, 1):
            # Only one class present
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            cm_full = np.zeros((2, 2), dtype=int)
            if 1 in unique_true or 1 in unique_pred:
                # Attack class present
                cm_full[1, 1] = cm[0, 0] if len(unique_true) == 1 and len(unique_pred) == 1 else cm[0, 0]
            else:
                # Benign class present
                cm_full[0, 0] = cm[0, 0]
            cm = cm_full
        elif cm.shape != (2, 2):
            # Reshape to 2x2
            cm_full = np.zeros((2, 2), dtype=int)
            min_dim = min(cm.shape[0], 2)
            min_dim2 = min(cm.shape[1], 2)
            cm_full[:min_dim, :min_dim2] = cm[:min_dim, :min_dim2]
            cm = cm_full
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['false_positive_rate'] = 0.0
        metrics['true_positive_rate'] = 0.0
        metrics['true_negative_rate'] = 0.0
    
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def evaluate_model_on_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = 'standard'
) -> Dict[str, Any]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model or aggregator
        X_test: Test features
        y_test: Test labels
        model_type: Type of model ('standard', 'ensemble', 'aggregator')
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    if model_type == 'ensemble':
        # Ensemble aggregator
        y_pred = model.predict(X_test.values)
    elif hasattr(model, 'predict'):
        # Standard sklearn model or aggregator
        y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Compute metrics
    metrics = compute_metrics(y_test.values, y_pred)
    
    # Add classification report
    metrics['classification_report'] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    
    return metrics


def compare_approaches(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Compare results from different approaches.
    
    Args:
        results: Dictionary mapping approach names to their metrics
        metric: Metric to compare
        
    Returns:
        DataFrame with comparison
    """
    comparison = {}
    for approach, metrics in results.items():
        comparison[approach] = metrics.get(metric, 0.0)
    
    df = pd.DataFrame([comparison]).T
    df.columns = [metric]
    df = df.sort_values(by=metric, ascending=False)
    
    return df


def extract_trust_evolution_metrics(
    client_info: List[Dict[str, Any]],
    trust_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extract trust evolution metrics from client info and trust manager.
    
    Args:
        client_info: List of client information dictionaries
        trust_manager: Optional TrustManager instance with trust histories
        
    Returns:
        Dictionary with trust evolution metrics
    """
    evolution_metrics = {
        'final_trust_statistics': {},
        'trust_evolution': {},
        'client_trends': {},
        'trust_changes': {}
    }
    
    # Extract final trust scores
    trust_scores = [c.get('trust_score', 0) for c in client_info if c.get('trust_score') is not None]
    if trust_scores:
        evolution_metrics['final_trust_statistics'] = {
            'mean': float(np.mean(trust_scores)),
            'std': float(np.std(trust_scores)),
            'min': float(np.min(trust_scores)),
            'max': float(np.max(trust_scores)),
            'median': float(np.median(trust_scores)),
            'count': len(trust_scores)
        }
    
    # Extract trust evolution from performance history if available
    for client in client_info:
        client_id = client.get('client_id', 'unknown')
        perf_history = client.get('performance_history', [])
        
        if perf_history:
            # Extract trust scores over rounds
            trust_over_rounds = [p.get('trust_score', 0) for p in perf_history if 'trust_score' in p]
            if trust_over_rounds:
                evolution_metrics['trust_evolution'][client_id] = {
                    'rounds': list(range(1, len(trust_over_rounds) + 1)),
                    'trust_scores': trust_over_rounds,
                    'initial_trust': trust_over_rounds[0] if trust_over_rounds else 0,
                    'final_trust': trust_over_rounds[-1] if trust_over_rounds else 0,
                    'total_change': trust_over_rounds[-1] - trust_over_rounds[0] if len(trust_over_rounds) > 1 else 0
                }
                
                # Calculate trend
                if len(trust_over_rounds) >= 2:
                    trend = client.get('trend', 'stable')
                    evolution_metrics['client_trends'][client_id] = trend
                    
                    # Calculate total change
                    total_change = trust_over_rounds[-1] - trust_over_rounds[0]
                    evolution_metrics['trust_changes'][client_id] = {
                        'total_change': float(total_change),
                        'percent_change': float((total_change / trust_over_rounds[0] * 100) if trust_over_rounds[0] > 0 else 0),
                        'trend': trend
                    }
    
    # Extract from TrustManager if available
    if trust_manager and hasattr(trust_manager, 'trust_histories'):
        for client_id, history in trust_manager.trust_histories.items():
            if client_id not in evolution_metrics['trust_evolution']:
                trust_scores = history.trust_scores
                if len(trust_scores) > 1:
                    evolution_metrics['trust_evolution'][client_id] = {
                        'rounds': history.round_numbers[1:],  # Skip initial round 0
                        'trust_scores': trust_scores[1:],
                        'initial_trust': trust_scores[0],
                        'final_trust': trust_scores[-1],
                        'total_change': trust_scores[-1] - trust_scores[0]
                    }
                    
                    # Calculate statistics
                    if len(trust_scores) > 1:
                        trust_changes = [trust_scores[i] - trust_scores[i-1] for i in range(1, len(trust_scores))]
                        evolution_metrics['trust_changes'][client_id] = {
                            'total_change': float(trust_scores[-1] - trust_scores[0]),
                            'percent_change': float((trust_scores[-1] - trust_scores[0]) / trust_scores[0] * 100) if trust_scores[0] > 0 else 0,
                            'avg_change_per_round': float(np.mean(trust_changes)),
                            'max_increase': float(np.max(trust_changes)) if trust_changes else 0,
                            'max_decrease': float(np.min(trust_changes)) if trust_changes else 0,
                            'trend': history.get_trend()
                        }
    
    return evolution_metrics


def generate_results_summary(
    centralized_results: Dict[str, Any],
    federated_results: Dict[str, Any],
    trust_aware_results: Dict[str, Any],
    client_info: Optional[List[Dict[str, Any]]] = None,
    trust_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive results summary.
    
    Args:
        centralized_results: Results from centralized learning
        federated_results: Results from standard federated learning
        trust_aware_results: Results from trust-aware federated learning
        client_info: Optional list of client information dictionaries
        trust_manager: Optional TrustManager instance for trust evolution metrics
        
    Returns:
        Dictionary with summary
    """
    summary = {
        'approaches': {
            'centralized': centralized_results,
            'federated_equal_weight': federated_results,
            'trust_aware': trust_aware_results
        },
        'comparison': {
            'accuracy': {
                'centralized': centralized_results.get('accuracy', 0),
                'federated_equal_weight': federated_results.get('accuracy', 0),
                'trust_aware': trust_aware_results.get('accuracy', 0)
            },
            'f1_score': {
                'centralized': centralized_results.get('f1_score', 0),
                'federated_equal_weight': federated_results.get('f1_score', 0),
                'trust_aware': trust_aware_results.get('f1_score', 0)
            },
            'false_positive_rate': {
                'centralized': centralized_results.get('false_positive_rate', 0),
                'federated_equal_weight': federated_results.get('false_positive_rate', 0),
                'trust_aware': trust_aware_results.get('false_positive_rate', 0)
            }
        }
    }
    
    if client_info:
        summary['clients'] = client_info
        # Compute trust statistics
        trust_scores = [c.get('trust_score', 0) for c in client_info if c.get('trust_score') is not None]
        if trust_scores:
            summary['trust_statistics'] = {
                'mean': float(np.mean(trust_scores)),
                'std': float(np.std(trust_scores)),
                'min': float(np.min(trust_scores)),
                'max': float(np.max(trust_scores)),
                'median': float(np.median(trust_scores))
            }
        
        # Extract trust evolution metrics
        trust_evolution = extract_trust_evolution_metrics(client_info, trust_manager)
        if trust_evolution['trust_evolution'] or trust_evolution['final_trust_statistics']:
            summary['trust_evolution'] = trust_evolution
    
    # Add trust statistics from results if available (multi-round mode)
    if 'trust_statistics' in trust_aware_results:
        summary['trust_statistics'] = trust_aware_results['trust_statistics']
    
    return summary

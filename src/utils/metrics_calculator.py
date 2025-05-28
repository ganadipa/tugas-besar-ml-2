import numpy as np
from typing import List, Tuple

class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                           num_classes: int, average: str = 'macro') -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1-score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            num_classes: Number of classes
            average: 'macro', 'micro', or 'weighted'
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Calculate per-class metrics
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_id in range(num_classes):
            tp = np.sum((y_true == class_id) & (y_pred == class_id))
            fp = np.sum((y_true != class_id) & (y_pred == class_id))
            fn = np.sum((y_true == class_id) & (y_pred != class_id))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        if average == 'macro':
            return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
        elif average == 'micro':
            # Calculate micro-averaged metrics
            total_tp = sum(np.sum((y_true == i) & (y_pred == i)) for i in range(num_classes))
            total_fp = sum(np.sum((y_true != i) & (y_pred == i)) for i in range(num_classes))
            total_fn = sum(np.sum((y_true == i) & (y_pred != i)) for i in range(num_classes))
            
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
            
            return micro_precision, micro_recall, micro_f1
        else:
            # Weighted average
            class_counts = [np.sum(y_true == i) for i in range(num_classes)]
            total_samples = len(y_true)
            
            weighted_precision = sum(p * c for p, c in zip(precisions, class_counts)) / total_samples
            weighted_recall = sum(r * c for r, c in zip(recalls, class_counts)) / total_samples
            weighted_f1 = sum(f * c for f, c in zip(f1_scores, class_counts)) / total_samples
            
            return weighted_precision, weighted_recall, weighted_f1
    
    @staticmethod
    def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        """Calculate macro F1-score"""
        _, _, f1 = MetricsCalculator.precision_recall_f1(y_true, y_pred, num_classes, 'macro')
        return f1
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str] = None) -> str:
        """Generate classification report"""
        num_classes = max(max(y_true), max(y_pred)) + 1
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        report_lines = ["Classification Report"]
        report_lines.append("=" * 50)
        report_lines.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report_lines.append("-" * 50)
        
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        
        for i in range(num_classes):
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            support = np.sum(y_true == i)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            supports.append(support)
            
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            report_lines.append(f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
        
        report_lines.append("-" * 50)
        
        # Macro average
        macro_prec = np.mean(precisions)
        macro_rec = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)
        total_support = sum(supports)
        
        report_lines.append(f"{'Macro avg':<15} {macro_prec:<10.3f} {macro_rec:<10.3f} {macro_f1:<10.3f} {total_support:<10}")
        
        # Weighted average
        weighted_prec = sum(p * s for p, s in zip(precisions, supports)) / total_support
        weighted_rec = sum(r * s for r, s in zip(recalls, supports)) / total_support
        weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support
        
        report_lines.append(f"{'Weighted avg':<15} {weighted_prec:<10.3f} {weighted_rec:<10.3f} {weighted_f1:<10.3f} {total_support:<10}")
        
        # Overall accuracy
        accuracy = np.mean(y_true == y_pred)
        report_lines.append("-" * 50)
        report_lines.append(f"Accuracy: {accuracy:.3f}")
        
        return "\n".join(report_lines)
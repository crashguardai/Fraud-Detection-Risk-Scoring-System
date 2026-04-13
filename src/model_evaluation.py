"""
Model Evaluation and Analysis Tools

This module provides comprehensive evaluation tools for fraud detection models,
including detailed metrics, visualizations, and analysis tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score)
from sklearn.calibration import calibration_curve
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation class for fraud detection
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.models = {}
        
    def load_models_and_data(self, models_dir='../models', data_dir='../data'):
        """
        Load trained models and test data
        
        Args:
            models_dir: Directory containing model files
            data_dir: Directory containing data files
        """
        print("Loading models and data...")
        
        # Load models
        try:
            self.models['Random Forest'] = joblib.load(f'{models_dir}/random_forest.pkl')
            self.models['Logistic Regression'] = joblib.load(f'{models_dir}/logistic_regression.pkl')
            
            # Try to load tuned model if exists
            try:
                self.models['Tuned Random Forest'] = joblib.load(f'{models_dir}/tuned_random_forest.pkl')
            except FileNotFoundError:
                print("Tuned Random Forest model not found, using original models")
                
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            return None, None, None, None
        
        # Load test data
        try:
            test_data = pd.read_csv(f'{data_dir}/test_processed.csv')
            X_test = test_data.drop('is_fraud', axis=1)
            y_test = test_data['is_fraud']
            
            print(f"Loaded {len(self.models)} models and test data with {X_test.shape[0]} samples")
            return X_test, y_test
            
        except FileNotFoundError as e:
            print(f"Data file not found: {e}")
            return None, None
    
    def comprehensive_evaluation(self, model, X_test, y_test, model_name):
        """
        Perform comprehensive evaluation of a model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Cost-sensitive metrics (assuming different costs)
        cost_fp = 10  # Cost of false positive (customer inconvenience)
        cost_fn = 1000  # Cost of false negative (fraud loss)
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        # Precision-Recall curve data
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Calibration data
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'avg_precision': avg_precision,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_cost': total_cost,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precisions': precisions.tolist(),
                'recalls': recalls.tolist(),
                'thresholds': pr_thresholds.tolist()
            },
            'calibration': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
        
        self.evaluation_results[model_name] = results
        return results
    
    def print_detailed_results(self, model_name):
        """
        Print detailed evaluation results for a model
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.evaluation_results:
            print(f"No results found for {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        
        print(f"\n{'='*60}")
        print(f"DETAILED EVALUATION RESULTS: {model_name}")
        print(f"{'='*60}")
        
        print(f"\nBASIC METRICS:")
        print(f"  Accuracy:           {results['accuracy']:.4f}")
        print(f"  Precision:          {results['precision']:.4f}")
        print(f"  Recall:             {results['recall']:.4f}")
        print(f"  F1-Score:           {results['f1_score']:.4f}")
        print(f"  AUC:                {results['auc']:.4f}")
        print(f"  Avg Precision:      {results['avg_precision']:.4f}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Positives:     {results['true_positives']}")
        print(f"  True Negatives:     {results['true_negatives']}")
        print(f"  False Positives:    {results['false_positives']}")
        print(f"  False Negatives:    {results['false_negatives']}")
        
        print(f"\nDERIVED METRICS:")
        print(f"  Specificity:        {results['specificity']:.4f}")
        print(f"  Sensitivity:        {results['sensitivity']:.4f}")
        print(f"  False Positive Rate: {results['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {results['false_negative_rate']:.4f}")
        
        print(f"\nCOST ANALYSIS:")
        print(f"  Total Cost:         ${results['total_cost']:,.2f}")
        print(f"  Cost per Transaction: ${results['total_cost']/sum(results['true_positives'] + results['true_negatives'] + results['false_positives'] + results['false_negatives']):.2f}")
        
        print(f"\nCLASSIFICATION REPORT:")
        cm = np.array(results['confusion_matrix'])
        print(f"  Legitimate Precision: {cm[0,0]/(cm[0,0]+cm[1,0]):.4f}")
        print(f"  Legitimate Recall:    {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
        print(f"  Fraud Precision:       {cm[1,1]/(cm[1,1]+cm[0,1]):.4f}")
        print(f"  Fraud Recall:          {cm[1,1]/(cm[1,1]+cm[1,0]):.4f}")
    
    def plot_comprehensive_comparison(self, figsize=(20, 15)):
        """
        Create comprehensive comparison plots for all models
        
        Args:
            figsize: Figure size
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluation first.")
            return
        
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle('Comprehensive Model Evaluation Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(self.evaluation_results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        # 1. Basic Metrics Comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        ax = axes[0, 0]
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            values = [self.evaluation_results[model_name][metric] for metric in metrics]
            ax.bar(x + i * width, values, width, label=model_name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Basic Metrics Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        ax = axes[0, 1]
        for model_name, color in zip(model_names, colors):
            results = self.evaluation_results[model_name]
            ax.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'], 
                   label=f'{model_name} (AUC = {results["auc"]:.3f})', color=color, linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        ax = axes[0, 2]
        for model_name, color in zip(model_names, colors):
            results = self.evaluation_results[model_name]
            ax.plot(results['pr_curve']['recalls'], results['pr_curve']['precisions'], 
                   label=f'{model_name} (AP = {results["avg_precision"]:.3f})', color=color, linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. Confusion Matrices
        for i, model_name in enumerate(model_names):
            if i < 2:  # Only show first 2 models
                ax = axes[0, 3] if i == 0 else axes[1, 0]
                cm = np.array(self.evaluation_results[model_name]['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Legitimate', 'Fraud'],
                           yticklabels=['Legitimate', 'Fraud'])
                ax.set_title(f'Confusion Matrix - {model_name}')
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
        
        # 5. Cost Analysis
        ax = axes[1, 1]
        costs = [self.evaluation_results[name]['total_cost'] for name in model_names]
        bars = ax.bar(model_names, costs, color=colors, alpha=0.8)
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Cost Analysis (FP=$10, FN=$1000)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.01,
                   f'${cost:,.0f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)
        
        # 6. Error Rates
        ax = axes[1, 2]
        fpr_rates = [self.evaluation_results[name]['false_positive_rate'] for name in model_names]
        fnr_rates = [self.evaluation_results[name]['false_negative_rate'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, fpr_rates, width, label='False Positive Rate', alpha=0.8)
        ax.bar(x + width/2, fnr_rates, width, label='False Negative Rate', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Rate')
        ax.set_title('Error Rates Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Calibration Curves
        ax = axes[1, 3]
        for model_name, color in zip(model_names, colors):
            results = self.evaluation_results[model_name]
            ax.plot(results['calibration']['mean_predicted_value'], 
                   results['calibration']['fraction_of_positives'], 
                   's-', label=model_name, color=color, linewidth=2, markersize=6)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 8. Threshold Analysis
        ax = axes[2, 0]
        for model_name, color in zip(model_names, colors):
            results = self.evaluation_results[model_name]
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            thresholds = results['roc_curve']['thresholds']
            
            # Calculate Youden's J statistic for optimal threshold
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]
            
            ax.axvline(optimal_threshold, color=color, linestyle='--', alpha=0.7,
                      label=f'{model_name}: {optimal_threshold:.3f}')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Value')
        ax.set_title('Optimal Thresholds (Youden\'s J)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 9. Performance Summary Table
        ax = axes[2, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for model_name in model_names:
            results = self.evaluation_results[model_name]
            summary_data.append([
                model_name,
                f"{results['accuracy']:.3f}",
                f"{results['precision']:.3f}",
                f"{results['recall']:.3f}",
                f"{results['f1_score']:.3f}",
                f"{results['auc']:.3f}",
                f"${results['total_cost']:,.0f}"
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Model', 'Acc', 'Prec', 'Rec', 'F1', 'AUC', 'Cost'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax.set_title('Performance Summary', pad=20)
        
        # 10. Recommendation
        ax = axes[2, 2]
        ax.axis('off')
        
        # Find best model based on different criteria
        best_f1 = max(model_names, key=lambda x: self.evaluation_results[x]['f1_score'])
        best_auc = max(model_names, key=lambda x: self.evaluation_results[x]['auc'])
        best_cost = min(model_names, key=lambda x: self.evaluation_results[x]['total_cost'])
        
        recommendation_text = f"""
MODEL RECOMMENDATIONS:

Best F1-Score: {best_f1}
Best AUC: {best_auc}
Lowest Cost: {best_cost}

Overall Recommendation: {best_f1}

Key Insights:
- F1-Score balances precision and recall
- AUC measures overall discriminative power
- Cost considers business impact
- Consider threshold tuning for specific needs
        """
        
        ax.text(0.1, 0.9, recommendation_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Recommendations', pad=20)
        
        # Remove empty subplots
        axes[2, 3].remove()
        
        plt.tight_layout()
        plt.show()
    
    def threshold_analysis(self, model_name, X_test, y_test, figsize=(12, 8)):
        """
        Analyze different threshold values for a model
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            figsize: Figure size
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Cost calculation
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            cost = (fp * 10) + (fn * 1000)
            
            metrics.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cost': cost,
                'false_positives': fp,
                'false_negatives': fn
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Plot threshold analysis
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Threshold Analysis - {model_name}', fontsize=14, fontweight='bold')
        
        # Metrics vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(metrics_df['threshold'], metrics_df['accuracy'], 'b-', label='Accuracy')
        ax1.plot(metrics_df['threshold'], metrics_df['precision'], 'g-', label='Precision')
        ax1.plot(metrics_df['threshold'], metrics_df['recall'], 'r-', label='Recall')
        ax1.plot(metrics_df['threshold'], metrics_df['f1_score'], 'm-', label='F1-Score')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cost vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(metrics_df['threshold'], metrics_df['cost'], 'r-', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Total Cost ($)')
        ax2.set_title('Cost vs Threshold')
        ax2.grid(True, alpha=0.3)
        
        # Error Counts vs Threshold
        ax3 = axes[1, 0]
        ax3.plot(metrics_df['threshold'], metrics_df['false_positives'], 'b-', label='False Positives')
        ax3.plot(metrics_df['threshold'], metrics_df['false_negatives'], 'r-', label='False Negatives')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Count')
        ax3.set_title('Error Counts vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Precision-Recall Trade-off
        ax4 = axes[1, 1]
        ax4.plot(metrics_df['recall'], metrics_df['precision'], 'g-', linewidth=2)
        ax4.scatter(metrics_df['recall'], metrics_df['precision'], c=metrics_df['threshold'], 
                   cmap='viridis', s=50, alpha=0.7)
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Trade-off')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for thresholds
        cbar = plt.colorbar(ax4.scatter(metrics_df['recall'], metrics_df['precision'], 
                                       c=metrics_df['threshold'], cmap='viridis', s=50, alpha=0.7),
                           ax=ax4)
        cbar.set_label('Threshold')
        
        plt.tight_layout()
        plt.show()
        
        # Print optimal thresholds
        print(f"\nOptimal Thresholds for {model_name}:")
        print(f"Best Accuracy: {metrics_df.loc[metrics_df['accuracy'].idxmax(), 'threshold']:.3f}")
        print(f"Best F1-Score: {metrics_df.loc[metrics_df['f1_score'].idxmax(), 'threshold']:.3f}")
        print(f"Lowest Cost: {metrics_df.loc[metrics_df['cost'].idxmin(), 'threshold']:.3f}")
        
        return metrics_df
    
    def save_evaluation_report(self, filepath='../models/evaluation_report.json'):
        """
        Save comprehensive evaluation report
        
        Args:
            filepath: Path to save the report
        """
        if not self.evaluation_results:
            print("No evaluation results to save")
            return
        
        # Prepare data for saving
        report_data = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'models': list(self.evaluation_results.keys()),
            'results': self.evaluation_results,
            'summary': {
                'best_f1_model': max(self.evaluation_results.keys(), 
                                   key=lambda x: self.evaluation_results[x]['f1_score']),
                'best_auc_model': max(self.evaluation_results.keys(), 
                                    key=lambda x: self.evaluation_results[x]['auc']),
                'lowest_cost_model': min(self.evaluation_results.keys(), 
                                      key=lambda x: self.evaluation_results[x]['total_cost'])
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Evaluation report saved to {filepath}")

def main():
    """
    Main function to run comprehensive evaluation
    """
    evaluator = ModelEvaluator()
    
    # Load models and data
    result = evaluator.load_models_and_data()
    if result is None:
        return
    
    X_test, y_test = result
    
    # Evaluate all models
    for model_name, model in evaluator.models.items():
        evaluator.comprehensive_evaluation(model, X_test, y_test, model_name)
        evaluator.print_detailed_results(model_name)
    
    # Create comprehensive comparison plots
    evaluator.plot_comprehensive_comparison()
    
    # Threshold analysis for best model
    best_model = max(evaluator.evaluation_results.keys(), 
                    key=lambda x: evaluator.evaluation_results[x]['f1_score'])
    evaluator.threshold_analysis(best_model, X_test, y_test)
    
    # Save evaluation report
    evaluator.save_evaluation_report()
    
    print("\nComprehensive evaluation completed!")

if __name__ == "__main__":
    main()

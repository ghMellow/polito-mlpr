# Exam 1 Solution - SVM Classifier with Hyperparameter Optimization

## Problem

Train and apply an SVM classifier with RBF kernel, optimizing hyperparameters using single-fold cross-validation. Note: This is similar to Exam 1 but with corrected function name.

## Python Code Solution

```python
import numpy as np

# Define hyperparameter grids for optimization
C_values = [0.1, 1.0, 10.0, 100.0]
gamma_values = [0.001, 0.01, 0.1, 1.0]

# Initialize variables to track best hyperparameters
best_C = None
best_gamma = None
best_score = float('inf')  # Assuming lower scores are better

print("Hyperparameter optimization using single-fold cross-validation:")
print("=" * 60)

# Single-fold cross-validation: train on DTR, validate on DVAL
for C in C_values:
    for gamma in gamma_values:
        # Train SVM model with current hyperparameters
        svm_model = trainBFKernelSVM(DTR, LTR, C, gamma)

        # Score validation data (note: corrected function name)
        validation_scores = scoreRBKernelSVM(svm_model, DVAL)

        # Evaluate performance on validation data
        performance_score = evaluateScores(validation_scores, LVAL)

        print(f"C={C}, gamma={gamma}: validation score = {performance_score:.4f}")

        # Update best hyperparameters if current combination is better
        if performance_score < best_score:
            best_score = performance_score
            best_C = C
            best_gamma = gamma

print("\nBest hyperparameters found:")
print(f"Best C = {best_C}")
print(f"Best gamma = {best_gamma}")
print(f"Best validation score = {best_score:.4f}")

# Train final model with best hyperparameters on training data
print("\nTraining final model with best hyperparameters...")
final_svm_model = trainBFKernelSVM(DTR, LTR, best_C, best_gamma)

# Evaluate on test data
print("Evaluating on test data...")
test_scores = scoreRBKernelSVM(final_svm_model, DTE)
test_performance = evaluateScores(test_scores, LTE)

print(f"Test set performance: {test_performance:.4f}")

```

## Key Points

1. **Function Name Correction**: The scoring function is `scoreRBKernelSVM` (not `scoreRBFKernelSVM` as in Exam 1).
2. **Same Approach**: Uses single-fold cross-validation with DTR/LTR for training and DVAL/LVAL for validation.
3. **Grid Search**: Systematically tests combinations of C and gamma values.
4. **Model Selection**: Selects the hyperparameter combination with the best validation performance.
5. **Final Evaluation**: Trains the final model and evaluates on the independent test set.
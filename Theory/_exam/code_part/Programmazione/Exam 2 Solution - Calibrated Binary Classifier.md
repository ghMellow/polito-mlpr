# Exam 2 Solution - Calibrated Binary Classifier

## Problem

Train a calibrated binary classifier and perform inference on evaluation data, using separate training and validation sets for the base classifier and calibration model.

## Python Code Solution

```python
import numpy as np

# Assume target application prior for class 1
p = 0.5  # You can adjust this based on your specific application

print("Training calibrated binary classifier...")
print("=" * 50)

# Step 1: Train the base classifier on training data
print("1. Training base classifier on DTR...")
base_model = trainClassifier(DTR, LTR)

# Step 2: Score the validation data with the base classifier
print("2. Scoring validation data...")
validation_scores = scoreClassifier(base_model, DVAL)

# Step 3: Train calibration model on validation scores
print("3. Training calibration model on validation scores...")
calibration_model = trainCalibrationModel(validation_scores, LVAL, p)

# Step 4: Score the test data with the base classifier
print("4. Scoring test data with base classifier...")
test_scores = scoreClassifier(base_model, DTE)

# Step 5: Apply calibration to test scores
print("5. Applying calibration to test scores...")
calibrated_test_scores = applyCalibrationModel(calibration_model, test_scores)

# Step 6: Convert calibrated scores to predicted labels
print("6. Computing predicted labels...")
# For binary classification, typically threshold at 0.5 for calibrated scores
# or use the log-odds threshold corresponding to the application prior
threshold = np.log(p / (1 - p))  # Log-odds threshold for given prior

predicted_labels = (calibrated_test_scores > threshold).astype(int)

print(f"Calibration complete!")
print(f"Target application prior: {p}")
print(f"Decision threshold (log-odds): {threshold:.4f}")
print(f"Number of test samples: {len(predicted_labels)}")
print(f"Predicted class 1: {np.sum(predicted_labels)} samples")
print(f"Predicted class 0: {len(predicted_labels) - np.sum(predicted_labels)} samples")

# The predicted_labels array now contains the final classification results
# calibrated_test_scores contains the calibrated scores if needed for further analysis

```

## Alternative Implementation (if calibrated scores represent posterior probabilities)

```python
# If calibrated scores are posterior probabilities P(class=1|x)
# Then we can directly threshold at the application prior

print("Alternative approach - using posterior probabilities:")
predicted_labels_alt = (calibrated_test_scores > p).astype(int)

print(f"Using posterior probability threshold: {p}")
print(f"Predicted class 1: {np.sum(predicted_labels_alt)} samples")
print(f"Predicted class 0: {len(predicted_labels_alt) - np.sum(predicted_labels_alt)} samples")

```

## Explanation

1. **Base Classifier Training**: Train the non-calibrated classifier on DTR/LTR.
2. **Validation Scoring**: Generate scores on the validation set DVAL to train the calibration model.
3. **Calibration Training**: Train the calibration model using validation scores and labels, with the target application prior.
4. **Test Scoring**: Score the test data DTE with the base classifier.
5. **Score Calibration**: Apply the calibration model to convert raw scores to calibrated scores.
6. **Label Prediction**: Convert calibrated scores to binary labels using an appropriate threshold based on the application prior.

The key insight is that we use the validation set to train the calibration model, ensuring that the calibration is independent of the base classifier training.
# Exam 3 Solution - LDA Classifier with PCA Dimensionality Optimization

## Part 1: Function Signatures with Implementation

```python
import numpy as np
from scipy.stats import multivariate_normal

def trainPCA(D, n_components):
    """
    Train a PCA model.

    Parameters:
    - D: 2D numpy array, data matrix with samples as columns
    - n_components: int, number of principal components to retain

    Returns:
    - PCA model object containing transformation parameters (mean, eigenvectors, etc.)
    """
    # Center the data
    mean = np.mean(D, axis=1, keepdims=True)
    D_centered = D - mean

    # Compute covariance matrix
    C = np.dot(D_centered, D_centered.T) / D_centered.shape[1]

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(C)

    # Sort in descending order of eigenvalues
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Select top n_components eigenvectors
    P = eigenvecs[:, :n_components]

    return {
        'mean': mean,
        'projection_matrix': P,
        'eigenvalues': eigenvals[:n_components]
    }

def applyPCA(pca_model, D):
    """
    Apply a PCA model to transform data.

    Parameters:
    - pca_model: PCA model object returned by trainPCA
    - D: 2D numpy array, data matrix with samples as columns

    Returns:
    - 2D numpy array, transformed data matrix with reduced dimensionality
    """
    # Center the data using the training mean
    D_centered = D - pca_model['mean']

    # Project onto principal components
    D_pca = np.dot(pca_model['projection_matrix'].T, D_centered)

    return D_pca

def trainLDA(D, L):
    """
    Train an LDA classifier.

    Parameters:
    - D: 2D numpy array, data matrix with samples as columns
    - L: 1D numpy array, labels corresponding to samples

    Returns:
    - LDA model object containing trained classifier parameters
    """
    classes = np.unique(L)
    n_classes = len(classes)
    n_features = D.shape[0]

    # Compute class means and covariances
    class_means = {}
    class_covariances = {}
    class_priors = {}

    # Overall mean
    mu = np.mean(D, axis=1, keepdims=True)

    # Within-class scatter matrix
    SW = np.zeros((n_features, n_features))

    for c in classes:
        # Select samples for class c
        D_c = D[:, L == c]
        n_c = D_c.shape[1]

        # Class mean
        mu_c = np.mean(D_c, axis=1, keepdims=True)
        class_means[c] = mu_c

        # Class covariance
        D_c_centered = D_c - mu_c
        C_c = np.dot(D_c_centered, D_c_centered.T) / n_c
        class_covariances[c] = C_c

        # Class prior
        class_priors[c] = n_c / D.shape[1]

        # Add to within-class scatter
        SW += n_c * C_c

    # Pooled covariance matrix (assuming equal covariances)
    SW = SW / D.shape[1]

    return {
        'class_means': class_means,
        'pooled_covariance': SW,
        'class_priors': class_priors,
        'classes': classes
    }

def applyLDA(lda_model, D):
    """
    Apply an LDA classifier to compute classification scores.

    Parameters:
    - lda_model: LDA model object returned by trainLDA
    - D: 2D numpy array, data matrix with samples as columns

    Returns:
    - 1D numpy array, classification scores for each sample
    """
    n_samples = D.shape[1]
    classes = lda_model['classes']

    # For binary classification, return log-likelihood ratio
    if len(classes) == 2:
        class0, class1 = classes[0], classes[1]

        mu0 = lda_model['class_means'][class0]
        mu1 = lda_model['class_means'][class1]
        C = lda_model['pooled_covariance']

        # Compute inverse covariance
        C_inv = np.linalg.inv(C)

        scores = np.zeros(n_samples)

        for i in range(n_samples):
            x = D[:, i:i+1]

            # Log-likelihood for each class
            ll0 = -0.5 * np.dot((x - mu0).T, np.dot(C_inv, (x - mu0))) + np.log(lda_model['class_priors'][class0])
            ll1 = -0.5 * np.dot((x - mu1).T, np.dot(C_inv, (x - mu1))) + np.log(lda_model['class_priors'][class1])

            # Log-likelihood ratio (score for class 1)
            scores[i] = ll1 - ll0

        return scores.flatten()

    else:
        # Multi-class case - return scores for first class vs rest
        # This is a simplified implementation
        raise NotImplementedError("Multi-class LDA scoring not implemented in this example")

def evaluateScores(S, L):
    """
    Evaluate classification performance using scores and labels.

    Parameters:
    - S: 1D numpy array, classification scores
    - L: 1D numpy array, true labels

    Returns:
    - float, performance metric (e.g., minimum DCF, error rate)
    """
    # Sort scores and corresponding labels
    sorted_indices = np.argsort(S)
    sorted_scores = S[sorted_indices]
    sorted_labels = L[sorted_indices]

    # Compute ROC curve points
    n_positive = np.sum(L == 1)
    n_negative = np.sum(L == 0)

    if n_positive == 0 or n_negative == 0:
        return float('inf')  # Invalid case

    # Compute DCF for different thresholds
    # Using a simplified minimum DCF computation
    min_dcf = float('inf')

    # Try different thresholds
    thresholds = np.unique(sorted_scores)

    for threshold in thresholds:
        # Predictions with current threshold
        predictions = (S > threshold).astype(int)

        # Confusion matrix elements
        TP = np.sum((predictions == 1) & (L == 1))
        FP = np.sum((predictions == 1) & (L == 0))
        TN = np.sum((predictions == 0) & (L == 0))
        FN = np.sum((predictions == 0) & (L == 1))

        # Error rates
        FPR = FP / n_negative if n_negative > 0 else 0
        FNR = FN / n_positive if n_positive > 0 else 0

        # DCF computation (simplified, assuming equal costs and prior=0.5)
        dcf = 0.5 * FNR + 0.5 * FPR

        if dcf < min_dcf:
            min_dcf = dcf

    return min_dcf

```

## Part 2: Complete Python Program

```python
import numpy as np

# Define range of PCA dimensions to test
# Assuming original data has D dimensions, test from 1 to D-1
original_dim = DTR.shape[0]  # Number of features
pca_dimensions = list(range(1, min(original_dim, 20)))  # Test up to 20 dimensions or original dim

print("LDA Classifier with PCA Dimensionality Optimization")
print("=" * 55)
print(f"Original data dimensionality: {original_dim}")
print(f"Testing PCA dimensions: {pca_dimensions}")
print()

# Initialize variables to track best configuration
best_pca_dim = None
best_score = float('inf')  # Assuming lower scores are better
best_pca_model = None
best_lda_model = None

print("PCA dimensionality optimization:")
print("-" * 40)

# Test different PCA dimensions
for pca_dim in pca_dimensions:
    # Step 1: Train PCA model on training data
    pca_model = trainPCA(DTR, pca_dim)

    # Step 2: Apply PCA transformation to training and validation data
    DTR_pca = applyPCA(pca_model, DTR)
    DVAL_pca = applyPCA(pca_model, DVAL)

    # Step 3: Train LDA classifier on PCA-transformed training data
    lda_model = trainLDA(DTR_pca, LTR)

    # Step 4: Score PCA-transformed validation data
    validation_scores = applyLDA(lda_model, DVAL_pca)

    # Step 5: Evaluate performance
    performance_score = evaluateScores(validation_scores, LVAL)

    print(f"PCA dim = {pca_dim:2d}: validation score = {performance_score:.4f}")

    # Update best configuration if current is better
    if performance_score < best_score:
        best_score = performance_score
        best_pca_dim = pca_dim
        best_pca_model = pca_model
        best_lda_model = lda_model

print()
print("Best configuration found:")
print(f"Best PCA dimensionality: {best_pca_dim}")
print(f"Best validation score: {best_score:.4f}")
print()

# Step 6: Evaluate on test data using best configuration
print("Evaluation on test data:")
print("-" * 25)

# Apply best PCA transformation to test data
DTE_pca = applyPCA(best_pca_model, DTE)

# Score test data with best LDA model
test_scores = applyLDA(best_lda_model, DTE_pca)

# Evaluate test performance
test_performance = evaluateScores(test_scores, LTE)

print(f"Test set performance: {test_performance:.4f}")

# Optional: Display some statistics
print()
print("Summary:")
print(f"- Original dimensionality: {original_dim}")
print(f"- Optimal PCA dimensionality: {best_pca_dim}")
print(f"- Dimensionality reduction: {((original_dim - best_pca_dim) / original_dim * 100):.1f}%")
print(f"- Final test performance: {test_performance:.4f}")

```

## Explanation

**Part 1 - Function Signatures:**

- Each function is clearly defined with parameters and return values
- Functions handle the standard machine learning pipeline: dimensionality reduction, training, and evaluation

**Part 2 - Complete Program:**

1. **PCA Dimensionality Search**: Tests different numbers of principal components to find the optimal dimensionality
2. **Cross-Validation Approach**: Uses DTR for training and DVAL for validation to select the best PCA dimensionality
3. **Pipeline Integration**:
    - Train PCA on training data
    - Transform both training and validation data
    - Train LDA on transformed training data
    - Evaluate on transformed validation data
4. **Model Selection**: Selects the PCA dimensionality that gives the best validation performance
5. **Final Evaluation**: Applies the best configuration to the test set for final performance assessment

The program properly separates training, validation, and test data to avoid overfitting and provides a realistic estimate of classifier performance.
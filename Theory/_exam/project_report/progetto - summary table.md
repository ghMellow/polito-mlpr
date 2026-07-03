# Fingerprint Classification Models - Performance Summary

| Rank | Model | Variant | Error Rate (%) | minDCF | Why This Performance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Diagonal GMM** | N₍ₕ₀₎=8, N₍ₕ₁₎=32 | **5.20** | **0.131** | **Best overall**: Multiple components capture multimodality; diagonal assumption matches low feature correlation; excellent calibration |
| 2 | **Full GMM** | N₍ₕ₀₎=1, N₍ₕ₁₎=16 | **9.95** | 0.150 | **Spurious correlations**: Full covariance introduces bias from non-existent correlations; diagonal version performs much better |
| 3 | **RBF SVM** | γ=e⁻², C=32 | **4.45** | 0.177 | **Best error rate**: Can model arbitrarily complex decision boundaries, perfectly capturing multimodal distributions and non-linear separations |
| 4 | **Quadratic LR** | Feature expansion Φ(x) | **5.90** | 0.244 | **Non-linear capture**: Feature space expansion models quadratic interactions missing in linear version |
| 5 | **Polynomial SVM** | d=2, c=1, C=3.2e-5 | **7.80** | 0.245 | **Good quadratic modeling**: Captures curved decision boundaries better than linear models, suitable for dataset's quadratic nature |
| 6 | **Naive Gaussian** | Diagonal covariance | **7.20** | 0.259 | **Confirms decorrelation**: Nearly matches MVG performance, validating low feature correlation assumption |
| 7 | **MVG** | Full covariance | **7.00** | 0.263 | **Decent baseline**: Models separate covariances per class but limited by unimodal Gaussian assumption for multimodal features f₅,f₆ |
| 8 | **Linear SVM** | C = 1.0 | **9.05** | 0.358 | **Linear limitation**: Assumes linear boundaries inadequate for curved class separations; poor calibration |
| 9 | **Linear LR** | λ = 1.0e-2 | **9.25** | 0.361 | **Linear constraint**: Cannot capture quadratic decision boundaries present in dataset; overly restrictive assumptions |
| 10 | **Tied Gaussian** | Shared covariance | **9.30** | 0.363 | **Violated assumptions**: Assumes identical covariances between classes, contradicted by significantly different class variances |

## Key Insights

### **Top Performers (minDCF < 0.200)**

- **Complex boundary models excel**: RBF SVM and GMM can capture the dataset's intrinsic complexity
- **Multimodal modeling crucial**: Features f₅,f₆ show three-peak distributions requiring flexible models

### **Mid-Range Performers (minDCF 0.200 - 0.300)**

- **Quadratic approaches**: Both Polynomial SVM and Quadratic LR benefit from modeling curved boundaries
- **Gaussian baselines**: MVG and Naive Gaussian provide solid performance confirming low feature correlation

### **Poor Performers (minDCF > 0.300)**

- **Linear models struggle**: Cannot capture the dataset's quadratic decision boundaries
- **Violated assumptions**: Tied Gaussian fails due to different class variances
- **Over-parameterized**: Full GMM suffers from modeling non-existent correlations

### **Best Overall Choice**

**Diagonal GMM** offers the optimal balance: excellent error rate (5.20%), best minDCF (0.131), and superior calibration, making it the most robust choice across different applications.
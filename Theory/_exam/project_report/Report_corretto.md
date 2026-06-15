# Politecnico di Torino
### Master's degree in Computer Engineering

* **Academic year:** 2023-2024
* **Course notes:** Machine learning and pattern recognition
* **Professor:** Sandro Cumani
* **Author:** Alberto Ciccomascolo
* **Student ID:** 333968

---

## 1. Dataset visualization

The training dataset is composed by 6000 samples, each one corresponding to a set of data related to a single fingerprint. Every sample is described by 6 different features and can belong to two classes, that is, a genuine (labelled as 1 or h1) or fake (labelled as 0 or h0) fingerprint. The dataset contains a balanced presence of both classes (3010 genuine fingerprints and 2990 fake fingerprints).

The following histograms show the (normalized) distribution of all the samples with respect to each feature:

*[Figura: Sei istogrammi che confrontano la distribuzione normalizzata di impronte vere e false per ciascuna delle 6 feature]*

On a first look, the dataset seems to show similar distributions (sometimes mirrored) related to the samples among pair-wise features, that is, the feature pairs (f1,f2), (f3,f4), (f5,f6). We can better observe the behavior of the pair-wise feature distributions by plotting the related scatter plots.

*[Figura: Tre grafici scatter che mostrano la distribuzione congiunta delle coppie di feature (f1,f2), (f3,f4) e (f5,f6)]*

Moreover, in order to better understand the data, we can compute the values of the mean and the variance associated to each univariate feature distribution and divided for class (genuine or fake fingerprint). The resulting values can be found in the following table:

| | Mean Genuine | Mean Fake | Variance Genuine | Variance Fake |
| :--- | :--- | :--- | :--- | :--- |
| **feature 1** | 5.44547838e-04 | 0.00287744 | 1.43023345 | 0.56958105 |
| **feature 2** | -8.52437392e-03 | 0.01869316 | 0.57827792 | 1.42086571 |
| **feature 3** | 6.65237846e-01 | -0.68094016 | 0.5489026 | 0.54997702 |
| **feature 4** | -6.64195349e-01 | 0.6708362 | 0.55334275 | 0.53604266 |
| **feature 5** | -4.17251858e-02 | 0.02795697 | 1.31776792 | 0.6800736 |
| **feature 6** | 2.39384879e-02 | -0.0058274 | 1.28702609 | 0.70503844 |

By looking at the plots and at the previous table, we can clearly say that:
* The first and second features have a similar overall plot, but they show an opposite class distribution. In both cases the two classes overlap over an interval of values that is approximately equal to [-2,2], where there is the higher sample frequency. This means that most of the feature values are covered by samples of both the class 0 (fake) and the class 1 (genuine). Only genuine class for feature f1 and fake class for feature f2 can assume values without any overlap with the opposite class; in fact, they have a higher spread. Despite this, both the classes for features f1 and f2 have a similar (but not exactly equal) mean, which tends to zero. In fact, the code shows that for feature f1 the variance of the genuine class is higher than the variance of the fake class and for feature f2 the variance of the genuine class is lower than the variance of the fake class. For both features, the two classes show essentially a single mode, that is located near the related mean value.
* The third and fourth features show again a similar graphical behavior as the overall plots have the same shape, but the class distributions are mirrored. In this case we can observe two peaks for each plot, one corresponding to a genuine sample and the other corresponding to a fake sample. The class overlap interval is the same for both features and corresponds approximately to the interval [-2,2]. Apart from this, the scatter plot indicates an overlapping that is smaller than the one of the previous case, denoting more uncorrelation.
* The fifth and sixth features have again the same overall plot shape, but this time the single classes also show an equal behavior among the two features, that is, the distributions are not mirrored. In fact, in both plots, we can observe a central peak corresponding to a fake sample and two main side peaks corresponding to genuine samples, which are symmetrical with respect to the origin. As the scatter plot shows, the fact that the peak of a class corresponds to low frequency intervals of the other class, makes the overlapping area less impactful.

---

## 2. Dimensionality reduction

### 2.1 Principal Component Analysis (PCA)
The first approach for reducing the dimensionality of the dataset is applying PCA. We start by retrieving the 6 PCA directions ui and then we project the dataset over each one of these directions (starting from the principal) by calculating D_hat = ui^T * D where D is the dataset and D_hat is the resulting projected dataset. This means reducing the dimensionality of the problem from the initial value of n=6 to m=1.

*[Figura: Sei istogrammi che mostrano la proiezione del dataset lungo le 6 direzioni della PCA in ordine decrescente di varianza conservata]*

What we can notice from the previous plots is the fact that the less is the variance preserved by the projection direction, the less distinguishable are the classes distributions. In particular, the dataset projected over the principal direction (direction 1 in the plot) is the one that is better able to separate the two classes into two clusters.

### 2.2 Linear Discriminant Analysis (LDA)
A better way to reduce the dataset dimensionality preserving class-discriminant information is applying LDA, which prioritize the projection directions that allow to better separate the classes. In this case, since we are dealing with a binary problem, LDA can retrieve only one discriminant direction over which to project the dataset.

*[Figura: Istogramma della proiezione del dataset lungo la singola direzione LDA]*

With LDA we can notice that the resulting dataset projection does not differ too much from the dataset projection over the PCA direction with the larger variance. LDA is particularly useful for classification purposes. In order to perform a classification, we need:
* a classifier
* a training set, over which we train the classifier
* a validation set, over which we evaluate the performance classifier

The last two points can be obtained by splitting the original dataset into a training set (consisting of 2/3 of the total samples) and a validation set (consisting of 1/3 of the total samples).

*[Figura: Istogrammi delle proiezioni del training set e del validation set lungo la direzione LDA]*

These two plots look quite similar to the plot of the overall dataset, implying that the dataset has a low internal variability, making overfitting a minor issue. We then choose a simple classifier based on a threshold defined as:

t = (mean_genuine + mean_fake) / 2

where mean_genuine is the mean of genuine samples and mean_fake is the mean of fake samples as calculated from the training set. Then, we assign a label to a test sample (included in the validation set) with the following rule:
h0 if x < t, h1 if x >= t.

| Threshold t | Total validation samples | Total classification errors | Error rate |
| :--- | :--- | :--- | :--- |
| -0.018534376786207174 | 2000 | 186 | 9.30% |

### 2.3 LDA + PCA as pre-processing
A more complete approach consists in combining PCA and LDA by using PCA as a pre-processing technique for reducing the dimensionality of the training set and then using LDA as a classification tool as shown before.

| PCA dimensions | Threshold t | Misclassifications | Error rate |
| :--- | :--- | :--- | :--- |
| m=6 | -0.018534376786207285 | 186 | 9.30% |
| m=5 | -0.018519178157787586 | 186 | 9.30% |
| m=4 | -0.01831713244834865 | 185 | 9.25% |
| m=3 | -0.018398189889104022 | 185 | 9.25% |
| m=2 | -0.01828175332678572 | 185 | 9.25% |
| m=1 | -0.01759903023797671 | 187 | 9.35% |

---

## 3. Gaussian generative models
A more robust classification model that we can apply to the problem is a generative model, specifically the Multivariate Gaussian Classifier, for which we assume that the class-conditional probability of each class c can be approximated with a gaussian distribution.

*[Figura: Sei grafici che mostrano il fit gaussiano sovrapposto agli istogrammi di ciascuna classe per ogni feature]*

From these plots we can notice that the gaussian fitting does not have the same goodness for all the features:
* The gaussian fitting seems to be particularly effective for both classes in features f1, f2, f3 and f4.
* On the other hand, the gaussian fitting is much less accurate with both classes of features f5 and f6.

### 3.1 Multivariate Gaussian model (MVG)
The first gaussian classification model we apply is the default Multivariate Gaussian classifier (MVG), for which we model the likelihood distribution as a normal distribution where mean and covariance matrix are related to class c. Then we can use this PDF to evaluate the optimal Bayes decision for a test sample by calculating the related log-posterior ratio.

Since we assume that the priors are equal, the prior-dependent term is null and we can classify the samples only according to the log-likelihood ratio.

| Total validation samples | Total classification errors | Error rate |
| :--- | :--- | :--- |
| 2000 | 140 | 7.00% |

### 3.2 Gaussian model with tied covariances
We now try to apply the tied gaussian model, which assumes that all the sample clusters are spread in the same way around their mean, that is, they have the same covariance matrix.

| Total validation samples | Total classification errors | Error rate |
| :--- | :--- | :--- |
| 2000 | 186 | 9.30% |

### 3.3 Naive Gaussian model
The naive gaussian model assumes uncorrelation between the features of each sample.

| Total validation samples | Total classification errors | Error rate |
| :--- | :--- | :--- |
| 2000 | 144 | 7.20% |

*[Figura: Due matrici di correlazione termica (heatmaps) calcolate secondo Pearson per le impronte Fake e Genuine]*

### 3.4 Gaussian models goodness evaluation
To verify the fitting inaccuracy on features f5 and f6, we can repeat the three gaussian classifications excluding them.

| Model | Classification errors | Error rate |
| :--- | :--- | :--- |
| Multivariate gaussian | 159 | 7.95% |
| Tied gaussian | 190 | 9.50% |
| Naive Bayes gaussian | 153 | 7.65% |

For features f1 and f2 we obtain the following results:

| Model | Classification errors | Error rate |
| :--- | :--- | :--- |
| Multivariate gaussian | 730 | 36.50% |
| Tied gaussian | 989 | 49.45% |
| Naive Bayes gaussian | 726 | 36.30% |

For features f3 and f4 we obtain the following results:

| Model | Classification errors | Error rate |
| :--- | :--- | :--- |
| Multivariate gaussian | 189 | 9.45% |
| Tied gaussian | 188 | 9.40% |
| Naive Bayes gaussian | 189 | 9.45% |

### 3.5 Classification with PCA as pre-processing

| PCA dimension | Multivariate gaussian | Tied gaussian | Naive gaussian |
| :--- | :--- | :--- | :--- |
| 1 | 9.25% | 9.35% | 9.25% |
| 2 | 8.80% | 9.25% | 8.85% |
| 3 | 8.80% | 9.25% | 9.00% |
| 4 | 8.05% | 9.25% | 8.85% |
| 5 | 7.10% | 9.30% | 8.75% |
| 6 | 7.00% | 9.30% | 8.90% |

---

## 4. Bayes decisions and model evaluation
In this chapter we try to analyze our models according to the Bayes risk and introducing costs related to different types of mis-classifications.

### 4.1 Application (0.5, 1.0, 1.0)
For each model we define the actual (normalized) DCF.

| PCA | Multivariate Gaussian minDCF | Multivariate Gaussian actDCF | Tied Gaussian minDCF | Tied Gaussian actDCF | Naive Gaussian minDCF | Naive Gaussian actDCF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| No PCA | 0.130 | 0.140 | 0.181 | 0.186 | 0.131 | 0.144 |
| m=6 | 0.130 | 0.140 | 0.181 | 0.186 | 0.173 | 0.178 |
| m=5 | 0.133 | 0.142 | 0.181 | 0.186 | 0.174 | 0.175 |
| m=4 | 0.154 | 0.161 | 0.183 | 0.185 | 0.173 | 0.177 |
| m=3 | 0.173 | 0.176 | 0.183 | 0.185 | 0.176 | 0.180 |
| m=2 | 0.173 | 0.176 | 0.179 | 0.185 | 0.171 | 0.177 |
| m=1 | 0.177 | 0.185 | 0.177 | 0.187 | 0.177 | 0.185 |

### 4.4 Bayes error plot
*[Figura: Grafico dell'errore di Bayes (Bayes Error Plot) per i modelli gaussiani senza PCA pre-processing. Mostra l'actDCF e il minDCF al variare del priore effettivo.]*

---

## 5. Logistic regression model

### 5.1 Non prior-weighted model employment

| lambda | Error rate | actDCF | minDCF |
| :--- | :--- | :--- | :--- |
| 1.0e-4 | 9.30% | 0.403 | 0.364 |
| 3.2e-4 | 9.35% | 0.407 | 0.365 |
| 1.0e-3 | 9.35% | 0.414 | 0.365 |
| 3.2e-3 | 9.25% | 0.430 | 0.364 |
| 1.0e-2 | 9.25% | 0.460 | 0.361 |
| 3.2e-2 | 9.20% | 0.584 | 0.362 |
| 1.0e-1 | 9.25% | 0.853 | 0.364 |
| 3.2e-1 | 9.30% | 0.995 | 0.364 |
| 1.0 | 9.25% | 1.000 | 0.364 |
| 3.2 | 9.25% | 1.000 | 0.364 |
| 1.0e1 | 9.50% | 1.000 | 0.363 |
| 3.2e1 | 9.60% | 1.000 | 0.362 |
| 1.0e2 | 12.85% | 1.000 | 0.362 |

*[Figura: Grafici di actDCF e minDCF in funzione di lambda per il training set intero e un set ridotto]*

### 5.3 Quadratic logistic regression

| lambda | Error rate | actDCF | minDCF |
| :--- | :--- | :--- | :--- |
| 1.0e-4 | 6.05% | 0.280 | 0.260 |
| 3.2e-4 | 6.05% | 0.267 | 0.261 |
| 1.0e-3 | 5.95% | 0.278 | 0.259 |
| 3.2e-3 | 5.90% | 0.280 | 0.253 |
| 1.0e-2 | 5.90% | 0.348 | 0.249 |
| 3.2e-2 | 5.90% | 0.500 | 0.244 |
| 1.0e-1 | 6.05% | 0.757 | 0.247 |
| 3.2e-1 | 6.10% | 0.965 | 0.263 |
| 1.0 | 6.40% | 1.000 | 0.284 |
| 3.2 | 7.10% | 1.000 | 0.309 |
| 1.0e1 | 7.10% | 1.000 | 0.324 |
| 3.2e1 | 7.35% | 1.000 | 0.326 |
| 1.0e2 | 9.30% | 1.000 | 0.326 |

---

## 6. Support Vector Machines (SVM)

### 6.1 Linear SVM
The primary assumption of SVM for linearly separable classes is that there exists a decision surface that maximizes the margin.

| C | Error rate | actDCF | minDCF |
| :--- | :--- | :--- | :--- |
| 1.0e-5 | 50.40% | 1.000 | 1.000 |
| 3.2e-5 | 9.15% | 1.000 | 0.362 |
| 1.0e-4 | 9.20% | 1.000 | 0.364 |
| 3.2e-4 | 9.25% | 0.999 | 0.362 |
| 1.0e-3 | 9.30% | 0.963 | 0.362 |
| 3.2e-3 | 9.35% | 0.848 | 0.365 |
| 1.0e-2 | 9.25% | 0.677 | 0.362 |
| 3.2e-2 | 9.15% | 0.583 | 0.359 |
| 1.0e-1 | 9.15% | 0.523 | 0.358 |
| 3.2e-1 | 9.10% | 0.501 | 0.358 |
| 1.0 | 9.05% | 0.491 | 0.358 |

### 6.3 SVM with Radial Basis Function (RBF) kernel

*[Figura: Grafico di performance (actDCF e minDCF) in funzione di C per SVM con RBF kernel a vari livelli di gamma]*

---

## 7. Gaussian Mixture Models (GMM)

### Full covariance GMM
| | 1 | 2 | 4 | 8 | 16 | 32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Error: 13.75%, actDCF: 0.305, minDCF: 0.263 | Error: 13.75%, actDCF: 0.237, minDCF: 0.265 | Error: 9.90%, actDCF: 0.305, minDCF: 0.214 | Error: 9.05%, actDCF: 0.196, minDCF: 0.185 | Error: 9.95%, actDCF: 0.206, minDCF: 0.150 | Error: 11.05%, actDCF: 0.227, minDCF: 0.185 |
| **8** | Error: 7.25%, actDCF: 0.200, minDCF: 0.176 | Error: 7.20%, actDCF: 0.191, minDCF: 0.181 | Error: 5.95%, actDCF: 0.199, minDCF: 0.196 | Error: 6.05%, actDCF: 0.193, minDCF: 0.179 | Error: 6.25%, actDCF: 0.172, minDCF: 0.153 | Error: 7.15%, actDCF: 0.190, minDCF: 0.175 |

### Diagonal GMM
| | 1 | 2 | 4 | 8 | 16 | 32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **8** | Error: 6.85%, actDCF: 0.176, minDCF: 0.173 | Error: 7.00%, actDCF: 0.187, minDCF: 0.176 | Error: 5.60%, actDCF: 0.183, minDCF: 0.158 | Error: 5.45%, actDCF: 0.180, minDCF: 0.146 | Error: 5.05%, actDCF: 0.149, minDCF: 0.132 | Error: 5.20%, actDCF: 0.152, minDCF: 0.131 |

---

## 8. Classification models recap

| Model | Error rate | Best minDCF | Model parameters |
| :--- | :--- | :--- | :--- |
| MVG | - | 0.263 | No PCA |
| Tied gaussian | - | 0.363 | No PCA |
| Naive gaussian | - | 0.259 | No PCA |
| Linear non prior-weighted LR | 9.25% | 0.361 | lambda=1.0e-2 |
| Linear prior-weighted LR | 50.40% | 0.362 | lambda=3.2e1 |
| Quadratic non prior-weighted LR | 5.90% | 0.244 | lambda=3.2e-2 |
| Linear SVM | 9.05% | 0.358 | C=1.0 |
| Polynomial SVM | 7.80% | 0.245 | C=3.2e-5 |
| RBF SVM | 4.45% | 0.177 | gamma=e-2, C=3.2e1 |
| Full covariance GMM | 9.95% | 0.150 | N_h0=1 N_h1=16 |
| Diagonal GMM | 5.20% | 0.131 | N_h0=8 N_h1=32 |

---

## 9. Calibration and fusion

### 9.1 Calibration

| Scores | Quadratic LR actDCF | Quadratic LR minDCF | SVM RBF actDCF | SVM RBF minDCF | Diagonal GMM actDCF | Diagonal GMM minDCF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Raw | 0.500 | 0.244 | 0.431 | 0.177 | 0.152 | 0.131 |
| Calibrated | 0.272 | 0.248 | 0.179 | 0.188 | 0.150 | 0.133 |

*[Figura: Grafico dell'errore di Bayes che confronta i modelli calibrati (LR, SVM, GMM) e pre-calibrati al variare delle applicazioni]*

### 9.2 Fusion

| Score fusion (LR SVM GMM) | Error rate | actDCF | minDCF |
| :--- | :--- | :--- | :--- |
| | 5.10% | 0.166 | 0.127 |

---

## 10. Final evaluation

| Diagonal GMM with 8 and 32 components | Error rate | actDCF | minDCF |
| :--- | :--- | :--- | :--- |
| | 5.40% | 0.193 | 0.181 |

*[Figura: Plot finale dell'errore di Bayes che testa il modello Diagonal GMM scelto (8 e 32 componenti) sul set di valutazione per varie applicazioni target]*

| Scores | Quadratic LR actDCF | Quadratic LR minDCF | SVM RBF actDCF | SVM RBF minDCF | Diagonal GMM actDCF | Diagonal GMM minDCF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Calibrated | 0.366 | 0.352 | 0.270 | 0.264 | 0.190 | 0.187 |

| Score fusion (LR SVM GMM) | Error rate | actDCF | minDCF |
| :--- | :--- | :--- | :--- |
| | 6.15% | 0.200 | 0.189 |

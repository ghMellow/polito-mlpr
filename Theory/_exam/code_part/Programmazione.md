# Programmazione

# exam 1

You are given the following functions (assume these functions are already implemented):

- trainBFKernelSVM(D, L, C, gamma): trains an SVM model with an RBF kernel with hyper-parameter gamma and returns an object containing the trained model information; D is the training data matrix, L is the corresponding label array, and C is the SVM cost-vs-margin trade-off coefficient
- scoreRBKernelSVM(smModel, D): computes the classification scores for samples in the data matrix D for an SVM model svmModel (as returned by the function trainBFKernelSVM) and returns an array of scores
- evaluateScores (S, L): computes an evaluation metric (e.g. minimum DCF) over the array of scores S with associated array of labels L

Assume that you have at your disposal a training set, already divided in model training data (DTR, LTR) and validation data (DVAL, LVAL), and an evaluation set (DTE, LTE). DTR, DVAL and DTE are data matrices, with samples organized as column vectors, whereas LTR, LVAL and LTE are arrays containing the corresponding labels.

Write the Python code to train and apply an SVM classifier. In particular, the code should

- Train an SVM classifier, optimizing the value of the hyper-parameters with respect to the metric function evaluateScores using a single-fold cross-validation approach.
- Evaluate the selected SVM model on the evaluation data, using the provided metric.

[Exam 1 Solution - SVM Classifier with Hyperparameter Optimization](Programmazione/Exam%201%20Solution%20-%20SVM%20Classifier%20with%20Hyperparamet%202626adce669080c98287c854390f795b.md)

---

# Exam 2

Consider a binary classification problem, with classes labeled as 1 and 0, respectively.

Let (DTR, LTR), (DVAL, LVAL) represent a labeled training set and a labeled validation set. DTR and DVAL are 2-D numpy arrays containing the dataset samples (stored as column vectors), whereas LTR and LVAL are 1-D numpy arrays containing the sample labels. Let also DTE represent the dataset matrix (again, a 2-D numpy array) containing the samples that our application should classify.

Write a Python code fragment that:

1. trains a calibrated binary classifier
2. performs inference (i.e. computes predicted labels) on the evaluation data

You can assume that the following functions have been defined:

- trainClassifier (D, L): train a non-calibrated classification model (e.g., an SVM or an LDA classifier) on the training matrix D with associated labels array L, and return a python object containing the trained model (assume that the model does not contain tunable hyper-parameters)
- scoreClassifier (model, D) : compute the non-calibrated classification scores for model model (as returned by trainClassifier) for the samples in data matrix D and return a 1-D array of scores
- trainCalibrationModel(S, L, prior): train a calibration model on the 1-D array of scores S, with associated array of labels L, for a binary application with prior prior for class 1, and return a python object containing the trained model
- applyCalibrationModel (calModel, S) : apply the calibration model calModel (as returned by trainCalibrationModel) to the 1-D array of scores S, and return a 1-D array of calibrated scores

NOTE: assume that the target application is characterized by an effective prior p for class 1.

You are not required to tune the calibration model hyper-parameter prior, but you can assume that the calibration model can be trained using the target application prior p.

[Exam 2 Solution - Calibrated Binary Classifier](Programmazione/Exam%202%20Solution%20-%20Calibrated%20Binary%20Classifier%202626adce669080ef8fe1d84990aed61d.md)

---

# Exam 3

Given the following functions (assume these functions are already implemented):

- trainPCA: trains a PCA model
- applyPCA: applies a PCA model to some data
- trainClassifier(D, L): trains a given classifier from the data matrix D and the label
vector L; returns an object containing the trained model parameters
- scoreClassifier(clsModel, D): computes the array of scores for classifier clsModel (as
returned by the function trainClassifier) for the samples in data matrix D
- evaluateScores(S, L): computes a performance metric (e.g. minimum DCF) over the
score array S with label vector L

a) Provide a possible signature and an implementation of the functions trainPCA and applyPCA,
briefly explaining also the function parameters and the return value.

b) Using these functions, write the Python code to:

- Train the classifier on a training set, optimizing the PCA dimension with respect to the
provided metric function using a single-fold cross-validation approach
- Evaluate its performance on an evaluation set.
Assume that you have at your disposal a training set, already divided in model training data
(DTR, LTR) and validation data (DVAL, LVAL), and an evaluation set (DTE, LTE). DTR, DVAL
and DTE are data matrices, with samples organized as column vectors, whereas LTR, LVAL and LTE are arrays containing the corresponding labels. To select the PCA dimension m consider all possible values of m that are compatible with the dimension of the feature vectors. Assume that the classifier is invariant to aﬃne transformations, that it does not include hyper-parameters to tune, and that PCA is the only kind of pre-processing to analyze.

[Exam 3 Solution - LDA Classifier with PCA Dimensionality Optimization](Programmazione/Exam%203%20Solution%20-%20LDA%20Classifier%20with%20PCA%20Dimensio%202626adce669080f39fd2ca93f75b493d.md)

---
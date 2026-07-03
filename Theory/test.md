---
title: "Multivariate Gaussian classifier assumptions"
source: "https://claude.ai/chat/ce0b9b89-fa50-4dbf-9fc3-985619ff68af"
author:
published:
created: 2026-07-03
description: "Claude conversation with 2 messages"
tags:
  - "clippings"
---
## Multivariate Gaussian Classifier

**Model assumptions**

The MVG is a generative model: we model the joint distribution of feature vectors and class labels, factorized as

$$
f_{X,C}(x_t, c) = f_{X|C}(x_t \mid c)\, P_C(c)
$$

We assume that samples are i.i.d. and that the class-conditional likelihood of each class is a multivariate Gaussian, with a class-dependent mean and covariance matrix:

$$
(X_t \mid C_t = c) \sim \mathcal{N}(\mu_c, \Sigma_c), \qquad f_{X_t|C_t}(x_t \mid c) = \mathcal{N}(x_t \mid \mu_c, \Sigma_c)
$$

The class prior $P_C(c)$ is not a model parameter: it depends on the application.

**Estimation of the model parameters**

The parameters $\theta = [(\mu_1, \Sigma_1) \ldots (\mu_k, \Sigma_k)]$ are estimated by maximum likelihood from the labeled training set $\mathcal{D} = \{(x_1,c_1)\ldots(x_n,c_n)\}$ . Since observations are i.i.d., the log-likelihood factorizes over classes, so each pair $(\mu_c, \Sigma_c)$ can be estimated independently from the samples of class $c$ :

$$
\ell(\theta) = \sum_{c} \sum_{i \mid c_i = c} \log \mathcal{N}(x_i \mid \mu_c, \Sigma_c)
$$

Setting the gradient with respect to $\mu_c$ and $\Lambda_c = \Sigma_c^{-1}$ to zero yields the empirical mean and covariance of each class:

$$
\mu_c^* = \frac{1}{N_c}\sum_{i \mid c_i = c} x_i, \qquad \Sigma_c^* = \frac{1}{N_c}\sum_{i \mid c_i = c}(x_i - \mu_c^*)(x_i - \mu_c^*)^T
$$

**Inference — multiclass**

For a test sample $x_t$ we plug the estimates into Bayes' theorem to obtain class posteriors:

$$
P(C = h_i \mid x_t) = \frac{f_{X|C}(x_t \mid h_i)\,P(h_i)}{\sum_{h'} f_{X|C}(x_t \mid h')\,P(h')}
$$

The optimal Bayes decision selects the class with highest posterior; since the denominator is the same for all classes,

$$
c_t^* = \arg\max_h\; \log f_{X|C}(x_t \mid h) + \log P(h)
$$

The first term is the classifier output, the second depends on the application.

**Inference — binary**

For $C \in \{h_1, h_0\}$ with prior $\pi = P(h_1)$ , comparing posteriors is equivalent to computing the log posterior ratio

$$
\log r(x_t) = \underbrace{\log \frac{f_{X|C}(x_t \mid h_1)}{f_{X|C}(x_t \mid h_0)}}_{\mathrm{llr}(x_t)} + \log \frac{\pi}{1-\pi}
$$

The classifier should output the llr, which acts as a score independent of the application; the decision $\log r(x_t) \gtrless 0$ becomes a comparison with an application-dependent threshold:

$$
\mathrm{llr}(x_t) \gtrless -\log\frac{\pi}{1-\pi}
$$

**Decision rules for binary problems**

Substituting the Gaussian densities (with $\Lambda_c = \Sigma_c^{-1}$ ), the llr is a quadratic function of $x$ :

$$
\mathrm{llr}(x) = x^T A x + x^T b + c
$$

with

$$
A = -\tfrac{1}{2}(\Lambda_1 - \Lambda_0), \quad b = \Lambda_1\mu_1 - \Lambda_0\mu_0, \quad c = -\tfrac{1}{2}\!\left(\mu_1^T\Lambda_1\mu_1 - \mu_0^T\Lambda_0\mu_0\right) + \tfrac{1}{2}\!\left(\log|\Lambda_1| - \log|\Lambda_0|\right)
$$

so the decision surfaces (points where the assigned class changes) are quadratic surfaces.

**Naive Bayes variant**

Naive Bayes assumes feature independence given the class, so the class likelihood factorizes over the components, each modeled as a univariate Gaussian:

$$
f_{X|C}(x \mid c) \approx \prod_{j=1}^{D} \mathcal{N}(x^{[j]} \mid \mu_{c,[j]},\, \sigma^2_{c,[j]})
$$

The ML solution is obtained independently per component:

$$
\mu_{c,[j]}^* = \frac{1}{N_c}\sum_{i \mid c_i = c} x_{i,[j]}, \qquad \sigma^{2*}_{c,[j]} = \frac{1}{N_c}\sum_{i \mid c_i = c}(x_{i,[j]} - \mu_{c,[j]}^*)^2
$$

This corresponds to an MVG with diagonal covariance matrices $\Sigma_c = \mathrm{diag}(\sigma^2_{c,[1]},\ldots,\sigma^2_{c,[D]})$ . Inference is unchanged, and since the $\Sigma_c$ are in general different across classes, the binary llr is still quadratic (with diagonal $A$ ). The benefit is the drastic reduction of parameters ($D$ instead of $D(D+1)/2$ per covariance), which makes estimation more robust when we have few samples compared to the dimensionality; the limitation is that the model cannot capture within-class feature correlations, so it may perform poorly on strongly correlated data.

**Tied covariance variant**

Here each class keeps its own mean but all classes share the same covariance matrix:

$$
f_{X|C}(x \mid c) = \mathcal{N}(x \mid \mu_c, \Sigma)
$$

The ML estimates are the per-class means and the within-class covariance averaged over all samples:

$$
\mu_c^* = \frac{1}{N_c}\sum_{i \mid c_i=c} x_i, \qquad \Sigma^* = \frac{1}{N}\sum_c \sum_{i \mid c_i=c}(x_i - \mu_c^*)(x_i - \mu_c^*)^T
$$

Since $\Lambda_1 = \Lambda_0 = \Lambda = \Sigma^{-1}$ , the quadratic term of the llr cancels and the binary decision rule becomes linear:

$$
\mathrm{llr}(x) = x^T b + c, \qquad b = \Lambda(\mu_1 - \mu_0), \qquad c = -\tfrac{1}{2}\!\left(\mu_1^T\Lambda\mu_1 - \mu_0^T\Lambda\mu_0\right)
$$

The benefit is that a single shared covariance is estimated from all $N$ samples, so the estimate is more reliable with scarce data, while correlations between features can still be captured. The limitation is that the model may perform poorly when the classes actually have very different covariances, and it can only produce linear separation surfaces; conversely, if we have reason to believe the class covariances are similar, tying provides a more reliable estimate than the unconstrained model.

---


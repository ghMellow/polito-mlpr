# PCA — Practice attempts

Log of handwritten practice answers, transcribed, with judgment/corrections.
New attempts get appended below with an incremented number.

---

## Attempt 1

### Transcription

PCA (as a pre-processing technique)

PCA is an unsupervised dimensionality-reduction technique. It retains the direction(s) that maximize the variance, and it is used to reduce the number of features, to prevent overfitting, reduce the noise and remove problematic directions.

So it reduces the dimensionality of the space from $n$ to $k$ ($k<n$), and projects the samples by $Y=P^TX$, while the projection back in the original space is given by

$\tilde X = PY = PP^TX$

The objective is to minimize the average reconstruction error (MSE):

$P^* = \arg\min_P \frac1k\sum_{i=1}^{k}\|x_i - PP^Tx_i\|^2 \quad \text{with } PP^T=I$

This is equivalent to maximizing the trace:

$L = \mathrm{Tr}\Big(P^T \underbrace{\big(\tfrac1k\sum x x^T\big)}_{C} P\Big)$

The solution follows from the eigenvalue decomposition $C=U\Sigma U^T$.

The eigenvectors with the biggest eigenvalues, taken in decreasing order, are

$P^*=[c_1,c_2,\dots,c_m]$

— Furthermore, we need to choose the number of dimensions to retain; otherwise we might keep a non-discriminative/uninformative direction.

— The value of $m$ is a hyperparameter. There are 2 ways to estimate it:

$C=\frac1N\sum x x^T - \mu$

— choose a threshold that retains a certain amount of variance, or find the best value via cross-validation.

**Limitation:** being unsupervised, there's no guarantee that the directions we keep are the most discriminative ones. Too few and we lose useful information, too many and we keep noise/overfit. Plus, PCA can't be used by itself as a classification technique — it can only be used as a dimensionality-reduction technique combined with models that perform classification.

### Judgment / Correction

**Solid overall** — the core idea (max variance ⇔ min reconstruction error, eigen-decomposition of the covariance, components ordered by eigenvalue) is correctly grasped, and the limitations/variants pillar (choice of $m$ via variance threshold or CV, unsupervised ⇒ no guarantee of discriminativeness, can't classify alone) is well covered.

**Things to fix:**

- **Constraint is wrong**: it should be $P^TP=I_k$ (orthonormal columns of $P$, an $n\times k$ matrix), not $PP^T=I$. $PP^T=I$ would force $k=n$ — it's the projection-then-reconstruction operator $PP^T$ that approximates the identity, but the constraint on $P$ itself is $P^TP=I_k$.
- **Covariance matrix definition is off**: "$C=\frac1N\sum xx^T-\mu$" mixes a matrix and a vector. The correct definition requires **centering**: $C=\frac1N\sum_i(x_i-\mu)(x_i-\mu)^T$ (equivalently $\frac1N\sum x_ix_i^T-\mu\mu^T$ if using raw second moments). State explicitly that **PCA requires centering the data first** — this is a commonly-missed assumption.
- **Missing**: explicitly state that the resulting components are **orthonormal** and **ordered by decreasing eigenvalue = variance explained along that direction** — this directly answers the "characteristics of the PCA components" pillar.
- Minor: in the reconstruction-error objective, the sum should run over all $N$ samples, not $k$ (you wrote $\sum_{i=1}^{k}$ — $k$ is the target dimensionality, $N$ is the number of samples).

**Pillar coverage check** (from the prototype-questions list):

- Goal & applications ✅ (overfitting/noise/problematic directions — could add "visualization, whitening/preprocessing for other models")
- Formulation + training objective + solution ⚠️ (concept right, constraint and covariance definition need the fixes above; missing the centering step)
- Characteristics of components ❌ (not stated explicitly — add orthonormality + decreasing-variance ordering)
- Use in classification ✅ (correctly noted PCA is not a classifier on its own, needs a downstream model)

---

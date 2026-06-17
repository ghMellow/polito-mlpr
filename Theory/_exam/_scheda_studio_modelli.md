# Mnemonic Study Sheet — 6 Models + Connections

Only the bare minimum to keep "in your head". For full details → linked cards (not duplicated here).

---

## Summary table

| Model | Type | Uses labels? | Boundary / output | Training |
|---|---|---|---|---|
| **PCA** | Dimensionality reduction, unsupervised | No | linear projection, max variance | eigenvectors of $\Sigma$ |
| **LDA** | Dimensionality reduction + classification, supervised | Yes | linear (≤ $K-1$ directions) | generalized eigenvectors $S_W^{-1}S_B$ |
| **GGM/MVG** | Generative | Yes | quadratic (full) / linear (tied) | ML: $\mu_c,\Sigma_c$ per class |
| **Logistic Regression** | Discriminative, probabilistic | Yes | linear (unless feature expansion) | minimize cross-entropy (no closed form) |
| **SVM** | Discriminative, non-probabilistic | Yes | linear / non-linear (kernel) | QP primal/dual, no closed form |
| **GMM** | Generative, density estimation | depends (clustering = no, classification = yes) | arbitrary non-linear | EM |

---

## 1. PCA

- Maximizes the **variance** of the projected data: $\max \mathbf{w}^T\Sigma\mathbf{w}$ s.t. $\|\mathbf{w}\|=1$.
- Solution = **eigenvectors of $\Sigma$** ordered by decreasing eigenvalue.
- **Does not use labels** → no guarantee that maximum variance is discriminative.
- Requires **centering** the data. Often used as pre-processing (noise reduction / regularization for other models).

**Prototype questions to nail:**

- Goal and possible applications of the model (preprocessing, denoising, visualization — not just "reduces dimensionality").
- Formulation of the model, **including the training objective and its solution** (derive it, don't just state the result).
- Characteristics of the PCA components (ordering, orthogonality, variance explained).
- How the model can be employed in classification tasks (PCA alone does not classify — explain the bridge: projection + downstream classifier).

📄 Full card: [raw/01_PCA/pca.md](raw/01_PCA/pca.md)

---

## 2. LDA

- Maximizes the **Fisher ratio** $\dfrac{\mathbf{w}^TS_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}$ (between-class separation / within-class scatter).
- Solution = **generalized eigenvalue problem** $S_W^{-1}S_B\mathbf{w}=\lambda\mathbf{w}$.
- Binary case: closed form $\mathbf{w}\propto S_W^{-1}(\mu_2-\mu_1)$, **midpoint** threshold.
- At most $K-1$ useful directions; assumes Gaussians with **common covariance**.

**Prototype questions to nail:**

- Goals of the model and its formulation, and the training objective (Fisher ratio).
- Characteristics of the LDA discriminant directions.
- Model formulation, training objective and inference procedure for **binary** classification.
- The relationship between LDA and the Tied MVG classifier (same direction $\mathbf{w}$, different derivation and threshold).
- The form of the decision rule for the binary LDA classifier.
- As a **multiclass dimensionality-reduction technique**: the objective function and the limitations of the approach (scope = reduction, not classification accuracy).

📄 Full card: [raw/02_LDA/card.md](raw/02_LDA/card.md)

---

## 3. Generative Gaussian Models (GGM / MVG)

- Models $f(\mathbf{x}\mid C=c)\sim\mathcal{N}(\mu_c,\Sigma_c)$, then applies **Bayes** to get the posterior.
- ML estimation **per class**: $\mu_c^*$ and $\Sigma_c^*$ = empirical mean/covariance of the class.
- Binary decision = **LLR vs threshold** (log-odds of the priors).
- Variants: **Full** (quadratic boundary, QDA) · **Tied** (shared covariance → **linear** boundary, ≈ LDA) · **Naive Bayes** (diagonal covariance).

**Prototype questions to nail:**

- Model assumptions (Gaussian class-conditional densities, what each variant assumes about $\Sigma_c$).
- Estimation of the model parameters (ML → empirical mean/covariance per class).
- Inference procedure for both **binary and multiclass** problems.
- The form of the decision rule for binary problems (explicit $\mathbf{A},\mathbf{b},c$, quadratic vs linear).
- Naive Bayes and Tied Covariance **variants**: differences from the base (full) model, benefits and limitations of each.
- The relationship between Tied MVG and LDA.

📄 Full card: [raw/04_GGM/card.md](raw/04_GGM/card.md)

---

## 4. Logistic Regression (LR)

- Models the posterior **directly**: $P(C{=}1\mid x)=\sigma(\mathbf{w}^Tx+b)$.
- Training = minimizing **cross-entropy** (= ML on the labels), no closed form → numerical solver.
- Decision: $s=\mathbf{w}^Tx+b \gtrless 0$; for priors different from training, needs **recalibration** ($s_{llr}=s-\log\frac{n_T}{n_F}$).
- On separable classes the loss has no minimum ($\|w\|\to\infty$) → **regularization** is required.

**Prototype questions to nail:**

- Classification rule of the model and interpretation of the score (it's an estimate of the LLR / log-posterior-odds).
- Probabilistic interpretation of the model and of the classification score.
- Estimation of the parameters **and** possible interpretations of the training objective (ML ≡ minimizing cross-entropy ≡ empirical risk minimization — give all three readings, not just one).
- Compare the objective function with SVM's (log-loss vs hinge loss, both regularized risk).
- How the model can be **extended** for non-linear classification (feature expansion) and for score calibration (these "extensions" *are* the variants pillar for LR, even though the word "variant" never appears).

📄 Full card: [raw/06_LR/card.md](raw/06_LR/card.md)

---

## 5. Support Vector Machines (SVM)

- **Non-probabilistic**: maximizes the **margin** between classes.
- Primal (hard margin): $\min\frac12\|\mathbf{w}\|^2$ s.t. $z_i(\mathbf{w}^Tx_i+b)\ge1$ → convex QP.
- Soft margin: slack $\xi_i$, $C\sum\xi_i$ in the objective ↔ equivalent to the **hinge loss** $\max(0,1-z_is_i)$.
- Dual: $\max_\alpha \alpha^T\mathbf{1}-\frac12\alpha^TH\alpha$, $H_{ij}=z_iz_jx_i^Tx_j$, depends **only on dot products** → **kernel trick**.
- $\mathbf{w}^*=\sum_i\alpha_i^*z_ix_i$; only points on the margin have $\alpha_i\ne0$ → **support vectors** (complementary slackness/KKT).

**Prototype questions to nail:**

- Classification rule of SVM **and interpretation of the SVM score** (non-probabilistic, signed distance from the margin — not an LLR).
- The concept of margin.
- **Primal** (constrained convex QP form *and* hinge-loss form) **and dual** formulation of the objective, **and the explicit relationship between the primal and dual solutions** (these are 4 sub-answers, not 1).
- SVMs for **non-linear classification** (kernel trick, polynomial/RBF, Mercer's condition).
- Compare the objective function with LR's (hinge loss vs log-loss).

📄 Full card: [raw/09_SVM/card.md](raw/09_SVM/card.md)

---

## 6. Gaussian Mixture Models (GMM)

- Density = **weighted mixture of Gaussians**: $f(x)=\sum_c w_c\mathcal{N}(x;\mu_c,\Sigma_c)$.
- Training = **EM**: E-step computes responsibilities $\gamma_{c,i}$, M-step updates $\mu_c,\Sigma_c,w_c$. Likelihood is monotonic, but only reaches **local optima** (init via k-means/LBG).
- Classification: one GMM per class → LLR with mixture densities → **arbitrary non-linear** boundary.
- Variants: full / diagonal / tied (across components of the same class!). Risk: **unbounded** likelihood as $\Sigma_c\to0$.

**Prototype questions to nail:**

- Definition of the model, interpretation of the parameters, **and formulation of the GMM as a latent variable model** (two complementary "views" of the same object — both expected).
- Training procedure and estimation of the parameters (EM: E-step responsibilities, M-step updates; monotonic but only local optima).
- How the model is used for classification, for **both binary and multiclass** problems, **including open-set scenarios** (likelihood-threshold rejection — a detail unique to density-based models).
- The Naive Bayes assumption applied to GMM-based density models (explicit factorized/diagonal-covariance expression).
- Potential issues of GMMs and possible ways to address them (covariance regularization, eigenvalue constraints, EM initialization/LBG), and possible variants (full/diag/tied).
- The relationship between GMM and GGM (GGM = GMM with $K=1$ component per class).

📄 Full card: [raw/12_GMM/card.md](raw/12_GMM/card.md)

---

## Connections between models

### PCA ↔ LDA

- Both **linear** reductions, but PCA is unsupervised (max variance), LDA is supervised (max separability).
- Common pipeline **PCA → LDA** (PCA reduces noise/dimensionality before LDA).

📄 [03_PCA_vs_LDA.tex](03_PCA_vs_LDA.tex)

### LDA ↔ Tied MVG

- Same **direction** $\mathbf{w}$ (both assume common covariance), but derived differently: LDA from the Fisher ratio, Tied MVG from Bayes/ML.
- Different thresholds: LDA uses the midpoint (optimal under equal priors), Tied MVG uses the Bayes threshold based on log-priors.

📄 [raw/05_LDA_vs_TiedMVG/lda_vs_tiedmvg.md](raw/05_LDA_vs_TiedMVG/lda_vs_tiedmvg.md)

### LR ↔ GGM

- Discriminative (LR, no assumptions on the density) vs generative (GGM, assumes Gaussians).
- With tied covariances, GGM's boundary is **linear** like LR's — but the parameters are estimated differently (per-class ML vs cross-entropy minimization).

📄 [raw/07_LR_vs_GGM/lr_vs_ggm.md](raw/07_LR_vs_GGM/lr_vs_ggm.md)

### LR ↔ LDA

- Both produce **linear** boundaries, but with opposite estimation criteria: LR directly optimizes cross-entropy, LDA optimizes the Fisher ratio (not directly tied to classification error).

📄 [raw/08_LR_vs_LDA/lr_vs_lda.md](raw/08_LR_vs_LDA/lr_vs_lda.md)

### LR ↔ SVM

- Same **regularized risk** structure $\frac\lambda2\|w\|^2 + \frac1n\sum_i \text{loss}(z_is_i)$, differing only in the loss: **log-loss** (LR, always $>0$) vs **hinge loss** (SVM, $=0$ beyond the margin → sparsity/support vectors).
- LR gives calibratable posteriors, SVM does not (needs external calibration).

📄 [raw/10_LR_vs_SVM/lr_vs_svm.md](raw/10_LR_vs_SVM/lr_vs_svm.md)

### GGM ↔ SVM

- Generative/probabilistic (GGM, quadratic boundary) vs discriminative/non-probabilistic (SVM, max margin).
- Both can achieve non-linear boundaries: GGM via mixtures (→ GMM) or quadratic features, SVM via **kernels**.

📄 [raw/11_GGM_vs_SVM/ggm_vs_svm.md](raw/11_GGM_vs_SVM/ggm_vs_svm.md)

### GMM ↔ GGM

- **GGM is a special case of GMM** with $K=1$ component per class.
- GMM generalizes the boundary from quadratic (GGM) to **arbitrary**, by increasing $K_c$ (more components = more flexibility, but risk of overfitting/instability).

📄 [raw/13_GMM_vs_GGM/gmm_vs_ggm.md](raw/13_GMM_vs_GGM/gmm_vs_ggm.md)

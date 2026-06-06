# Cross-topic — GMM vs Generative Gaussian Models (MVG)

Cards used: `raw/04_GGM/card.md`, `raw/12_GMM/card.md`.

## 1. DIRECT CONNECTIONS
- **Special case / limit**: the MVG is exactly a **1-component GMM** ($K=1 \Rightarrow w_1=1$, $f_X=\mathcal{N}(\mu,\Sigma)$). Equivalently, a GMM is a *mixture* of MVG densities. Conversely, the marginal $f_X(x)=\sum_c\pi_c\mathcal{N}(x;\mu_c,\Sigma_c)$ of an MVG **classifier** over its classes is itself a GMM.
- **Same family**: both are **generative, probabilistic** models for the class-conditional density $f_{X|C}(x|c)$, used through **Bayes** for closed-set classification.
- **Same inference / decision rule**: identical generative machinery — binary LLR vs a log-odds threshold, multiclass $\arg\max_c[\log f_{X|C}(x|c)+\log P(C=c)]$. Only the form of $f_{X|C}$ changes (single Gaussian vs mixture).
- **Same per-component estimator**: the GMM M-step updates are the **soft (responsibility-weighted)** version of the MVG closed-form ML; with hard 0/1 responsibilities they collapse to per-class MVG ML.
- **Shared variant vocabulary**: both have **full / diagonal / tied** covariance options (with the caveat that "tied" and "diagonal" mean different things — see differences).

## 2. KEY DIFFERENCES
- **Latent variable**: GMM introduces a hidden **cluster** variable $C_i$ (categorical, $P=w_c$); MVG has no latent variable — the class label is observed in training.
- **Training**: MVG = **closed-form ML** in one shot; GMM = **iterative EM** (E-step responsibilities $\gamma_{c,i}$ + M-step updates), only **locally optimal**, init-dependent (K-means/LBG).
- **Likelihood**: MVG log-likelihood is well-posed and bounded; the GMM likelihood is **unbounded / ill-posed** ($\ge2$ components → degenerate $\Sigma_c\to0$), needs heuristics.
- **Decision boundary**: MVG → **linear** (tied) or **quadratic** (full QDA); GMM → **arbitrary non-linear** surface whose flexibility scales with the number of components $K_c$.
- **Supervision**: MVG ML is purely supervised (uses labels); the core GMM derivation is **unsupervised** ($\mathcal{D}$ unlabeled) — it can cluster and do density estimation, not only classify.
- **Model selection**: MVG has no $K$ to choose; GMM must pick the **number of components** — not by likelihood (always increases) but by **cross-validation**.
- **"Tied" / "diagonal" meaning**: in MVG, *tied* shares $\Sigma$ **across classes** and *diagonal* = Naive Bayes; in GMM, *tied* shares $\Sigma$ **across components of one class** and *diagonal* is **not** Naive Bayes.

## 3. COMPARISON TABLE

| Axis | Generative Gaussian (MVG) | Gaussian Mixture Model (GMM) |
|---|---|---|
| Objective | ML of a single Gaussian per class | ML of a mixture density (latent clusters) |
| Class-conditional | $\mathcal{N}(x;\mu_c,\Sigma_c)$ | $\sum_k w_{c,k}\mathcal{N}(x;\mu_{c,k},\Sigma_{c,k})$ |
| Assumptions | class data is **Gaussian** | data = weighted mix of Gaussians; cluster = latent |
| Training | **closed-form** ML, global optimum | **EM** (E/M steps), local optimum, init-dependent |
| Likelihood | bounded, well-posed | **unbounded/ill-posed**, degenerate risk |
| Decision rule | LLR / $\arg\max$ posterior; **linear (tied) or quadratic (full)** | LLR / $\arg\max$ posterior; **arbitrary non-linear** ($\uparrow$ with $K_c$) |
| Supervision | supervised (labels) | unsupervised core; also clustering / open-set |
| Model selection | none ($K=1$) | choose #components via **cross-validation** |
| Limitations | strong Gaussian assumption; full $\Sigma$ needs $\tfrac{D(D+1)}2$ params, $N>D$ | local optima, degeneracy, picks $K$, needs much data |

## 4. COMPARISON QUESTIONS
1. Explain why the MVG is a special case of a GMM and how the GMM EM updates relate to the MVG closed-form ML estimates.
2. Compare MVG and GMM as generative classifiers covering objective, training procedure and decision rule. Why is GMM training iterative while MVG is closed-form?
3. A single full-covariance MVG underfits a non-Gaussian class. How does a GMM address this, and what new problems (likelihood behaviour, model selection, initialization) does it introduce?

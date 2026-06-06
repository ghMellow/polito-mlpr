# Cross-topic comparison — LDA vs Tied-covariance MVG (Prompt 2)

Input cards: `raw/02_LDA/card.md`, `raw/04_GGM/card.md` (Tied variant).
Target exam question: **Theory question 2**.

---

## 1 — Direct connections

- **Same modeling assumption.** Both assume the within-class data are **Gaussian with a single shared (tied) covariance** $\boldsymbol\Sigma$. LDA's within-class scatter $S_W$ *is* (up to the $1/N$ normalization) the tied ML covariance estimate of the MVG.
- **Same discriminant direction.** Tied MVG binary score is linear with direction $\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)$; LDA binary direction is $S_W^{-1}(\boldsymbol\mu_2-\boldsymbol\mu_1)$. Since $\boldsymbol\Sigma\propto S_W$, **the two directions coincide**.
- **Same decision boundary (equal priors).** For tied $\boldsymbol\Sigma$ the quadratic term $A=-\frac12(\boldsymbol\Lambda_1-\boldsymbol\Lambda_0)$ **vanishes** → linear boundary. Setting $\mathrm{llr}=0$ (equal priors) gives
  $$(\boldsymbol\mu_1-\boldsymbol\mu_0)^T\boldsymbol\Sigma^{-1}\mathbf{x}=\tfrac12(\boldsymbol\mu_1-\boldsymbol\mu_0)^T\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1+\boldsymbol\mu_0)=\tfrac{m_1+m_0}{2},$$
  i.e. exactly LDA's midpoint threshold $t=\frac{m_1+m_2}{2}$. **Under equal priors the two classifiers are identical** (same direction *and* same threshold).
- **Same geometric reading.** Both reduce to a **nearest-centroid / Mahalanobis** classifier with shared covariance: assign to the class whose mean is closest in the $\boldsymbol\Sigma$-whitened space.

## 2 — Key differences

- **Derivation / objective.** LDA maximizes the **Fisher ratio** $s_B/s_W$ — purely geometric, *no probabilistic model*. Tied MVG maximizes the **per-class Gaussian likelihood** — a *generative probabilistic* model. They arrive at the same linear rule from different criteria ("same result obtained in a different way").
- **Threshold origin.** LDA's threshold is the **geometric midpoint** of the projected means, optimal *only* under equal priors. Tied MVG's threshold is the **log-odds** $-\log\frac{P(c_1)}{P(c_0)}$ → it natively handles **arbitrary priors and costs** (Bayes decision); LDA must shift the threshold ad hoc.
- **Multiclass behavior.** LDA generalizes as a **dimensionality-reduction** technique: at most $K-1$ discriminant directions, then a downstream classifier. Tied MVG generalizes as a **direct $K$-class** classifier via posteriors, **no $K-1$ limit**.
- **Output type.** LDA outputs a **projection/score** (uncalibrated). Tied MVG outputs **class-conditional log-likelihoods / LLR** usable directly in a Bayes decision framework.

## 3 — Comparison table

| Axis | LDA (binary) | Tied-covariance MVG (binary) |
|---|---|---|
| Objective | Maximize Fisher ratio $\frac{\mathbf{w}^TS_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}$ (geometric) | Maximize per-class Gaussian likelihood (generative) |
| Assumptions | Within-class Gaussian, equal covariance | Within-class Gaussian, **tied** covariance $\boldsymbol\Sigma$ |
| Training | Generalized eigenproblem $S_W^{-1}S_B\mathbf{w}=\lambda\mathbf{w}$ | ML: $\boldsymbol\mu_c$ = class means, $\boldsymbol\Sigma$ = pooled within-class covariance |
| Direction | $\mathbf{w}\propto S_W^{-1}(\boldsymbol\mu_2-\boldsymbol\mu_1)$ | $\propto\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)$ — **same** ($\boldsymbol\Sigma\propto S_W$) |
| Decision rule | $\mathbf{w}^T\mathbf{x}<t$, $t=\frac{m_1+m_2}{2}$ | $\mathrm{llr}(\mathbf{x})\gtrless-\log\frac{P(c_1)}{P(c_0)}$, **linear** ($A=0$) |
| Threshold / priors | Midpoint, equal priors only (shift manually) | Log-odds, native priors/costs |
| Multiclass | Dim. reduction, ≤ $K-1$ directions | Direct $K$-class posteriors, no limit |
| Output | Projection / score | Log-likelihoods / LLR (Bayes-ready) |

## 4 — Comparison questions (exam-style)

1. Considering LDA for binary classification and the Tied MVG binary classifier, detail model formulation, training objective and inference of each, the **relationship between the two**, and the **form of their decision rules**.
2. Show that, under equal priors, the Tied MVG binary decision boundary coincides with the LDA one. Where does the equivalence break (priors, multiclass)?
3. For multiclass problems explain how LDA is used as a dimensionality-reduction technique, its objective function and its limitations, contrasting with how Tied MVG handles the multiclass case.

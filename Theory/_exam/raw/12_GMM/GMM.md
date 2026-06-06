# Study Guide — Gaussian Mixture Models (GMM)

Source: `Theory/GaussianMixtureModels/GaussianMixtureModels.tex`. Exam-oriented breakdown along the 8 axes (motivation, formulation, training, inference, decision rule, assumptions, variants, limitations).

---

## 1. MAIN PILLARS

| # | Pillar | Exam axis |
|---|--------|-----------|
| P1 | What a GMM is and why we use it (density estimation, clustering, generative classifier) | Motivation / goal |
| P2 | Mathematical formulation: weighted sum of Gaussians, constraints, latent-variable / marginal view | Mathematical formulation |
| P3 | Likelihood and the ill-posed ML problem | Mathematical formulation + limitations |
| P4 | Responsibilities and hard-assignment training (and K-means as a special case) | Training |
| P5 | Soft assignments → EM algorithm (general derivation: ELBO, E-step, M-step) | Training |
| P6 | EM applied to GMM: closed-form M-step updates with statistics $N_c, F_c, S_c$ | Training |
| P7 | Inference: GMM as class-conditional density (closed-set) and open-set classification | Inference + decision rule |
| P8 | Variants: diagonal, tied, number of components, LBG/K-means initialization | Variants |
| P9 | Limitations: ill-posed likelihood, degenerate models, model selection, local optima | Limitations |

---

## 2. DEEP-DIVE PER PILLAR

### P1 — Motivation / goal
- **Essential**: a GMM models a generic distribution as a **weighted combination of Gaussians**; it can approximate *any sufficiently regular distribution* to a desired degree given enough data.
- Three uses: **density estimation**, **clustering** (generalizes K-means), **generative classifier** (model each class density with a GMM).
- **Logical move**: start from the Gaussian classifier — the *marginal* $f_X(x)=\sum_c \pi_c \mathcal{N}(x;\mu_c,\Sigma_c)$ over classes is *already* a GMM. Generalize $\pi_c \to w_c$.
- **Keywords**: density estimation, generative model, mixture component, unsupervised.
- **Easy to miss**: GMM is *not* tied to classification — $\mathcal{D}$ is treated as **unlabeled** in the general derivation.

### P2 — Mathematical formulation
- **Essential formula**:
  $$f_X(x)=\sum_{c=1}^{K} w_c\,\mathcal{N}(x;\mu_c,\Sigma_c), \qquad \sum_{c=1}^K w_c = 1,\; w_c\ge 0$$
- Parameters $\theta=(\mathcal{M},\mathcal{S},\mathbf{w})$ with $\mathcal{M}=\{\mu_c\}$, $\mathcal{S}=\{\Sigma_c\}$, $\mathbf{w}=\{w_c\}$.
- **Latent-variable view**: GMM is the **marginal** of a joint over (sample, cluster):
  $$f_{X_i}(x_i)=\sum_{c=1}^K f_{X_i|C_i}(x_i|c)P(C_i=c)=\sum_c w_c\mathcal{N}(x_i;\mu_c,\Sigma_c)$$
  with **categorical prior** $P(C_i=c)=w_c$ and **conditional** $f_{X_i|C_i}(x_i|c)=\mathcal{N}(x_i;\mu_c,\Sigma_c)$.
- **Logical move**: the constraint $\sum w_c=1$ comes from requiring $\int f_X = 1$.
- **Keywords**: weights, components, latent/hidden variable, cluster membership.

### P3 — Likelihood and ill-posed ML
- Likelihood / log-likelihood:
  $$\ell(\theta)=\sum_{i=1}^N \log \sum_{c=1}^K w_c\,\mathcal{N}(x_i;\mu_c,\Sigma_c)$$
- **The log of a sum** is what makes direct maximization hard (no closed form).
- **Ill-posed**: with $\ge 2$ components the likelihood is **unbounded above** (a component can collapse onto a single point, $\Sigma_c\to 0$, $\ell\to\infty$) → degenerate solutions. ML + heuristics still gives good density estimates.
- **Easy to miss**: the "log of sum" obstacle is exactly the reason EM (which optimizes the joint log-likelihood, a "sum of logs") is introduced.

### P4 — Responsibilities & hard assignment
- **Responsibility** (cluster posterior):
  $$\gamma_{c,i}=P(C_i=c|X_i=x_i)=\frac{w_c\mathcal{N}(x_i;\mu_c,\Sigma_c)}{\sum_{c'}w_{c'}\mathcal{N}(x_i;\mu_{c'},\Sigma_{c'})}$$
- **Hard assignment**: $\hat c_i=\arg\max_c \gamma_{c,i}$, then ML per cluster as if labels were known:
  $$\hat\mu_c=\tfrac1{N_c}\sum_{i:\hat c_i=c}x_i,\quad \hat\Sigma_c=\tfrac1{N_c}\sum_{i:\hat c_i=c}(x_i-\hat\mu_c)(x_i-\hat\mu_c)^T,\quad \hat w_c=\tfrac{N_c}{\sum_{c'}N_{c'}}$$
- **Special case — K-means**: fix $\Sigma_c=I$ and $w_c=1/K$ ⇒ assignment becomes $\hat c_i=\arg\min_c\|x_i-\mu_c\|^2$ → **K-means**.
- **Problem with hard clustering**: ignores uncertainty when clusters overlap; does **not** maximize the observed-data likelihood.

### P5 — EM (general derivation)
- EM handles ML for likelihoods expressible as marginals $f_X(x)=\int f_{X,H}(x,h)\,dh$, $H$ = latent variable.
- **ELBO decomposition** (with a distribution $Q(h)$):
  $$\log f_X(x;\theta)=\mathcal{L}_h(Q,\theta)+D_{KL}(Q\,\|\,f_{H|X}) ,\qquad D_{KL}\ge 0 \Rightarrow \mathcal{L}_h \le \log f_X$$
  so $\mathcal{L}_h$ is a **lower bound** on the log-likelihood.
- **E-step**: maximize bound w.r.t. $Q$ ⇒ choose $Q^t(h)=f_{H|X}(h|x;\theta^t)$ (drives $D_{KL}\to 0$, makes bound *tight*).
- **M-step**: maximize bound w.r.t. $\theta$:
  $$\theta^{t+1}=\arg\max_\theta \mathbb{E}_{Q^t(h)}[\log f_{X,H}(x,h;\theta)] \quad(\text{auxiliary function }\mathcal{Q}(\theta,\theta^t))$$
- **Monotonicity**: $\log f_X(x;\theta^{t+1})\ge \log f_X(x;\theta^t)$ — likelihood never decreases.
- Converges to a **stationary point** (saddle/local max) of $\ell(\theta)$, depending on initialization.
- **Logical move**: the trick is that $Q^t$ is *frozen* at $\theta^t$, so the M-step expectation no longer involves $\theta$ inside $Q$.

### P6 — EM for GMM (closed-form M-step)
- Latent variables $h=\{C_1,\dots,C_N\}$, joint $f_{X_i,C_i}(x_i,c)=w_c\mathcal{N}(x_i;\mu_c,\Sigma_c)$.
- **E-step**: compute $\gamma_{c,i}=P(C_i=c|x_i;\theta^t)$. Auxiliary function:
  $$\mathcal{Q}(\theta,\theta^t)=\sum_{i=1}^N\sum_{c=1}^K \gamma_{c,i}\big[\log\mathcal{N}(x_i;\mu_c,\Sigma_c)+\log w_c\big]$$
- **M-step** (subject to $\sum_k w_k=1$):
  $$\mu_c^*=\frac{\sum_i\gamma_{c,i}x_i}{\sum_i\gamma_{c,i}},\quad \Sigma_c^*=\frac{\sum_i\gamma_{c,i}(x_i-\mu_c^*)(x_i-\mu_c^*)^T}{\sum_i\gamma_{c,i}},\quad w_c^*=\frac{\sum_i\gamma_{c,i}}{\sum_{i,c'}\gamma_{c',i}}$$
- **Statistics**: zero / first / second order:
  $$N_c=\sum_i\gamma_{c,i},\quad F_c=\sum_i\gamma_{c,i}x_i,\quad S_c=\sum_i\gamma_{c,i}x_ix_i^T$$
  giving $\mu_c^*=F_c/N_c$, $\Sigma_c^*=S_c/N_c-\mu_c^*\mu_c^{*T}$, $w_c^*=N_c/N$.
- **Easy to miss**: M-step updates are **soft** versions of the hard-assignment ML (weighted by $\gamma_{c,i}$ instead of 0/1 membership).

### P7 — Inference (classification)
- **Closed-set classification**: fit one GMM per class, $X|C=c\sim GMM(\mathcal{M}_c,\mathcal{S}_c,\mathbf{w}_c)$ with $K_c$ components per class:
  $$f_{X_t|C_t}(x_t|c)=\sum_{k=1}^{K_c}w_{c,k}\mathcal{N}(x_t;\mu_{c,k},\Sigma_{c,k})$$
  $$P(C_t=c|x_t)\propto P(C_t=c)\sum_{k=1}^{K_c}w_{c,k}\mathcal{N}(x_t;\mu_{c,k},\Sigma_{c,k})$$
- Decision rule = standard generative Bayes: $\arg\max$ posterior (multiclass) / LLR vs threshold (binary). The class-conditional density is just **richer** than a single Gaussian.
- **Open-set classification**: use a GMM to model the heterogeneous **"none-of-the-others"** class from unlabeled data; GMM finds homogeneous sub-clusters. Known classes can stay MVG. Scores are often **not calibrated** → may need post-processing.
- **Clustering**: assign each point to $\arg\max_c\gamma_{c,i}$ (soft membership available).

### P8 — Variants
- **Diagonal covariance**: fewer parameters, less overfitting/cost; may need more components. **NOT the Naive Bayes assumption** — Naive Bayes would mean a *separate GMM per independent feature subset*.
- **Tied GMM**: all components of *one class's* GMM share one $\Sigma$. ≠ Tied MVG (which shares $\Sigma$ *across classes*).
- **Initialization**: matters a lot (EM only finds local optima).
  - **K-means** initializer (= hard-assignment + isotropic covariance limit).
  - **LBG**: split each component $\mu_c^\pm=\mu_c\pm\varepsilon$ (good $\varepsilon$ = displacement along principal eigenvector of $\Sigma_c$) → run EM on $2G$ components → iterate to desired $K$.

### P9 — Limitations
- **Unbounded likelihood** with $\ge 2$ components ⇒ degenerate models, numerical issues (need heuristics / constrained covariances).
- **Local optima**: EM only reaches a stationary point; depends on init → run multiple restarts.
- **Model selection**: more components ⇒ higher likelihood always, so can't pick $K$ by likelihood; use **cross-validation**.
- Needs **enough data** for reliable density estimates.

---

## 3. DEPENDENCY MAP

P1 (why) → P2 (formulation + latent view) → P3 (likelihood, why ML is hard) → P4 (responsibilities + hard assignment + K-means) → P5 (EM general: ELBO/E/M) → P6 (EM for GMM closed-form) → P7 (classification use) → P8 (variants/init) → P9 (limitations).

Master before moving on: P2 (latent-variable marginal) is the prerequisite for P5; P4 (responsibilities) is reused verbatim in P6 E-step. Do not attempt P5/P6 before the marginal/joint distinction in P2–P3 is solid.

---

## 4. VERIFICATION QUESTIONS

- **P1/P2**: Define a Gaussian Mixture Model. What are its parameters and what constraint must the weights satisfy and why?
- **P2/P3**: Write the GMM log-likelihood and explain why direct ML maximization is difficult / ill-posed.
- **P4**: Define the responsibilities $\gamma_{c,i}$. Show how, under isotropic covariance and uniform weights, hard-assignment GMM training reduces to K-means.
- **P5**: Derive the EM lower bound (ELBO) and state the E-step and M-step. Why is the likelihood guaranteed not to decrease?
- **P6**: Give the EM update equations for a GMM (E-step responsibilities and M-step $\mu_c,\Sigma_c,w_c$) in terms of the statistics $N_c,F_c,S_c$.
- **P7**: How is a GMM used as a classifier (closed-set)? How can it help in open-set classification?
- **P8**: Explain the difference between a diagonal GMM and the Naive Bayes assumption; between a tied GMM and a tied-covariance MVG. Describe the LBG algorithm.
- **P9**: Why can't we choose the number of components by maximizing the likelihood? What is a degenerate model?

---

## 5. FINAL SUMMARY CARD

See `card.md` (indexed by exam axis, same vocabulary as the other topics for cross-model comparison).

# Cross-topic — Logistic Regression vs Support Vector Machine

Card di input: [[06_LR]] e [[09_SVM]]. Pairing canonico d'esame (entrambi *risk minimization*).

## 1. DIRECT CONNECTIONS
- **Stessa famiglia (linear discriminative).** Entrambi cercano un iperpiano $s=\mathbf{w}^T\mathbf{x}+b$ e classificano col **segno** $s\gtrless0$; nessuno modella la densità $f_X(\mathbf{x})$.
- **Stessa struttura objective = regularized risk minimization.** $\frac\lambda2\|\mathbf{w}\|^2+\frac1n\sum_i\ell(z_is_i)$: differiscono **solo nella loss per-sample**.
  - LR: log/softplus $\ell(z_is)=\log(1+e^{-z_is})$.
  - SVM: hinge $\ell(z_is)=\max(0,1-z_is)$.
- **Stesso ruolo della regolarizzazione.** Su dati separabili LR ha loss senza minimo ($\|\mathbf{w}\|\to\infty$): il termine $\|\mathbf{w}\|^2$ la stabilizza. In SVM lo **stesso** $\frac12\|\mathbf{w}\|^2$ ha interpretazione **geometrica** = massimizzare il margine. SVM è la "lettura geometrica" della regolarizzazione di LR.
- **Stessa via alla non-linearità per dot-product.** Entrambi ottengono boundary non-lineari mappando $\mathbf{x}\to\phi(\mathbf{x})$; i kernel non sono esclusivi di SVM (LR kernelizzabile).
- **Stesso problema di prior/bilanciamento.** Nessuno dei due ha score legato al prior dell'applicazione: LR → recalibrazione $s_{llr}=s-\log\frac{n_T}{n_F}$ o prior-weighted; SVM → costi per-classe $C_T,C_F$.

## 2. KEY DIFFERENCES
- **Loss e sparsità.** Hinge è **esattamente 0** oltre il margine ⇒ solo i **support vector** ($\alpha_i\ne0$) definiscono la soluzione; la log-loss è **sempre >0** ⇒ tutti i punti contribuiscono.
- **Output probabilistico.** LR dà un **posterior** $\sigma(s)$ e $s$ è un log-posterior-ratio interpretabile/calibrato; lo score SVM **non ha interpretazione probabilistica** → serve calibrazione a posteriori.
- **Come si risolve.** LR: minimizzazione numerica diretta del primal (loss + gradiente). SVM: **QP convesso**, tipicamente via **duale** (Lagrangiana + KKT), che apre al kernel e dà i support vector.
- **Cosa "guida" la soluzione.** LR pesa tutti i margini in modo morbido; SVM punta a **massimizzare il margine** ignorando i punti ben classificati.
- **Multiclasse.** LR → **softmax** naturale; SVM → nativamente binario, multiclasse **difficile** (OvO/OvA).

## 3. COMPARISON TABLE

| Asse | Logistic Regression | Support Vector Machine |
|---|---|---|
| Objective | $\frac\lambda2\|\mathbf{w}\|^2+\frac1n\sum\log(1+e^{-z_is_i})$ | $\frac\lambda2\|\mathbf{w}\|^2+\frac1n\sum\max(0,1-z_is_i)$ |
| Loss per-sample | log / softplus (sempre $>0$) | hinge (0 oltre il margine) |
| Assumptions | log-odds lineare; nessun modello di $f_X$ | margine massimo; nessun modello di $f_X$ |
| Training | ML / cross-entropy, solver numerico | QP convesso, duale + KKT |
| Inference / score | $s=\mathbf{w}^T\mathbf{x}+b$, posterior $\sigma(s)$ | $s=\mathbf{w}^T\mathbf{x}+b$ o $\sum_{SV}\alpha_iz_ik(\mathbf{x}_i,\mathbf{x})+b$; **no posterior** |
| Decision rule | $s\gtrless0$; soglia bayesiana dopo recalibrazione | $s\gtrless0$; nessuna soglia bayesiana nativa |
| Non-linearità | feature expansion $\phi(\mathbf{x})$ (esplicita) | kernel $k$ (implicita, anche $\infty$-dim) |
| Sparsità | no (tutti i punti) | sì (solo support vector) |
| Multiclasse | softmax | difficile (binario nativo) |
| Limitazioni | score legato al prior empirico; boundary lineare salvo $\phi$ | niente probabilità; non invariante affine; CV per $C$/kernel |

## 4. COMPARISON QUESTIONS
- «Describe and compare Logistic Regression and SVM as risk-minimization methods: write both objectives, compare the per-sample losses and explain the role of the regularization term.»
- «Both LR and SVM produce a linear score $s=\mathbf{w}^T\mathbf{x}+b$. What is the difference in the meaning of $s$ and in how non-linear boundaries are obtained?»
- «Why does SVM depend only on the support vectors while LR depends on all training points? Relate this to the two loss functions and to complementary slackness.»

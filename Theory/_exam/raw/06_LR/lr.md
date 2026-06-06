# Study Guide — Logistic Regression (LR)

*(Prompt 1 applicato a `Theory/LogisticRegression/LogisticRegression.tex`. Target d'esame: Theory question 4.)*

---

## 1 — Main pillars

1. **Discriminative model of the posterior.** LR modella *direttamente* $P(C|X)$ (non le densità class-conditional), assumendo che il log-odds sia lineare → posterior = sigmoide di uno score lineare.
2. **Classification rule = linear hyperplane.** Lo score $s=\mathbf{w}^T\mathbf{x}+b$ è il log-posterior-ratio; la regola è $s\gtrless 0$, un iperpiano ortogonale a $\mathbf{w}$.
3. **Training = ML ⇔ min cross-entropy ⇔ empirical risk minimization.** Le tre letture dello stesso obiettivo, niente soluzione in forma chiusa → solver numerico con gradiente.
4. **Regolarizzazione.** Necessaria perché su dati linearmente separabili la loss non ha minimo (infimo a $\|w\|\to\infty$). Penalità $\frac{\lambda}{2}\|w\|^2$.
5. **Score, priori e non-linearità.** Lo score riflette il prior empirico del training → recalibrazione a LLR; prior-weighted LR per prior noto; feature expansion $\phi(x)$ per boundary non-lineari; softmax per il multiclasse.

---

## 2 — Deep-dive per pillar

### P1 — Discriminative posterior
- **Formula chiave.** Assumendo $P(C{=}h_1|x)=e^{\mathbf{w}^Tx+b}\,P(C{=}h_0|x)$ e normalizzando:
  $$P(C{=}h_1|x,\mathbf{w},b)=\sigma(\mathbf{w}^Tx+b),\qquad \sigma(t)=\frac{1}{1+e^{-t}}.$$
- **Logical move.** Discriminativo: non serve modellare $f_X(x)$ — nella likelihood congiunta $f_{X,C}=P(C|X,\theta)f_X(x)$ il termine $f_X$ non dipende da $\theta$ e si scarta.
- **Keyword.** sigmoide/logistica, log-odds lineare, $\theta=(\mathbf{w},b)$.
- **Dettagli dimenticabili.** $\sigma'(t)=\sigma(t)(1-\sigma(t))$; $1-\sigma(t)=\sigma(-t)$.

### P2 — Classification rule e interpretazione dello score
- **Formula.** $\log\frac{P(C{=}1|x)}{P(C{=}0|x)}=\mathbf{w}^Tx+b=s$. Decisione: $s\gtrless 0$.
- **Logical move.** $s$ è (proporzionale alla) distanza con segno dall'iperpiano; $|s|$ grande = predizione "confidente".
- **Keyword.** log-posterior-ratio, iperpiano ortogonale a $\mathbf{w}$, decision boundary lineare.

### P3 — Training (le tre letture)
- **ML.** $\hat\theta=\arg\max\sum_i\log P(c_i|x_i,\theta)$; con label Bernoulli $y_i=\sigma(s_i)$:
  $$\ell(\mathbf{w},b)=\sum_i[c_i\log y_i+(1-c_i)\log(1-y_i)].$$
- **Cross-entropy.** Minimizzare $J=-\ell=\sum_i H(c_i,y_i)$ = avvicinare $\mathrm{Ber}(y_i)$ alla distribuzione empirica $\mathrm{Ber}(c_i)$; $H(P,Q)=E_P[-\log Q]$, minima per $Q=P$.
- **Empirical risk.** Con $z_i=2c_i-1\in\{-1,+1\}$ e $s_i=\mathbf{w}^Tx_i+b$:
  $$J(\mathbf{w},b)=\sum_i\log\!\big(1+e^{-z_i s_i}\big)=\sum_i\ell(-z_i s_i),$$
  loss per-campione $\ell$ = costo: piccolo (asintoticamente→0) se predizione concorde, cresce ~linearmente in $|s|$ se discorde. → **risk minimization** $R(\theta)=\sum_i\ell(\theta,x_i,z_i)$.
- **Dettaglio.** Niente forma chiusa → solver numerico che usa loss + gradiente.

### P4 — Regolarizzazione
- **Problema.** Classi linearmente separabili ⇒ $\inf J=0$ a $\|w\|\to\infty$, parametri non convergono.
- **Fix.** $R(\mathbf{w},b)=\frac{\lambda}{2}\|w\|^2+\frac1n\sum_i\log(1+e^{-z_i s_i})$. $\lambda$ iper-parametro (cross-validation, *non* minimizzabile rispetto a $\lambda$ → darebbe 0).
- **Trade-off.** $\lambda$ grande → norma piccola ma scarsa separazione; piccolo → overfit. Il modello regolarizzato **non** è invariante a trasformazioni lineari → utile pre-processing (centering, standardizzazione, whitening, length-norm).

### P5 — Score/priori, prior-weighting, multiclasse, non-linearità
- **Recalibrazione a LLR.** Lo score riflette il prior empirico; per usarlo come LLR: $s_{llr}=\mathbf{w}^Tx+b-\log\frac{n_T}{n_F}$, poi decidi $s_{llr}\gtrless\log\frac{\pi_T}{1-\pi_T}$.
- **Prior-weighted LR.** Se $\pi_T$ è noto a priori, pesa le due classi: $R(\mathbf{w})=\frac\lambda2\|w\|^2+\frac{\pi_T}{n_T}\sum_{z_i=1}\ell+\frac{1-\pi_T}{n_F}\sum_{z_i=-1}\ell$. La LR standard = prior-weighted col prior empirico.
- **Multiclasse (softmax).** $P(C{=}k|x)=\dfrac{e^{\mathbf{w}_k^Tx+b_k}}{\sum_j e^{\mathbf{w}_j^Tx+b_j}}$; obiettivo = min cross-entropy multiclasse $-\sum_i\sum_k z_{ik}\log y_{ik}$ (softmax loss). Over-parametrizzato (somma costante a tutti i $w_j$ invariante).
- **Non-linearità.** Feature expansion $\phi(x)$: LR lineare in $\phi(x)$ ⇒ boundary non-lineare (es. quadratico) in $x$. Costo: dimensione di $\phi$ cresce in fretta → costo + overfit.

---

## 3 — Dependency map

sigmoide/log-odds lineare → posterior $P(C|x)=\sigma(s)$ → classification rule $s\gtrless0$
→ ML su label Bernoulli → cross-entropy → (riscrittura con $z_i$) empirical risk minimization
→ separabilità ⇒ regolarizzazione → (score riflette prior) recalibrazione LLR / prior-weighting
→ softmax (multiclasse) · feature expansion (non-linearità).

## 4 — Verification questions

1. Perché LR non deve modellare $f_X(x)$? (discriminativo: $f_X$ indipendente da $\theta$).
2. Qual è l'interpretazione probabilistica dello score $s$? (log-posterior-ratio).
3. Mostra l'equivalenza ML ⇔ min cross-entropy ⇔ empirical risk.
4. Perché serve la regolarizzazione e cosa succede senza, su dati separabili?
5. Come si recupera un LLR dallo score? Come si tratta un prior diverso dall'empirico?
6. Come si ottengono decision function non-lineari?

## 5 — Final summary card → vedi `card.md`

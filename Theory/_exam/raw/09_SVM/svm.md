# Study Guide — Support Vector Machines (SVM)

Fonte: `Theory/SupportVectorMachines/SupportVectorMachines.tex`. Argomento d'esame, classificatore binario di tipo *risk minimization* con interpretazione geometrica del termine di regolarizzazione.

---

## 1. MAIN PILLARS (mappati agli assi d'esame)

| # | Pillar | Asse d'esame |
|---|---|---|
| P1 | Motivazione: dare interpretazione geometrica alla regolarizzazione, separazione non-lineare senza espansione esplicita, niente posterior | Motivation / goal |
| P2 | Maximum-margin hyperplane: definizione di margine e criterio di ottimalità | Mathematical formulation |
| P3 | Riformulazione canonica → primal QP convesso | Mathematical formulation |
| P4 | Lagrangiana, problema duale, KKT, support vectors | Training |
| P5 | Soft margin (slack $\xi$, costante $C$) per classi non separabili | Variants / Training |
| P6 | Hinge loss e confronto con Logistic Regression (risk minimization) | Mathematical formulation / Variants |
| P7 | Inferenza: score primal vs dual, dipendenza dai soli dot-product | Inference / Decision rule |
| P8 | Kernel trick, condizione di Mercer, kernel notevoli | Variants (non-linearità) |
| P9 | Considerazioni pratiche: $C$/iperparametri, invarianza, bilanciamento classi, niente probabilità, multiclasse | Limitations / Variants |

---

## 2. DEEP-DIVE PER PILLAR

### P1 — Motivazione e obiettivo (Motivation / goal)
- **Frasi chiave:** SVM dà *interpretazione geometrica del termine di regolarizzazione* di LR; permette separazione non-lineare **senza espansione esplicita** delle feature (kernel); **l'output NON è un posterior** $P(C\mid X)$.
- **Logical move:** problema binario, classi linearmente separabili ⇒ esistono **infiniti** iperpiani separatori → quale scegliere?
- **Keyword:** generalized risk minimization, maximum margin, no probabilistic output.
- **Facile da dimenticare:** LR sceglie l'iperpiano che massimizza i posterior, ma può ridurre la loss aumentando $\|\mathbf{w}\|$ a parità di orientamento (posterior → 0/1): da qui la necessità di un criterio diverso (il margine).

### P2 — Maximum-margin hyperplane (Mathematical formulation)
- **Formule:** $f(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b$; encoding $z_i\in\{+1,-1\}$; distanza $d(\mathbf{x}_i)=\frac{|f(\mathbf{x}_i)|}{\|\mathbf{w}\|}$.
- **Criterio (max-min):** $\mathbf{w}^*,b^*=\arg\max_{\mathbf{w},b}\min_i \frac{|z_i(\mathbf{w}^T\mathbf{x}_i+b)|}{\|\mathbf{w}\|}$.
- **Logical move:** se le classi sono separabili allora esiste soluzione con $z_i(\mathbf{w}^T\mathbf{x}_i+b)>0\ \forall i$ → posso togliere il valore assoluto e scrivere $\arg\max_{\mathbf{w},b}\frac{1}{\|\mathbf{w}\|}\min_i[z_i(\mathbf{w}^T\mathbf{x}_i+b)]$.
- **Keyword:** margine = distanza del punto più vicino; functional vs geometric margin.

### P3 — Forma canonica → primal QP (Mathematical formulation)
- **Logical move 1 (invarianza di scala):** l'obiettivo è invariante a $(\mathbf{w},b)\to(\alpha\mathbf{w},\alpha b)$ → classe di equivalenza; fisso il rappresentante con $\min_i z_i(\mathbf{w}^T\mathbf{x}_i+b)=1$.
- **Conseguenza:** vincoli $z_i(\mathbf{w}^T\mathbf{x}_i+b)\ge 1\ \forall i$.
- **Primal:** $\arg\min_{\mathbf{w},b}\frac12\|\mathbf{w}\|^2\ \text{s.t.}\ z_i(\mathbf{w}^T\mathbf{x}_i+b)\ge 1$.
- **Logical move 2:** all'ottimo il vincolo è attivo ($\min_i=1$): se fosse $\psi>1$ potrei riscalare per $1/\psi$ riducendo la norma → contraddizione.
- **Keyword:** convex quadratic programming (obiettivo convesso, vincoli = insieme convesso) ⇒ ogni minimo locale è globale.

### P4 — Lagrangiana, duale, KKT, support vectors (Training)
- **Lagrangiana (hard):** $L=\frac12\|\mathbf{w}\|^2-\sum_i\alpha_i[z_i(\mathbf{w}^T\mathbf{x}_i+b)-1]$, $\alpha_i\ge0$.
- **Stazionarietà:** $\mathbf{w}=\sum_i\alpha_i z_i\mathbf{x}_i$ e $\sum_i\alpha_i z_i=0$.
- **Duale:** $\max_{\boldsymbol\alpha}\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j z_i z_j\mathbf{x}_i^T\mathbf{x}_j=\boldsymbol\alpha^T\mathbf{1}-\frac12\boldsymbol\alpha^T\mathbf{H}\boldsymbol\alpha$, $H_{ij}=z_iz_j\mathbf{x}_i^T\mathbf{x}_j$; s.t. $\alpha_i\ge0,\ \sum_i\alpha_iz_i=0$.
- **Logical move chiave:** $\mathbf{H}$ dipende dai dati **solo via dot-product** $\mathbf{x}_i^T\mathbf{x}_j$ → apre al kernel.
- **Dualità debole/forte:** $L_D(\boldsymbol\alpha)\le L_P(\mathbf{w},b)$; all'ottimo **duality gap = 0**, $L_D(\boldsymbol\alpha^*)=L_P(\mathbf{w}^*,b^*)$.
- **KKT (le 5):** stazionarietà in $\mathbf{w}$, in $b$; feasibility primale; $\alpha_i\ge0$; **complementary slackness** $\alpha_i[z_i(\mathbf{w}^T\mathbf{x}_i+b)-1]=0$.
- **Support vectors:** se $z_i(\mathbf{w}^T\mathbf{x}_i+b)>1$ (fuori dal margine) ⇒ $\alpha_i=0$; $\alpha_i\ne0$ ⇒ punto **sul margine** = support vector. I non-SV **non influenzano** la superficie. $b$ si ricava dalle KKT dopo aver ottenuto $\boldsymbol\alpha$.

### P5 — Soft margin (Variants / Training)
- **Problema:** classi non separabili ⇒ qualunque $\mathbf{w}$ viola dei vincoli.
- **Slack:** $\xi_i\ge0$, vincolo rilassato $z_i(\mathbf{w}^T\mathbf{x}_i+b)\ge 1-\xi_i$.
- **Funzionale generale:** $\frac12\|\mathbf{w}\|^2+C\,F(\sum_i\xi_i^\sigma)$. Per $\sigma$ piccolo $\sum\xi_i^\sigma$ ≈ numero di punti dentro il margine (problema difficile) → si semplifica con $\sigma=1$, $F(u)=u$.
- **Obiettivo finale:** $\min_{\mathbf{w},b,\boldsymbol\xi}\frac12\|\mathbf{w}\|^2+C\sum_i\xi_i$ s.t. $z_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1-\xi_i,\ \xi_i\ge0$ (ancora QP convesso).
- **Interpretazione $\xi$:** non contano gli errori ma sono una **penalità**; punti dentro il margine $\xi_i>0$, mis-classificati $\xi_i>1$; $\sum_i\xi_i$ = **upper bound** sul numero di errori. $C$ = trade-off margine vs errori. Soluzione = *soft margin hyperplane*.
- **Duale soft margin:** identico a hard ma con **box constraint** $0\le\alpha_i\le C$ (oltre a $\sum_i\alpha_iz_i=0$). Compare il moltiplicatore $\mu_i\ge0$ per $\xi_i\ge0$, con KKT $C-\alpha_i-\mu_i=0$ e $\mu_i\xi_i=0$.

### P6 — Hinge loss e confronto con LR (Mathematical formulation / Variants)
- **Eliminazione dei vincoli:** sui punti corretti $\xi_i=0$, altrimenti $\xi_i=1-z_i(\mathbf{w}^T\mathbf{x}_i+b)$ → primal non vincolato $\min_{\mathbf{w},b}\frac12\|\mathbf{w}\|^2+C\sum_i\max[0,1-z_i(\mathbf{w}^T\mathbf{x}_i+b)]$.
- **Hinge loss:** $f(s)=\max(0,1-s)$.
- **Confronto objective (stessa forma regolarizzata):**
  - SVM: $\frac\lambda2\|\mathbf{w}\|^2+\frac1n\sum_i\max[0,1-z_i(\mathbf{w}^T\mathbf{x}_i+b)]$ (hinge).
  - LR: $\frac\lambda2\|\mathbf{w}\|^2+\frac1n\sum_i\log[1+e^{-z_i(\mathbf{w}^T\mathbf{x}_i+b)}]$ (log/softplus).
- **Logical move:** entrambi sono **regularized risk minimization** $\Phi(\mathbf{w})+\frac1n\sum_i\ell$; differiscono **solo nella loss per-sample**. La hinge è esattamente 0 sui punti ben classificati oltre il margine (⇒ sparsità / support vectors); la log-loss è sempre $>0$.

### P7 — Inferenza e score (Inference / Decision rule)
- **Score primal:** $s(\mathbf{x}_t)=\mathbf{w}^T\mathbf{x}_t+b$, complessità $O(D)$.
- **Score dual:** $s(\mathbf{x}_t)=\sum_i\alpha_iz_i\mathbf{x}_i^T\mathbf{x}_t+b=\sum_{i:\alpha_i>0}\alpha_iz_i\mathbf{x}_i^T\mathbf{x}_t+b$, complessità $O(\#SV)$.
- **Decision rule (binaria):** segno dello score, $s(\mathbf{x}_t)\gtrless 0$ → iperpiano $\perp\mathbf{w}$. **Niente soglia bayesiana** in senso proprio: lo score non è un LLR.
- **Multiclasse:** SVM è nativamente binario; estensione difficile (one-vs-one / one-vs-all, non parte della formulazione).
- **Facile da dimenticare:** lo score dipende **solo dai dot-product** $\mathbf{x}_i^T\mathbf{x}_t$ → kernelizzabile anche in inferenza.

### P8 — Kernel trick e Mercer (Variants — non-linearità)
- **Idea:** mapping $\boldsymbol\Phi:\mathbb{R}^D\to\mathcal{H}$; nel duale e nello score servono **solo** $\boldsymbol\Phi(\mathbf{x}_i)^T\boldsymbol\Phi(\mathbf{x}_j)$ → li calcolo con un **kernel** $k(\mathbf{x}_1,\mathbf{x}_2)=\boldsymbol\Phi(\mathbf{x}_1)^T\boldsymbol\Phi(\mathbf{x}_2)$ senza costruire $\boldsymbol\Phi$.
- **Matrici:** $H_{ij}=z_iz_j k(\mathbf{x}_i,\mathbf{x}_j)$; $s(\mathbf{x}_t)=\sum_{i:\alpha_i>0}\alpha_iz_i k(\mathbf{x}_i,\mathbf{x}_t)+b$.
- **Esempio quadratico:** $\boldsymbol\Phi(\mathbf{x})=[\text{vec}(\mathbf{x}\mathbf{x}^T);\sqrt2\,\mathbf{x};1]$ ⇒ $k(\mathbf{x}_1,\mathbf{x}_2)=(\mathbf{x}_1^T\mathbf{x}_2+1)^2$. Generale: **polinomiale di grado $d$** $k=(\mathbf{x}_1^T\mathbf{x}_2+1)^d$.
- **RBF / Gaussian:** $k(\mathbf{x}_1,\mathbf{x}_2)=e^{-\gamma\|\mathbf{x}_1-\mathbf{x}_2\|^2}$ → mapping **infinito-dimensionale**; $\gamma$ = larghezza ($\gamma$ piccolo = kernel largo, SV influenza molti punti; $\gamma$ grande = kernel stretto, influenza locale).
- **Mercer:** condizione **sufficiente** perché $k$ sia un dot-product in qualche spazio: $k$ simmetrica e $\int k(\mathbf{u},\mathbf{v})g(\mathbf{u})g(\mathbf{v})\,d\mathbf{u}\,d\mathbf{v}>0$ per ogni $g$ a quadrato integrabile. Non dice **come** costruire kernel; si usano kernel noti o regole di composizione (es. somma di kernel è kernel).
- **Logical move:** separazione lineare nello spazio espanso ⇔ separazione **non-lineare** nello spazio originale. Il costo del duale dipende solo da $N$ (non da $\dim\mathcal{H}$).

### P9 — Considerazioni pratiche e limiti (Limitations / Variants)
- Scelta del kernel e dei suoi iperparametri → **cross-validation**; scelta di $C$ → cross-validation.
- A volte basta il lineare (feature già ad alta dimensione, classi quasi separabili).
- **Non invariante** a trasformazioni affini → spesso center + whiten.
- **Niente interpretazione probabilistica** dello score → serve **post-processing / calibrazione** per stimare posterior.
- **Sbilanciamento classi** / prior diverso da quello target: usare $C$ per-classe/per-sample, $C_T=C\frac{\pi_T}{\pi_T^{emp}}$, $C_F=C\frac{\pi_F}{\pi_F^{emp}}$ (simula le proporzioni d'applicazione al training) — ma sempre senza score probabilistico.
- **Multiclasse difficile**; i kernel non sono esclusivi delle SVM (es. LR kernelizzabile: kernel trick applicabile a problemi che dipendono solo da dot-product).

---

## 3. DEPENDENCY MAP (ordine di studio)

1. **P1** motivazione (perché il margine, link a LR) →
2. **P2** definizione di margine e criterio max-min →
3. **P3** riformulazione canonica → primal QP (qui stanno i due "logical move": invarianza di scala + vincolo attivo) →
4. **P4** Lagrangiana → duale → KKT → support vectors (cuore matematico) →
5. **P5** soft margin (generalizza P3/P4 con slack e box $0\le\alpha\le C$) →
6. **P6** hinge loss + confronto LR (richiede P5) →
7. **P7** inferenza primal/dual (richiede P4) →
8. **P8** kernel (richiede che P4/P7 dipendano solo da dot-product) →
9. **P9** pratica/limiti (richiede tutto).

Da padroneggiare prima di proseguire: i **due logical move** di P3 e la **complementary slackness** di P4 (spiega i support vectors). Senza questi, P5–P8 restano formule.

---

## 4. VERIFICATION QUESTIONS (libro chiuso)

- **P1/P2:** «Why is the maximum-margin criterion needed, given that infinitely many separating hyperplanes exist? Define the margin and write the max-min objective.»
- **P3:** «Show how the max-margin problem is turned into $\min\frac12\|\mathbf{w}\|^2$ s.t. $z_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1$. Which two equivalences are used?»
- **P4:** «Derive the dual problem from the Lagrangian. What are the KKT conditions and what defines a support vector?»
- **P5:** «How is the soft-margin SVM formulated? What is the role of $\xi_i$ and $C$, and how does the dual change?»
- **P6:** «Write the SVM and LR objectives in regularized-risk form and compare their per-sample losses.»
- **P7:** «Give the scoring function in primal and dual form and their complexities. Is the score a posterior?»
- **P8:** «What is the kernel trick? State Mercer's condition and give the polynomial and RBF kernels, explaining $\gamma$.»
- **P9:** «How do you handle class imbalance in SVM? Why is calibration needed? Why preprocess the data?»

---

## 5. FINAL SUMMARY CARD

→ salvata separatamente in [card.md](card.md) (input riusabile per il Prompt 2, confronto con [[06_LR]]).

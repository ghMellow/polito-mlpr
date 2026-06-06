# Cross-topic — Generative Gaussian Models vs Support Vector Machine

Card di input: [[04_GGM]] e [[09_SVM]]. Confronto generativo-probabilistico vs discriminativo-geometrico.

## 1. DIRECT CONNECTIONS
- **Stesso compito:** classificazione binaria con superficie di separazione nello spazio delle feature.
- **Boundary condivisibili:** il **Tied MVG** dà un boundary **lineare** $s=\mathbf{w}^T\mathbf{x}+b$, $\mathbf{w}\propto\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)$ — stessa *forma* dell'SVM lineare; il **full MVG (QDA)** dà un boundary **quadratico**, eguagliato dall'**SVM con kernel polinomiale** $d=2$. Stessa famiglia di superfici, ottenuta in modi opposti.
- **Entrambi affrontano il prior/bilanciamento**, ma in modi diversi (vedi sotto).

## 2. KEY DIFFERENCES
- **Generativo vs discriminativo.** GGM modella $f(\mathbf{x}\mid c)P(c)$ e ricava il posterior con Bayes; SVM non modella nessuna densità, cerca direttamente l'iperpiano a margine massimo.
- **Probabilità vs geometria.** GGM produce un **LLR**/posterior, pronto per soglia bayesiana $\mathrm{llr}\gtrless-\log\frac{P(c_1)}{P(c_0)}$ e per cambiare prior/costi a deployment. SVM produce uno score **senza** significato probabilistico → calibrazione necessaria, niente soglia bayesiana nativa.
- **Training.** GGM: ML in **forma chiusa** classe-per-classe ($\boldsymbol\mu_c,\boldsymbol\Sigma_c$). SVM: **QP convesso** (duale + KKT), nessuna forma chiusa.
- **Cosa usa dei dati.** GGM usa statistiche aggregate (medie, covarianze) di **tutti** i punti; SVM dipende **solo dai support vector** e dai **dot-product** (kernelizzabile).
- **Assunzioni.** GGM assume class-conditional **gaussiane** (forte; data-efficient se vera) e $\boldsymbol\Sigma$ invertibile ($N>D$, spesso PCA). SVM **nessuna** assunzione di densità; più robusto a feature non-gaussiane.
- **Non-linearità.** GGM: full $\boldsymbol\Sigma$ → quadratico (cresce con $\frac{D(D+1)}2$ parametri per classe). SVM: kernel → anche $\infty$-dim senza costruire $\boldsymbol\Phi$.
- **Multiclasse.** GGM: nativo via $\arg\max$ del log-posterior su $K$ classi. SVM: difficile.

## 3. COMPARISON TABLE

| Asse | Generative Gaussian (MVG / Tied) | Support Vector Machine |
|---|---|---|
| Type | Generativo: $f(\mathbf{x}\mid c)P(c)$, poi Bayes | Discriminativo, non probabilistico |
| Objective | ML delle gaussiane class-conditional | margine massimo / hinge + $\frac12\|\mathbf{w}\|^2$ |
| Assumptions | class-conditional gaussiane; $\boldsymbol\Sigma$ invertibile | nessuna assunzione di densità |
| Training | forma chiusa, classe per classe | QP convesso (duale, KKT) |
| Inference / score | LLR $\log\frac{f(\mathbf{x}\mid c_1)}{f(\mathbf{x}\mid c_0)}$, posterior via Bayes | $s=\sum_{SV}\alpha_iz_ik(\mathbf{x}_i,\mathbf{x})+b$; no posterior |
| Decision rule | $\mathrm{llr}\gtrless-\log\frac{P(c_1)}{P(c_0)}$; quadratico (lineare se tied) | $s\gtrless0$; lineare o non-lineare via kernel |
| Priors | separati ⇒ soglia/costi nativi a deployment | nessun prior nativo; costi $C_T,C_F$ |
| Non-linearità | quadratico (full $\boldsymbol\Sigma$), molti parametri | kernel (anche $\infty$-dim) |
| Multiclasse | nativo ($\arg\max$ log-posterior) | difficile |
| Limitazioni | assunzione gaussiana; $N>D$ | niente probabilità; non invariante affine; CV |

## 4. COMPARISON QUESTIONS
- «Compare a Generative Gaussian classifier and an SVM on objective, training and decision rule. Which one yields a Bayes-ready score and why?»
- «Both a Tied-covariance MVG and a linear SVM produce a linear boundary $\mathbf{w}^T\mathbf{x}+b$. How is $\mathbf{w}$ obtained in each case, and what does the score mean?»
- «How does each model achieve a non-linear decision boundary, and how do they handle class priors / imbalance?»

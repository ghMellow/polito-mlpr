# Cross-topic comparison — Logistic Regression vs LDA (Prompt 2)

Input cards: `raw/06_LR/card.md`, `raw/02_LDA/card.md`.
Tema: due **classificatori binari lineari**, criterio geometrico vs discriminativo. Confronto extra
(non tra i pairing pubblicati; il confronto "canonico" di LDA è con il Tied MVG, Q2).

---

## 1 — Direct connections

- **Stessa decision function lineare.** Entrambi classificano con $s=\mathbf{w}^T\mathbf{x}+b$ e un
  boundary che è un **iperpiano ortogonale a $\mathbf{w}$**.
- **Stessa hypothesis class.** Il boundary lineare è ciò che un modello gaussiano a covarianza comune
  implica: LDA lo assume esplicitamente (gaussiane equal-cov), LR lo assume sotto forma di **log-odds
  lineare**. Stesso spazio di soluzioni, criteri diversi per sceglierne una.

## 2 — Key differences

- **Obiettivo / derivazione.** LDA massimizza il **Fisher ratio** $S_B/S_W$ — *geometrico*, in **forma
  chiusa** (eigenproblem $S_W^{-1}S_B\mathbf{w}=\lambda\mathbf{w}$). LR minimizza la **cross-entropy /
  empirical risk** — *discriminativo*, **numerico** (no closed form).
- **Assunzioni.** LDA assume **gaussiane a covarianza comune**; LR assume solo **log-odds lineare**,
  nessun modello di densità → più robusta a feature non-gaussiane.
- **Doppio ruolo di LDA.** LDA è anche una tecnica di **dimensionality reduction** (≤ $K-1$ direzioni);
  LR è **solo** un classificatore, non proietta.
- **Soglia / priori.** LDA usa la soglia **midpoint** $t=\frac{m_1+m_2}{2}$ (ottima a priori uguali);
  lo score LR riflette il **prior empirico** → recalibrabile a LLR. Entrambi lineari.
- **Non-linearità.** LR la ottiene con feature expansion $\phi(x)$; LDA, essendo geometrica/eigen, non
  si estende naturalmente al non-lineare.
- **Multiclasse.** LDA: riduzione di dimensionalità ≤ $K-1$ + classificatore a valle. LR: **softmax** diretto.

## 3 — Comparison table

| Axis | Logistic Regression | LDA (binary) |
|---|---|---|
| Obiettivo | Min cross-entropy / empirical risk (discriminativo) | Max Fisher ratio $\frac{\mathbf{w}^TS_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}$ (geometrico) |
| Training | Numerico, no closed form | **Closed form** (eigenproblem / $S_W^{-1}\Delta\mu$) |
| Assunzioni | Log-odds lineare; nessun modello di $f_X$ | Gaussiane a **covarianza comune** |
| Direzione $\mathbf{w}$ | Stimata ottimizzando il posterior | $\propto S_W^{-1}(\boldsymbol\mu_2-\boldsymbol\mu_1)$ |
| Decision rule | $s\gtrless0$ (recalibrabile a LLR) | $\mathbf{w}^T\mathbf{x}\lessgtr t=\frac{m_1+m_2}{2}$ (midpoint) |
| Doppio uso | Solo classificatore | Anche **dim. reduction** (≤ $K-1$ dir.) |
| Non-linearità | Sì, via $\phi(x)$ | No (geometrica) |
| Multiclasse | Softmax diretto | Dim. reduction ≤ $K-1$ + classificatore |

## 4 — Comparison questions (exam-style)

1. LDA e LR producono entrambi un classificatore lineare: confronta **obiettivo di training** (Fisher
   ratio vs cross-entropy), forma della soluzione (closed form vs numerica) e assunzioni.
2. Perché LR si estende facilmente al non-lineare e al multiclasse mentre LDA ha il doppio ruolo di
   riduzione di dimensionalità? Discuti il limite $K-1$ di LDA.
3. Confronta la soglia di decisione dei due modelli (midpoint vs score legato al prior empirico) e
   come ciascuno gestisce priori diversi da quelli del training.

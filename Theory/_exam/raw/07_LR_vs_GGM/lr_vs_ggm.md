# Cross-topic comparison — Logistic Regression vs Generative Gaussian Models (Prompt 2)

Input cards: `raw/06_LR/card.md`, `raw/04_GGM/card.md`.
Tema: **discriminativo vs generativo** (il confronto concettuale per eccellenza). Non è un pairing
esplicito tra le domande pubblicate, ma sostanzia la "probabilistic interpretation" della Theory Q4.

---

## 1 — Direct connections

- **Entrambi probabilistici → Bayes decision.** Tutti e due producono un posterior $P(C\mid x)$ e
  decidono confrontando un (log-)posterior-ratio con una soglia da priori/costi.
- **Stessa forma lineare (LR ↔ Tied MVG).** Il **Tied MVG** ha log-posterior-ratio
  $s=\mathbf{w}^T x+b$ con $\mathbf{w}\propto\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)$
  — **esattamente la forma parametrica che la LR assume a priori**. LR e Tied MVG vivono nella
  **stessa hypothesis class** (log-odds lineare): differiscono solo in *come* stimano $(\mathbf{w},b)$.
- **Stessa estensione quadratica.** La **full MVG (QDA)** ha boundary quadratico; la LR con feature
  expansion $\phi(x)$ quadratica produce la stessa famiglia di superfici quadratiche.
- **Entrambi ML.** MVG: ML sulle densità class-conditional. LR: ML sulle label (conditional likelihood).

## 2 — Key differences

- **Cosa modellano.** MVG (**generativo**) modella la *joint* via $f(x\mid c)$ e prior $P(c)$, poi
  applica Bayes. LR (**discriminativo**) modella *direttamente* il posterior $P(c\mid x)$ e **ignora**
  $f_X(x)$.
- **Stima dei parametri.** MVG: **forma chiusa** (medie + covarianze ML per classe). LR: **nessuna
  forma chiusa**, ottimizzazione numerica della cross-entropy/empirical risk.
- **Assunzioni.** MVG assume **gaussianità** (forte); LR assume solo **log-odds lineare** (più debole).
  Se la gaussiana è corretta, MVG è più *data-efficient*; se è violata, LR è spesso più robusta.
- **Priori e calibrazione.** MVG separa nativamente likelihood e prior → cambi prior/costi a
  deployment solo spostando la soglia (LLR Bayes-ready). LR incorpora il **prior empirico** del
  training nello score → serve **recalibrazione** ($s_{llr}=s-\log\frac{n_T}{n_F}$) o **prior-weighting**.
- **Numero di parametri.** Full MVG: $\frac{D(D+1)}{2}$ per classe (covarianze) → serve $N>D$,
  spesso PCA prima. LR lineare: solo $D+1$ → scala meglio, ma boundary lineare salvo expansion.
- **Multiclasse.** MVG: posterior per-classe diretti. LR: **softmax**.

## 3 — Comparison table

| Axis | Logistic Regression | Generative Gaussian (MVG / Tied) |
|---|---|---|
| Tipo | Discriminativo: $P(C\mid x)$ diretto | Generativo: $f(x\mid c)P(c)$ → Bayes |
| Formulazione | $P(1\mid x)=\sigma(\mathbf{w}^Tx+b)$ | $(X\mid c)\sim\mathcal N(\boldsymbol\mu_c,\boldsymbol\Sigma_c)$ |
| Training | ML/cross-entropy, **numerico** (no closed form) | ML **closed-form** per classe |
| Assunzioni | Log-odds **lineare**; nessun modello di $f_X$ | **Gaussianità** class-conditional |
| Decision rule (bin.) | $s\gtrless0$; lineare (quadr. se $\phi(x)$) | LLR $\gtrless-\log\frac{P(c_1)}{P(c_0)}$; quadr. (QDA), lineare se tied |
| Priori | Score legato al prior empirico → recalibra | Prior separato → soglia regolabile nativamente |
| # parametri | $D+1$ (lineare) | full $\frac{D(D+1)}{2}$/classe; tied 1 $\boldsymbol\Sigma$ |
| Multiclasse | Softmax | Posterior per-classe diretti |
| Forza | Robusto a feature non-gaussiane; pochi parametri | Data-efficient se la gaussiana regge; LLR pronto |

## 4 — Comparison questions (exam-style)

1. Confronta classificazione **discriminativa e generativa** usando LR e MVG: cosa modella ciascuno,
   come si stimano i parametri, quali assunzioni e quali implicazioni su robustezza e data-efficiency.
2. Mostra che LR (lineare) e **Tied MVG** condividono la stessa forma del posterior
   $\mathbf{w}^Tx+b$: dove sta allora la differenza (stima generativa vs discriminativa, gestione del prior)?
3. In funzione delle caratteristiche del dataset (gaussianità, $N$ vs $D$, prior mismatch), quando
   preferiresti LR e quando MVG?

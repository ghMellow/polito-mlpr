# Study Guide — Score Calibration and Fusion

Fonte: `Theory/ScoreCalibrationAndFusion/ScoreCalibrationAndFusion.tex`.
Secondo bacino del **slot teorico da 4 punti**. Si aggancia direttamente al gap actual−min DCF
della card 14: la calibrazione è *come* si chiude quel gap.

---

## Blocco 1 — Main pillars

1. **Perché gli score sono mis-calibrati**.
2. **Due strategie** (soglia su validation vs trasformazione app-independent).
3. **Funzione di calibrazione monotòna** + 3 metodi (isotonic, prior-weighted LR, generative).
4. **Prior-weighted logistic regression** (dettaglio del metodo affine).
5. **Generative score model** (gaussiano tied).
6. **Calibration set, split a 3 parti, K-fold**.
7. **Score fusion** (combinare più classificatori).

---

## Blocco 2 — Deep-dive per pillar

### P1 — Perché mis-calibrazione
- Molti score **non hanno interpretazione probabilistica** né sono LLR per la popolazione di valutazione.
- Cause tipiche (keyword):
  - **SVM**: gli score non sono probabilità.
  - **LR regolarizzata**: forte regolarizzazione distorce gli score.
  - **Modelli generativi**: overfitting o **mismatch** train/test → score mal calibrati.
- Conseguenza: decidere direttamente sugli score → mis-calibrazione → actual DCF > min DCF (card 14).

### P2 — Due strategie
1. **Soglia su validation set**: trovi una soglia (quasi) ottima per *una* applicazione. Devi conoscere l'applicazione target.
2. **Trasformazione $f$ degli score in LLR ben calibrati**: app-**independent**, decisioni ottime per diverse applicazioni. La conoscenza dell'app aiuta ma basta una stima grezza.
- Si sceglie la 2ª. $s_{cal} = f(s)$ con $f$ **monotòna** (preserva: score alti → $H_T$, bassi → $H_F$).

### P3 — Funzione di calibrazione + 3 metodi
- **Isotonic regression**: non-parametrica, monotòna, fit ottimo *sui dati di training*. Funzione **piecewise** non-lineare → serve interpolazione per score non visti, **niente estrapolazione** fuori range, costosa con set grande.
- **Prior-weighted logistic regression**: trasformazione **affine**, incorpora i prior; semplice ed efficace, **estrapola**, veloce, ma fit buono solo per un range limitato di operating point.
- **Generative score models**: assumono un modello probabilistico per la distribuzione degli score per classe.

### P4 — Prior-weighted LR (dettaglio)
- Score non calibrati trattati come **feature 1-D**; mapping affine:
  $$f(s) = \alpha s + \gamma$$
- $f(s)$ deve essere un LLR: $f(s) = \log\frac{f_{S|C}(s|H_T)}{f_{S|C}(s|H_F)} = \alpha s + \gamma$.
- Posterior log-odds per prior $\tilde\pi$: $\log\frac{P(H_T|s)}{P(H_F|s)} = \alpha s + \gamma + \log\frac{\tilde\pi}{1-\tilde\pi} = \alpha s + \beta$.
- Si stimano $\alpha,\beta$ con LR **prior-weighted non regolarizzata** sul calibration training set; poi $f(s) = \alpha s + \beta - \log\frac{\tilde\pi}{1-\tilde\pi}$.
- Objective (peso per classe $w_i$):
  $$R(\alpha,\gamma) = \sum_i w_i \log\!\Big(1 + e^{-z_i(\alpha s + \gamma + \log\frac{\tilde\pi}{1-\tilde\pi})}\Big),\quad w_i = \tilde\pi/n_T\ (z_i{=}{+}1),\ (1-\tilde\pi)/n_F\ (z_i{=}{-}1)$$
- Serve specificare $\tilde\pi$ → si ottimizza per una specifica app, ma spesso calibra bene anche app vicine.

### P5 — Generative score model
- Si modellano le densità class-conditional degli score $S|H_T$, $S|H_F$; calibrazione = LLR del modello:
  $$f_{cal}(s) = \log\frac{f_{S|H_T}(s)}{f_{S|H_F}(s)}$$
- Esempio gaussiano: $f_{S|H_T}=\mathcal{N}(s|\mu_T,v_T)$, $f_{S|H_F}=\mathcal{N}(s|\mu_F,v_F)$.
- **Tied** $v_T=v_F=v$ → trasformazione **monotòna** (preferito).

### P6 — Calibration set, split, K-fold
- Il calibration set deve essere **indipendente** da model-training e validation/eval (altrimenti bias).
- **Split a 3 parti**: Model train | Calibration train | (Cal.) Validation. Spesso calibration+validation estratti dal *former validation set* per riusare i modelli già addestrati.
- Trade-off: calibration grande → validation meno affidabile; calibration piccola → calibrazione meno affidabile.
- **Due scenari**:
  1. mis-calibrazione da score non-probabilistici / over-underfitting, ma popolazioni simili → calibration set dal training material; regolarizzazione di solito non serve (dati 1-D, basso overfitting).
  2. mis-calibrazione da **mismatch** train/applicazione → calibration set deve **mimare l'applicazione** (servono meno dati che per addestrare un classificatore intero; modelli gaussiani estendibili a dati non etichettati).
- **K-fold**: usa ogni campione in tutti i ruoli (train classifier, train calibration, valutazione). Per le metriche: **pool degli score** di tutti i fold, *non* media delle metriche per-fold; stessa soglia per tutti.
- **LOO**: K = N, caso estremo, robusto ma costoso. K grande = più robusto e più lento.
- **Modello finale**: si riaddestra un modello $M$ su *tutto* il training set (i K modelli usano solo $\frac{K-1}{K}$ dei dati). Idealmente ogni $M_i$ simile a $M$.

### P7 — Score fusion
- Combinare più classificatori che estraggono informazioni diverse.
- **Majority voting**: assegna l'etichetta più votata; limiti: regole di tie-break, ignora la *confidenza*.
- **Fusione di score** (LLR): somma corretta solo se i sistemi sono **indipendenti**; se correlati, somma scorretta. Si pesano i contributi:
  $$s_{fused} = \alpha^T s + \gamma$$
  trasformazione affine del vettore di score; pesi $\alpha$, bias $\gamma$ stimati con **logistic regression** (come la calibrazione, ma su vettori m-dimensionali invece che 1-D). Con un solo sistema → si ricade esattamente nella calibrazione.
- **Multiclasse**: decisioni non più riducibili a soglia; LR multiclasse per stimare pesi/bias di calibrazione e fusione.

---

## Blocco 3 — Dependency map

gap actual−min DCF (card 14) → mis-calibrazione → strategia (soglia | trasformazione monotòna) → 3 metodi → [prior-weighted LR: $f(s)=\alpha s+\gamma$] / [generative tied gaussian] → calibration set indipendente → split 3 parti / K-fold → modello finale su tutto il training → estensione: **score fusion** ($s_{fused}=\alpha^T s+\gamma$, LR su score m-dim).

---

## Blocco 4 — Verification questions

1. 3 cause tipiche di mis-calibrazione (SVM, LR regolarizzata, generativo overfit/mismatch).
2. Le due strategie e perché si preferisce la trasformazione monotòna.
3. Isotonic vs prior-weighted LR: pro/contro (estrapolazione, range, costo).
4. Forma della calibrazione affine e perché $f(s)$ è un LLR.
5. Perché il calibration set deve essere indipendente; come si fa lo split a 3 parti.
6. Differenza tra i due scenari di mis-calibrazione e cosa cambia nel calibration set.
7. Perché nel K-fold si fa pooling degli score e non media delle metriche.
8. Relazione tra calibrazione e fusione ($s_{fused}=\alpha^T s+\gamma$).

---

## Blocco 5 — Final summary card
Vedi `card.md`.

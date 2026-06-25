# Summary Card — Score Calibration and Fusion

*(Slot teorico da 4 punti, secondo bacino. Si aggancia al gap actual−min DCF della card 14.)*

| Asse | Contenuto chiave |
|---|---|
| **Perché mis-calibrazione** | Gli score spesso non sono LLR per la popolazione di valutazione. Cause: **SVM** (score non probabilistici), **LR regolarizzata** (regolarizzazione distorce), **generativi** (overfit o **mismatch** train/test). → actual DCF > min DCF. |
| **Due strategie** | (1) **soglia su validation set** per una specifica app (serve conoscere l'app); (2) **trasformazione $f$** che mappa score → LLR ben calibrati, **app-independent**. Si preferisce la (2), con $f$ **monotòna**. |
| **3 metodi di $f$** | **Isotonic regression**: non-param., piecewise, fit ottimo sui dati, **no estrapolazione**, costosa. **Prior-weighted LR**: affine, incorpora prior, **estrapola**, veloce, range limitato. **Generative score model**: assume distribuzione degli score per classe. |
| **Prior-weighted LR** | Score = feature 1-D; $f(s)=\alpha s+\gamma$ interpretato come LLR. Posterior: $\log\frac{P(H_T\|s)}{P(H_F\|s)}=\alpha s+\gamma+\log\frac{\tilde\pi}{1-\tilde\pi}=\alpha s+\beta$. Si stimano $\alpha,\beta$ con LR prior-weighted non regolarizzata; $f(s)=\alpha s+\beta-\log\frac{\tilde\pi}{1-\tilde\pi}$. Serve specificare $\tilde\pi$. |
| **Generative score model** | $f_{cal}(s)=\log\frac{f_{S\|H_T}(s)}{f_{S\|H_F}(s)}$, es. gaussiano $\mathcal{N}(s\|\mu_T,v_T),\mathcal{N}(s\|\mu_F,v_F)$. **Tied** $v_T=v_F$ → trasformazione monotòna. |
| **Calibration set / split** | Deve essere **indipendente** da model-train e validation/eval. Split a 3: **Model train \| Calibration train \| (Cal.) Validation**; spesso cal.+val. dal *former validation set*. Trade-off dimensione. |
| **Due scenari** | (1) score non-probabilistici / over-underfit, popolazioni simili → cal. dal training material, no regolarizzazione (1-D). (2) **mismatch** train/app → cal. set deve **mimare l'applicazione** (meno dati di un classificatore intero). |
| **K-fold / LOO** | Ogni campione in tutti i ruoli. Metriche: **pool degli score**, non media per-fold; stessa soglia. **LOO**: K=N. Modello finale riaddestrato su **tutto** il training set. |
| **Score fusion** | Majority voting (ignora confidenza, tie-break). Fusione LLR: somma corretta solo se sistemi **indipendenti**. Generale: $s_{fused}=\alpha^T s+\gamma$, pesi via **LR** su vettori score m-dim. Un solo sistema → calibrazione. Multiclasse: LR multiclasse. |

LLR = log-likelihood ratio · $\tilde\pi$ = prior efficace

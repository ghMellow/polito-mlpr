# Blueprint flussi — domande di Programmazione MLPR

Ripasso rapido per le domande di codice. **Si assume sempre che i prototipi `train*` / `score*` / `evaluateScores` siano già implementati e funzionanti: il compito è orchestrarli.** Solo in PCA/LDA può capitare che chiedano di *implementare* anche le funzioni (vedi §A.1 / §A.2).

## Indice

**Regola d'oro** (quale set quando) — sotto.

Parte A — Blueprint per modello (ordine d'esame):
- [A.1 — PCA](#a1--pca-preprocessing)
- [A.2 — LDA](#a2--lda)
- [A.3 — Generative Gaussian Models (MVG / Naive Bayes / Tied)](#a3--generative-gaussian-models-ggm)
- [A.4 — Logistic Regression](#a4--logistic-regression-lr)
- [A.5 — Support Vector Machines](#a5--support-vector-machines-svm)
- [A.6 — Gaussian Mixture Models](#a6--gaussian-mixture-models-gmm)

Parte B — Pattern di task ricorrenti:
- [B.1 — Hyperparameter optimization (single-fold CV)](#b1--hyperparameter-optimization-single-fold-cv)
- [B.2 — Calibrazione binaria](#b2--calibrazione-binaria)
- [B.3 — PCA dim optimization + classificatore](#b3--pca-dim-optimization--classificatore)

Parte C — [Implementare le funzioni (plus PCA/LDA)](#parte-c--implementare-le-funzioni-plus-pcalda)

---

## Regola d'oro: quale set usare quando

| Fase | Dati | A cosa serve |
|------|------|--------------|
| Train del modello | **DTR, LTR** | addestrare |
| Selezione iperparam / model selection | **DVAL, LVAL** | scegliere, NON addestrare il modello finale |
| Valutazione finale | **DTE, LTE** | misura onesta, si tocca **una sola volta alla fine** |

Mai usare DTE per scegliere nulla. Mai usare LTE se non per l'`evaluateScores` finale. Qualsiasi pre-processing (PCA, normalizzazione, calibrazione) si **fitta su DTR** e si **applica** a DVAL/DTE.

**Score e soglia (caso binario, modelli generativi/probabilistici).** Lo score è un log-likelihood ratio `llr = log f(x|H1) - log f(x|H0)`. Con prior effettivo `π` e costi unitari la soglia bayesiana è `t = -log(π/(1-π))`; si predice classe 1 se `llr > t`.

**Direzione del confronto in model selection:** minDCF / error rate → tieni il **minore** (`<`); accuracy → il **maggiore** (`>`).

---

# Parte A — Blueprint per modello

> Convenzione: dati come **colonne** (`D` ha shape `(feature, n_sample)`). Prototipi assunti scritti in stile generico `trainX(...)` / `scoreX(model, D)`.

## A.1 — PCA (preprocessing)

Non è un classificatore: riduce la dimensionalità prima di un classificatore.

- **Prototipi:** `trainPCA(D, m)` → modello; `applyPCA(model, D)` → dati proiettati a `m` dim.
- **Iperparam:** `m` ∈ {1, …, n_feature}. È l'unico grado di libertà → si tuna su DVAL (vedi B.3).
- **Flusso:** fit su **DTR**, applica a DTR/DVAL/DTE, poi un classificatore a valle.

```python
pca  = trainPCA(DTR, m)
DTRp = applyPCA(pca, DTR)      # stessa proiezione applicata a tutto
DVALp, DTEp = applyPCA(pca, DVAL), applyPCA(pca, DTE)
```

**Promemoria:** PCA non supervisionato (non usa le label). Fit SOLO su DTR.

## A.2 — LDA

Due usi: (a) **riduzione** a ≤ C−1 dimensioni (C = n. classi), (b) **classificatore** binario diretto.

- **Prototipi:** `trainLDA(D, L)`; `scoreLDA(model, D)` → score (LLR per il caso binario a covarianza condivisa).
- **Iperparam:** nessuno da tunare. Invariante a trasformazioni affini ⇒ se davanti c'è PCA, la metti tu (B.3). Nel binario la "scelta" è solo la soglia.
- **Flusso classificatore:**

```python
m = trainLDA(DTR, LTR)
s = scoreLDA(m, DVAL)                 # poi DTE per la valutazione finale
print(evaluateScores(s, LVAL))
```

**Score & soglia:** LDA binario equivale al MVG tied → lo score è un LLR, soglia `t = -log(π/(1-π))`.

## A.3 — Generative Gaussian Models (GGM)

Famiglia: **MVG** (full covariance), **Naive Bayes** (cov. diagonale), **Tied** (cov. condivisa). Si stimano per classe da DTR.

- **Prototipi:** `trainGGM(D, L, model_type)` (o `trainMVG`/`trainTied`/`trainNaiveBayes`); `scoreGGM(model, D)` → **LLR** classe1/classe0.
- **Iperparam:** nessun iperparam continuo. La "scelta" è la **variante** (MVG / Naive / Tied) — la selezioni su DVAL provando le 3.
- **Flusso (selezione variante):**

```python
best, best_score = None, float('inf')
for mt in ['mvg', 'naive', 'tied']:
    m  = trainGGM(DTR, LTR, mt)
    sc = evaluateScores(scoreGGM(m, DVAL), LVAL)
    if sc < best_score:
        best_score, best = sc, mt
final = trainGGM(DTR, LTR, best)
print(evaluateScores(scoreGGM(final, DTE), LTE))
```

**Score & soglia:** score = LLR già calibrato (è un vero rapporto di verosimiglianze) → soglia `t = -log(π/(1-π))`, predici 1 se `llr > t`.

## A.4 — Logistic Regression (LR)

- **Prototipi:** `trainLogReg(D, L, lambda)` (eventualmente `prior` per la versione prior-weighted); `scoreLogReg(model, D)` → score `w·x + b`.
- **Iperparam:** `lambda` (regolarizzazione L2). Si tuna su DVAL (griglia log-spaziata, es. `[1e-4, 1e-3, …, 1]`). Opzionale: prior `πT` se prior-weighted.
- **Flusso:** identico al pattern B.1 ma con un solo iperparam `lambda`.

```python
best_l, best_score = None, float('inf')
for lam in lambdas:
    m  = trainLogReg(DTR, LTR, lam)
    sc = evaluateScores(scoreLogReg(m, DVAL), LVAL)
    if sc < best_score:
        best_score, best_l = sc, lam
final = trainLogReg(DTR, LTR, best_l)
print(evaluateScores(scoreLogReg(final, DTE), LTE))
```

**Score & soglia:** lo score LR ≈ LLR + log-odds del prior di training. Per ottenere l'LLR togli il bias del prior empirico: `llr = score - log(πemp/(1-πemp))` (con prior-weighted usi `πT`). Poi soglia `t = -log(π/(1-π))`. Se la traccia tratta lo score come già LLR, threshold diretto.

## A.5 — Support Vector Machines (SVM)

- **Prototipi:** `trainSVM(D, L, C, ...)`; kernel RBF → `gamma`; poly → grado `d` e bias `c`. `scoreSVM(model, D)` → score (margine), **non probabilistico**.
- **Iperparam:** `C` sempre; + `gamma` (RBF) o `(d, c)` (poly). Griglia 2D, si tuna su DVAL (vedi B.1).
- **Flusso:** vedi B.1 (doppio loop `C` × `gamma`).

**Score & soglia (IMPORTANTE):** lo score SVM **non** è un LLR → non si può applicare la soglia bayesiana né interpretarlo come probabilità. Per minDCF va bene (è una metrica su ordinamento), ma per **decisioni a un prior dato / actDCF serve calibrarlo** (→ B.2). Senza calibrazione la soglia "naturale" è solo 0 (sign del margine).

## A.6 — Gaussian Mixture Models (GMM)

- **Prototipi:** `trainGMM(D, L, num_components, cov_type)` (EM + LBG split); `scoreGMM(model, D)` → **LLR** classe1/classe0.
- **Iperparam:** **numero di componenti** per classe (potenze di 2: 1, 2, 4, 8, …) + tipo di covarianza (full / diagonal / tied). Si tunano su DVAL.
- **Flusso (selezione n. componenti):**

```python
best_g, best_score = None, float('inf')
for g in [1, 2, 4, 8, 16]:
    m  = trainGMM(DTR, LTR, g, 'full')
    sc = evaluateScores(scoreGMM(m, DVAL), LVAL)
    if sc < best_score:
        best_score, best_g = sc, g
final = trainGMM(DTR, LTR, best_g, 'full')
print(evaluateScores(scoreGMM(final, DTE), LTE))
```

**Score & soglia:** score = LLR → soglia `t = -log(π/(1-π))` come GGM.

---

# Parte B — Pattern di task ricorrenti

I task tipici combinano un modello (Parte A) con uno di questi schemi.

## B.1 — Hyperparameter optimization (single-fold CV)

**Cosa chiede:** ottimizza iperparam con single-fold, poi valuta su eval set.

1. Definisci le griglie.
2. Loop su tutti i combo: `train(DTR,...)` → `score(model, DVAL)` → `evaluateScores(s, LVAL)`; tieni il migliore.
3. Ri-addestra il modello finale coi best iperparam su **DTR**.
4. Valuta su **DTE/LTE** una volta sola.

```python
best, best_score = (None, None), float('inf')      # minDCF: meno è meglio
for C in C_values:
    for g in gamma_values:
        m  = trainSVM(DTR, LTR, C, g)
        sc = evaluateScores(scoreSVM(m, DVAL), LVAL)
        if sc < best_score:
            best_score, best = sc, (C, g)
final = trainSVM(DTR, LTR, *best)
print(evaluateScores(scoreSVM(final, DTE), LTE))
```

**Nota:** single-fold ⇒ il modello finale si allena su DTR. Solo se lo chiedono esplicitamente puoi unire DTR+DVAL (`np.hstack` / `np.concatenate`) per il fit finale.

## B.2 — Calibrazione binaria (no iperparam)

**Cosa chiede:** classificatore calibrato + label predette su eval, con prior effettivo `p` per la classe 1. Serve quando lo score non è un LLR (tipico: **SVM**).

**Idea:** base su DTR, **calibrazione allenata sugli score di DVAL** (indipendente dal training del base).

```python
base = trainClassifier(DTR, LTR)
cal  = trainCalibrationModel(scoreClassifier(base, DVAL), LVAL, p)   # score di DVAL + prior p
cTE  = applyCalibrationModel(cal, scoreClassifier(base, DTE))        # score calibrati = LLR
pred = (cTE > np.log(p / (1 - p))).astype(int)                       # soglia log-odds
```

**Param chiave:** `trainCalibrationModel(scoreDVAL, LVAL, p)`. La soglia è `log(p/(1-p))` perché lo score calibrato è un LLR (decisione bayesiana). Niente DTE per allenare/calibrare.

## B.3 — PCA dim optimization + classificatore (single-fold CV)

**Cosa chiede:** ottimizza la dim `m` di PCA con single-fold, poi valuta. Classificatore invariante ad affini e senza iperparam ⇒ **PCA è l'unico grado di libertà** e va ri-fittata dentro il loop.

**Punto critico:** PCA si **fitta su DTR** e si **applica** a DTR/DVAL/DTE (mai fit su val/test = data leakage).

```python
best_m, best_score = None, float('inf')
for m in range(1, DTR.shape[0] + 1):          # tutte le dim compatibili
    pca = trainPCA(DTR, m)
    clf = trainClassifier(applyPCA(pca, DTR), LTR)
    sc  = evaluateScores(scoreClassifier(clf, applyPCA(pca, DVAL)), LVAL)
    if sc < best_score:
        best_score, best_m, best_pca, best_clf = sc, m, pca, clf
sTE = scoreClassifier(best_clf, applyPCA(best_pca, DTE))
print(evaluateScores(sTE, LTE))
```

---

# Parte C — Implementare le funzioni (plus PCA/LDA)

Quando chiedono anche le implementazioni. Dati = **colonne** ⇒ `axis=1`.

**trainPCA(D, m):**
- `mu = D.mean(axis=1, keepdims=True)` ; `Dc = D - mu`
- `C = Dc @ Dc.T / N` (N = `D.shape[1]`)
- `eigvals, eigvecs = np.linalg.eigh(C)` (eigh → ordine **crescente**)
- top-m: `P = eigvecs[:, ::-1][:, :m]`
- return `{'mu': mu, 'P': P}`

**applyPCA(model, D):** `return model['P'].T @ (D - model['mu'])`

**trainLDA(D, L)** (gaussiano a covarianza condivisa):
- per ogni classe: `mu_c`, `n_c`, covarianza `C_c`; accumula `SW += n_c * C_c`
- `SW /= N` (pooled covariance); salva i prior `n_c/N` e le medie

**applyLDA(model, D)** binario → LLR per sample:
- `score = ll1 - ll0`, con `ll_c = -0.5 (x-mu_c)^T Cinv (x-mu_c) + log(prior_c)`

**Trucchi promemoria:**
- `eigh` → autovalori crescenti → inverti (`[:, ::-1]`) per i principali.
- Centra sempre con la **media del training** (in apply usi `model['mu']`, non ricalcoli).
- Covarianza `Dc @ Dc.T / N`, proiezione `P.T @ Dc`.

---

## Checklist mentale (vale per tutti)

1. **Quale set per cosa?** train=DTR, scegli=DVAL, valuta=DTE (una sola volta).
2. **Cosa ottimizzo?** SVM→`C`(+`gamma`); LR→`lambda`; GMM→n. componenti; GGM→variante; PCA→`m`; LDA→niente.
3. **Fit solo su DTR**, applica/score su DVAL e DTE (PCA, calibrazione: stessa regola).
4. **Lo score è un LLR?** GGM/GMM/LDA/LR sì (LR a meno del prior) → soglia `-log(π/(1-π))`. SVM no → calibra (B.2).
5. **Direzione confronto:** minDCF/error → `<` ; accuracy → `>`.
6. **Modello finale** ricostruito con la config migliore, poi un solo passaggio su DTE.

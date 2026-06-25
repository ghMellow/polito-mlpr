# Dev Log — MLPR

## 2026-06-23 — Card d'esame 14 (Bayes/Evaluation) + 15 (Calibration)  [sessione: c1ec221b]

**Intent:** appello 18-giu-2026 con struttura nuova (3 domande teoriche 12+8+4 invece di 2). L'utente nota che lo slot da **4 punti** pesca da "parti minori del programma" e ipotizza che il prossimo possa essere "la funzione di costo"; chiede *"trova possibili candidati per questa domanda"* e poi *"si crea e dimmi quali hai creato"* (libertà piena su cosa generare).

**Divergenze:**
- Proposto framing: lo slot 4pt copre esattamente i **due capitoli non toccati** dalle prime due domande-modello → Bayes Decisions & Model Evaluation (§5) e Score Calibration (§9). Verificato che `MLPR_exam.tex` (collection 01–13) **non aveva nessuna card** su valutazione/DCF/calibrazione: gap reale.
- Sul guess "funzione di costo": segnalato che la domanda uscita aveva **già bruciato** cost/empirical Bayes risk; candidati freschi più probabili = min vs actual DCF, DCF normalizzata/prior efficace, ROC/EER.
- Proposto di splittare in **due card** (14 + 15) a specchio dei due capitoli sorgente, anziché una sola.

**Decisioni:** utente accetta la creazione ("si crea"). Generate card 14 `14_Bayes_Evaluation` e 15 `15_Calibration` (study guide + card.md + .tex condensato ciascuna), agganciate al master dopo il blocco 13.

**Esito/Problemi:**
- Fase 3 (confronti) **saltata**: valutazione/calibrazione non sono modelli con pairing canonico.
- Creato anche `example_exam/MLPR_exam_2026-06-18.md` (testo appello).
- Master compila pulito (45 pp, exit 0); solo overfull hbox cosmetici dalle equazioni objective lunghe.

**Lesson learned:** la struttura a 3 domande sdoppia il programma in *modelli* (Q1/Q2, già coperti dalle 13 card) + *metodologia di valutazione* (Q3, prima scoperta). Lo slot 4pt è il bacino Bayes/DCF/calibrazione; le due card chiudono il gap a prescindere dalla rotazione del prof. Filo conduttore che lega le due card: gap **actual − min DCF** = perdita per mis-calibrazione (fine card 14) → la card 15 è *come* si chiude quel gap.

## 2026-06-06 — Gaussian Mixture Models + confronto GMM↔GGM

**Done:**
- `/prep-esame GMM` (run completo): study guide + card in `raw/12_GMM/` e testo d'esame `12_GMM.tex` (somma pesata di gaussiane, latent-variable/marginal view, log-lik ill-posed, responsabilità + hard-assignment, K-means come caso limite, EM generale ELBO/E/M + monotonicità, EM per GMM con statistiche $N_c,F_c,S_c$, classificazione closed-set/open-set, varianti diagonal/tied/LBG, degenerazione & scelta di K).
- Fase 3: confronto `13_GMM_vs_GGM` (MVG = GMM a 1 componente; closed-form ML vs EM; boundary lin/quad vs non-lineare arbitrario; supervised vs unsupervised), grezzo in `raw/` e `.tex` agganciato al master subito dopo GMM.
- Master compila pulito (30 pp, exit 0).

**Lesson learned:**
- Pairing canonico **GMM↔GGM**: l'M-step EM è la versione *soft* (pesata per $\gamma_{c,i}$) della ML closed-form della MVG → collassa ad essa con responsabilità hard 0/1.
- Attenzione ai falsi amici di terminologia: in GMM *tied* condivide $\Sigma$ tra le **componenti di una classe** (≠ Tied MVG, tra classi) e *diagonal* **non** è Naive Bayes.

## 2026-06-06 — Support Vector Machines + confronti LR↔SVM e GGM↔SVM

**Done:**
- `/prep-esame SVM` (run completo): study guide + card in `raw/09_SVM/` e testo d'esame `09_SVM.tex` (max-margin, forma canonica → primal QP, Lagrangiana/duale/KKT/support vector, soft margin + box constraint, hinge loss vs LR, score primal/dual, kernel trick + Mercer + poly/RBF, pratica).
- Fase 3: due confronti — `10_LR_vs_SVM` (canonico: risk minimization, hinge vs log-loss) e `11_GGM_vs_SVM` (generativo-probabilistico vs discriminativo-geometrico), grezzi in `raw/` e `.tex` agganciati al master subito dopo SVM.
- Master compila pulito (25 pp, exit 0).

**Lesson learned:**
- Chiuso il pairing **LR↔SVM** rimasto in sospeso: stessa forma regolarized-risk $\frac\lambda2\|\mathbf{w}\|^2+\frac1n\sum\ell(z_is)$, differenza *solo* nella loss per-sample (hinge azzera oltre il margine ⇒ sparsità/support vector; log-loss sempre >0 ⇒ tutti i punti). SVM = lettura geometrica della regolarizzazione di LR.
- Il legame GGM↔SVM passa per la *forma* del boundary (Tied=lineare, full=quadratico ≈ kernel poly d=2), ma il punto d'esame è generativo/Bayes-ready vs discriminativo/non-probabilistico (score SVM da calibrare).

## 2026-06-06 — Logistic Regression (Q4) + confronti LR↔GGM e LR↔LDA

**Done:**
- `/prep-esame LR` (run completo): generati study guide + card in `raw/06_LR/` e testo d'esame `06_LR.tex` (Theory Q4: classification rule, interpretazione probabilistica dello score, ML⇔cross-entropy⇔empirical risk, regolarizzazione, recalibrazione/prior-weighting, softmax, feature expansion).
- Fase 3: generati due confronti — `07_LR_vs_GGM` (discriminativo vs generativo) e `08_LR_vs_LDA` (geometrico vs discriminativo), con grezzi in `raw/` e `.tex` agganciati al master.
- Master compila pulito (17 pp, exit 0).

**Lesson learned:**
- Quale confronto ha senso per LR: **PCA no** (unsupervised, solo pre-processing); **LDA sì ma parziale** (stesso boundary lineare, criterio geometrico vs discriminativo); **GGM/MVG il più profondo** — il Tied MVG ha log-posterior-ratio $\mathbf{w}^Tx+b$, *stessa hypothesis class* di LR, differenza solo nella stima (generativa vs discriminativa).
- Il pairing che il prof chiede *esplicitamente* per LR è **LR↔SVM** (risk minimization): rimandato finché SVM non è generato.

## 2026-06-06 — Confronto LDA ↔ Tied MVG (Q2) + rifiniture PCA

**Done:**
- `/prep-esame GGM --solo-confronti`: generato confronto **LDA ↔ Tied MVG** (Theory Q2). Nuovo `05_LDA_vs_TiedMVG.tex` agganciato al master; grezzo in `raw/05_LDA_vs_TiedMVG/`.
- Rigenerate come **testo** le card mancanti: `raw/02_LDA/card.md`, `raw/04_GGM/card.md` (prima esistevano solo card-immagine).
- Rifiniture `01_PCA.tex`: aggiunte $\mathbf{y}=P^T\mathbf{x}$, $\hat{\mathbf{x}}=P\mathbf{y}$ e disambiguazione second-moment vs covarianza.
- Fix refuso master: "Generative Gaussian Moldes" → "Models". Master compila (12 pp, exit 0).

**Lesson learned:**
- Punto-chiave Q2: sotto **priori uguali** LDA binaria e Tied MVG sono **identici** (stessa direzione $\propto\Sigma^{-1}\Delta\mu$ *e* stessa soglia midpoint), perché $S_W$ = covarianza tied. Differiscono su: origine della soglia (midpoint vs log-odds → priori/costi), multiclasse (LDA ≤ $K-1$ direzioni vs MVG $K$ posteriors), output (score vs LLR).

## 2026-06-06 — Skill /prep-esame per il workflow d'esame

**Done:**
- Creata skill globale `prep-esame` (`~/.claude/skills/prep-esame/SKILL.md`) che automatizza: Prompt 1 (study guide + card di testo) → `.tex` condensato d'esame → Prompt 2 (confronti tra modelli) → aggancio a `MLPR_exam.tex` → compilazione `latexmk`.
- Registrata in `~/.claude/CLAUDE.md` con trigger `/prep-esame`.
- Sistemato mismatch di numerazione: `Theory/_exam/raw/03_GGM` → `04_GGM` (allineato a `_exam/04_GGM.tex`).

**Problemi:**
- La summary card della PCA era salvata come immagine (`pca.png`), quindi non riusabile dal Prompt 2. La skill ora salva sempre la card come testo (`raw/NN_Topic/card.md`) e rigenera quelle mancanti dei vecchi argomenti prima di confrontare.
- I capitoli del prof contengono più argomenti d'esame (PCA e LDA in `DimensionalityReduction.tex`): la skill estrae la sola sezione pertinente.

**Lesson learned:**
- I due prompt condividono lo stesso vocabolario di 8 assi d'esame: le card si impilano e la tabella comparativa del Prompt 2 si allinea riga-per-riga → la card va salvata come testo strutturato, non come immagine.
- I `.tex` in `_exam/` sono frammenti `\input`: niente `\documentclass`/`\section` di primo livello (lo aggiunge il master); `\section*` interni ammessi.


**Usare la skill**
- /prep-esame GMM
- /prep-esame SVM --solo-confronti (solo pt2)
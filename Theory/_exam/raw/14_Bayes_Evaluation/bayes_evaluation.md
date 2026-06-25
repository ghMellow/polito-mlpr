# Study Guide — Bayes Decisions and Model Evaluation

Fonte: `Theory/BayesDecisionsAndModelEvaluation/BayesDecisionsAndModelEvaluation.tex`.
Tema del **slot teorico da 4 punti** (valutazione / costo / DCF). Non è un "modello": è il
framework con cui *si misura* un classificatore. Gli 8 assi-modello non si applicano; struttura per
fasi della pipeline di valutazione.

---

## Blocco 1 — Main pillars

1. **Perché accuracy/error rate non bastano** (3 difetti).
2. **Confusion matrix e per-class error rates** (Pfn, Pfp, TPR, FPR; prior empirico).
3. **Costo e Bayes risk** (cost matrix, Bayes risk, empirical Bayes risk = DCFu).
4. **DCF normalizzata** (dummy systems, prior efficace, invarianza allo scaling).
5. **Optimal Bayes decision** (costo atteso a posteriori, soglia, regola LLR).
6. **Strumenti grafici** (ROC, DET, Equal Error Rate).
7. **min DCF vs actual DCF** (gap = mis-calibrazione).

---

## Blocco 2 — Deep-dive per pillar

### P1 — Perché accuracy non basta
- **accuracy** = #corretti / #totali; **error rate** = 1 − accuracy.
- **3 difetti** (keyword da citare sempre):
  1. **non considera il costo** dei diversi tipi di errore;
  2. **dipende dal prior empirico** dell'evaluation set, che può non riflettere il prior dell'applicazione;
  3. **non è normalizzata** → non dice da solo se il classificatore è *utile* (meglio di una decisione senza dati).
- *Logical move*: vogliamo una metrica che (i) dipenda dall'applicazione non dal prior empirico, (ii) pesi i costi, (iii) sia confrontabile, (iv) sia normalizzata → porta direttamente alla DCF.

### P2 — Confusion matrix e per-class rates
- Per ogni (classe vera $C_j$, predizione $C_i$): conteggi; diagonale = corretti.
- Binario: TN, FN, FP, TP.
- $P_{fn} = \frac{FN}{FN+TP}$ (false negative rate, errore classe positiva); $P_{fp} = \frac{FP}{FP+TN}$ (false positive rate, errore classe negativa).
- $TPR = 1 - P_{fn}$ (recall/sensitivity); $TNR = 1 - P_{fp}$ (specificity).
- **Keyword chiave**: Pfn e Pfp usano *una classe alla volta* → **non dipendono dalle proporzioni di classe**, solo dalla performance dentro la classe.
- Prior empirico dell'eval set: $\pi^{emp}_E = \frac{TP+FN}{N}$.
- **Scomposizione dell'error rate**: $\text{err} = P_{fp}(1-\pi^{emp}_E) + P_{fn}\,\pi^{emp}_E$.
- *Logical move*: misuro Pfn/Pfp su dataset bilanciato e poi predico l'errore reale con il prior vero: $\text{err}_{app} = P_{fp}(1-\pi) + P_{fn}\,\pi$ → non serve raccogliere tanti campioni della classe rara.

### P3 — Costo e Bayes risk
- Il **costo** non è monetario: quantifica l'effetto relativo dei diversi errori; cambia da applicazione ad applicazione.
- Azione $a$, costo $C(a|k)$ se scelgo $a$ e il campione è di classe $k$.
- **Bayes risk** (costo atteso sull'applicazione): $B = E_{X,C|E}[C(a(x,R)|c)] = \sum_c \pi_c\, E_{X|C,E}[C(a(x,R)|c)\mid c]$.
- Non calcolabile (non conosciamo $f_{X|C,E}$); si approssima con la media sui campioni → **empirical Bayes risk**:
  $$B_{emp} = \sum_c \pi_c \frac{1}{N_c}\sum_{i:c_i=c} C(a(x_i,R)|c)$$
- Serve: confusion matrix + cost matrix + prior.
- Se $\pi_c = \pi^{emp}_c$ → coincide col costo totale di misclassificazione (come error rate ma *pesato dai costi*).
- Binario, cost matrix con 0 sulla diagonale, $C_{fn}$, $C_{fp}$:
  $$B_{emp} = \pi_T C_{fn} P_{fn} + (1-\pi_T) C_{fp} P_{fp} = DCF_u(\pi_T,C_{fn},C_{fp})$$
  = **unnormalized Detection Cost Function**.

### P4 — DCF normalizzata
- **Dummy system**: classifica sempre uguale, usa solo prior+costi, ignora i dati.
  - Sempre accetta: $P_{fp}=1, P_{fn}=0 \Rightarrow DCF_u=(1-\pi_T)C_{fp}$.
  - Sempre rifiuta: $P_{fp}=0, P_{fn}=1 \Rightarrow DCF_u=\pi_T C_{fn}$.
- **DCF normalizzata**: $DCF = \dfrac{DCF_u}{\min(\pi_T C_{fn},\,(1-\pi_T)C_{fp})}$.
  - ≈1 → inutile (come dummy); ≈0 → molto meglio del dummy; >1 → peggio del dummy.
- **Invarianza allo scaling dei costi**: moltiplicare tutti i costi per una costante non cambia la DCF norm → si può riscalare a costi unitari e assorbire prior+costi in un **prior efficace** $\tilde{\pi}$.
- $(\pi_T, C_{fn}, C_{fp})$ = **working point**; la tripletta è ridondante (esistono triplette equivalenti con stessa decision rule).
- Multiclasse: niente singolo parametro (prior efficace), ma empirical Bayes risk + DCF norm (scala col costo del miglior dummy) si calcolano comunque.

### P5 — Optimal Bayes decision
- Se il classificatore è probabilistico: costo atteso dell'azione $a$: $C_{x,R}(a) = \sum_k C(a|k) P(C=k|x,R)$.
- **Decisione ottima**: $a^*(x,R) = \arg\min_a C_{x,R}(a)$ → minimizza il costo atteso secondo le *credenze* del recognizer.
- Se recognizer ed evaluator coincidono ($C|X,R \sim C|X,E$), le Bayes decision **minimizzano la Bayes risk** (dimostrazione: $C_{x,R}(a) \ge C_{x,R}(a^*)\ \forall x$).
- Binario con cost matrix (0, $C_{fn}$, $C_{fp}$): scegli $H_T$ se $C_{fp}P(H_F|x) < C_{fn}P(H_T|x)$.
- Con generativo: $\log\frac{f(x|H_T)}{f(x|H_F)} \gtrless -\log\frac{\pi_T C_{fn}}{(1-\pi_T)C_{fp}}$.
- Per **LLR ben calibrati** $s=\log\frac{f(x|H_T)}{f(x|H_F)}$: soglia ottima
  $$t = -\log\frac{\tilde{\pi}}{1-\tilde{\pi}},\quad \tilde{\pi} = \frac{\pi_T C_{fn}}{\pi_T C_{fn}+(1-\pi_T)C_{fp}}$$
- **Keyword**: gli LLR *disaccoppiano* il classificatore dall'applicazione.

### P6 — ROC / DET / EER
- Score singolo (LLR, posterior log-ratio, score SVM); decisione = score vs soglia.
- Variando la soglia → tutte le combinazioni (Pfn, Pfp).
- **Equal Error Rate**: soglia dove $P_{fn}=P_{fp}$.
- **ROC**: TPR vs FPR. **DET**: FNR vs FPR. Valutano il classificatore su *tutte* le soglie.

### P7 — min DCF vs actual DCF
- **actual DCF**: DCF usando la soglia teorica $t=-\log\frac{\tilde\pi}{1-\tilde\pi}$ (assume score ben calibrati).
- **min DCF**: si fa variare $t$ su tutte le soglie possibili dell'eval set, si prende il DCF minimo → costo che pagheremmo conoscendo *a priori* la soglia ottima. Misura la **qualità** del classificatore (potere discriminante puro).
- **Gap actual − min = perdita per mis-calibrazione**. Se grande → score mis-calibrati → applicare calibrazione (vedi card 15).

---

## Blocco 3 — Dependency map

accuracy (insufficiente) → confusion matrix / per-class rates → cost matrix → Bayes risk → empirical Bayes risk = DCFu → [normalizzazione: dummy + prior efficace] = DCF → [come decidere: optimal Bayes decision → soglia] → [su tutte le soglie: ROC/DET/EER] → [min vs actual DCF] → mis-calibrazione → **card 15 (calibrazione)**.

---

## Blocco 4 — Verification questions

1. Quali sono i 3 difetti dell'accuracy? → costo, prior empirico, non-normalizzazione.
2. Perché Pfn/Pfp non dipendono dalle proporzioni di classe?
3. Scrivi DCFu binaria e spiega ogni termine.
4. Cosa fa la normalizzazione della DCF e cos'è un dummy system?
5. Definisci il prior efficace e l'invarianza allo scaling dei costi.
6. Da dove esce la soglia ottima $t=-\log\frac{\tilde\pi}{1-\tilde\pi}$?
7. Differenza tra min DCF e actual DCF, e cosa misura il loro gap?
8. Differenza ROC vs DET; cos'è l'EER?

---

## Blocco 5 — Final summary card
Vedi `card.md` (formato tabellare indicizzato).

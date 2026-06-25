# Summary Card — Bayes Decisions and Model Evaluation

*(Tema valutazione / costo / DCF — slot teorico da 4 punti. Indicizzata per fase della pipeline di valutazione, non per asse-modello.)*

> **Come il prof può spezzarla in domande da 4 pt** — i tre blocchi qui sotto sono *porzioni auto-contenute*: una domanda tipicamente pesca **un** blocco (a volte blocco 1 + un pezzo del 2). Legenda: 🔴 già uscito (18-giu-2026) · 🟢 candidato fresco.

---

## 🔴 BLOCCO 1 — "Da accuracy a empirical Bayes risk" *(uscito 18-giu-2026)*

*Filo: perché accuracy fallisce → costo → empirical Bayes risk. Si ferma prima della metrica operativa.*

| Asse | Contenuto chiave |
|---|---|
| **Perché non accuracy** | accuracy = #corretti/#tot, err = 1−acc. **3 difetti**: (1) non pesa il **costo** dei diversi errori; (2) dipende dal **prior empirico** dell'eval set ≠ prior applicazione; (3) **non normalizzata** → non dice se è meglio di una decisione senza dati. Si vuole una metrica: dipendente dall'app, pesata sui costi, confrontabile, normalizzata. |
| **Confusion matrix / rates** | TN,FN,FP,TP. $P_{fn}=\frac{FN}{FN+TP}$, $P_{fp}=\frac{FP}{FP+TN}$, $TPR=1-P_{fn}$, $TNR=1-P_{fp}$. Pfn/Pfp usano **una classe alla volta** → indipendenti dalle proporzioni di classe. $\text{err}=P_{fp}(1-\pi)+P_{fn}\pi$ → con prior reale $\pi$ stimo $\text{err}_{app}$ da dataset bilanciato. |
| **Costo / Bayes risk** | Costo $C(a\|k)$ (non monetario, app-dipendente). Bayes risk $B=E_{X,C\|E}[C(a(x,R)\|c)]$, non calcolabile. **Empirical Bayes risk** $B_{emp}=\sum_c \pi_c\frac1{N_c}\sum_{i:c_i=c}C(a(x_i,R)\|c)$. Serve confusion + cost matrix + prior. |
| **DCF (unnorm)** | Binario: $DCF_u=\pi_T C_{fn}P_{fn}+(1-\pi_T)C_{fp}P_{fp}$ = empirical Bayes risk. |

---

## 🟢 BLOCCO 2 — "Da costo a metrica + decisione ottima" *(candidato fresco)*

*Filo: come la Bayes risk diventa metrica operativa e come si decide. È il proseguimento naturale del blocco 1.*

| Asse | Contenuto chiave |
|---|---|
| **DCF normalizzata** | $DCF=\dfrac{DCF_u}{\min(\pi_T C_{fn},(1-\pi_T)C_{fp})}$ (÷ miglior **dummy system**). ≈1 inutile, ≈0 ottimo, >1 peggio del dummy. **Invariante allo scaling dei costi** → tutto in **prior efficace** $\tilde\pi$. Tripletta $(\pi_T,C_{fn},C_{fp})$ = working point, ridondante. |
| **Optimal Bayes decision** | Costo atteso $C_{x,R}(a)=\sum_k C(a\|k)P(C=k\|x)$, $a^*=\arg\min_a C_{x,R}(a)$. Se R=E → minimizza la Bayes risk. Regola LLR: $\log\frac{f(x\|H_T)}{f(x\|H_F)}\gtrless-\log\frac{\pi_T C_{fn}}{(1-\pi_T)C_{fp}}$. |
| **Soglia ottima (LLR calibrati)** | $t=-\log\frac{\tilde\pi}{1-\tilde\pi}$, $\tilde\pi=\frac{\pi_T C_{fn}}{\pi_T C_{fn}+(1-\pi_T)C_{fp}}$. LLR **disaccoppiano** classificatore e applicazione. |

---

## 🟢 BLOCCO 3 — "Valutazione su tutte le soglie: curve + min/actual DCF" *(candidato fresco)*

*Filo: come si valuta il classificatore su tutte le soglie e si isola la mis-calibrazione. Ponte verso la card 15.*

| Asse | Contenuto chiave |
|---|---|
| **ROC / DET / EER** | Variando soglia → (Pfn,Pfp). **ROC**: TPR vs FPR. **DET**: FNR vs FPR. **EER**: soglia dove $P_{fn}=P_{fp}$. |
| **min vs actual DCF** | **actual**: soglia teorica $t$. **min**: minimo DCF su tutte le soglie dell'eval set (qualità/discriminazione pura). **gap = actual − min = perdita per mis-calibrazione** → calibrazione (card 15). |
| **Multiclasse** *(trasversale)* | Niente prior efficace singolo, ma empirical Bayes risk + DCF norm (÷ miglior dummy) restano calcolabili. |

---

LLR = log-likelihood ratio · DCF = Detection Cost Function · EER = Equal Error Rate

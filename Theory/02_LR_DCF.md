# Dubbi e Chiarimenti — Machine Learning
> Documento di supporto agli appunti: raccoglie i fraintendimenti emersi durante la lezione e le relative spiegazioni.

---

## 1. Modelli Generativi vs Discriminativi

### Teoria
- **Generativo**: modella P(X|C) — impara come sono distribuiti i dati per ogni classe, poi usa Bayes per ricavare P(C|X). Può generare nuovi campioni.
- **Discriminativo**: modella direttamente P(C|X) — impara il confine tra le classi senza mai modellare i dati.

### Dubbio
*"Cosa cambia tra modellare P(X|C) e P(C|X)? Non sta comunque modellando una distribuzione?"*

### Risposta
Sì, entrambi modellano una distribuzione, ma su cose diverse. P(X|C) risponde a "che aspetto hanno i dati della classe C?" — parti dalla classe e immagini i dati. P(C|X) risponde a "data questa X, a quale classe appartiene?" — parti dai dati e cerchi la classe. Il generativo fa il giro lungo (P(X|C) → Bayes → P(C|X)), il discriminativo ci arriva diretto. La differenza pratica: solo il generativo sa come sono fatti i dati e può generarne di nuovi.

---

## 2. Come Leggere la Probabilità Condizionata

### Teoria
P(A|B) = "probabilità di A, **dato** B". Quello a destra del | è ciò che già conosci.

### Dubbio
*"Come faccio mentalmente a tradurre P(X|C) vs P(C|X)?"*

### Risposta
Guarda cosa sta a destra del |:
- **P(X|C)**: so già la classe → mi chiedo che dati aspettarmi → **generativo**
- **P(C|X)**: ho in mano i dati → mi chiedo la classe → **discriminativo**

Il Teorema di Bayes non fa altro che invertire la condizione: `P(C|X) = P(X|C)·P(C) / P(X)`

---

## 3. LDA — Generativo o Discriminativo?

### Teoria
LDA può essere derivato in due modi che arrivano allo stesso risultato:
- **Derivazione generativa**: assume P(X|C) gaussiana con stessa Σ → applica Bayes
- **Derivazione discriminativa** (quella del prof): cerca la direzione w che massimizza `sB/sW` — nessuna assunzione sulla forma dei dati

### Dubbio
*"LDA assume gaussiane quindi è generativo? E come si collega a Logistic Regression?"*

### Risposta
Nel contesto del corso LDA è **discriminativo** — la derivazione del prof non usa mai Bayes né P(X|C). Media e covarianza sono usate come statistiche descrittive per trovare la direzione di massima separazione, non come parametri di un modello probabilistico. La nota "assume Gaussian" nei pro/cons significa solo che LDA *funziona meglio* se i dati sono gaussiani, non che li modella. LDA e LR producono entrambi un confine lineare, ma LDA lo trova tramite criterio geometrico (massimizza sB/sW), LR ottimizzando direttamente la likelihood di P(C|X).

---

## 4. Linearità e Non-linearità nella Logistic Regression

### Teoria
$$P(C=1|X) = \sigma(w^Tx + b) = \frac{1}{1+e^{-w^Tx}}$$

### Dubbio
*"La sigmoide aggiunge non-linearità al confine? LR è lineare o non lineare?"*

### Risposta
La LR è un **classificatore lineare**. Il confine decisionale (dove P=0.5, cioè wᵀx+b=0) è un iperpiano — lineare. La sigmoide non aggiunge non-linearità al confine: serve solo a trasformare uno score reale (-∞,+∞) in una probabilità (0,1). Stesso ragionamento per il Gaussian Classifier con tied covariance: il LLR è lineare in x, la non-linearità non c'è.

---

## 5. Log-Likelihood Ratio (LLR)

### Teoria
$$\text{llr}(x) = \log \frac{P(X=x|C=h_1)}{P(X=x|C=h_0)}$$

Score su (-∞,+∞): positivo → favorisce h1, negativo → favorisce h0. La decisione è `llr(x) ≷ soglia`.

### Dubbio
*"Cos'è l'LLR e perché è utile separarlo dal prior?"*

### Risposta
L'LLR misura quale classe spiega meglio il dato x, senza coinvolgere il prior. Il prior entra solo nella soglia di decisione. Questo permette di usare lo stesso modello in applicazioni diverse (con prior diversi) cambiando solo la soglia, senza riaddestrare. Il logaritmo serve per stabilità numerica e per trasformare il rapporto di gaussiane in una forma lineare in x.

### Approfondimento Gemini:

> ### **Il Log-Likelihood Ratio (LLR)**
> 
> L'**LLR** è uno strumento potentissimo proprio perché separa l'evidenza statistica "pura" (fornita dai dati) dalle assunzioni a priori o dai costi di errore, che possono variare da un caso all'altro.
>
> #### **Cosa significa l'acronimo?**
> **LLR** sta per **Log-Likelihood Ratio** (in italiano: **Rapporto di Log-Verosimiglianza**).
>
> #### **Scomposizione del termine**
> Per capire meglio perché è così utile, analizziamone i componenti:
> *   **Likelihood (Verosimiglianza):** Indica quanto è probabile osservare il dato $x$ se fosse vera l'ipotesi $H_1$ rispetto all'ipotesi $H_0$.
> *   **Ratio (Rapporto):** Si calcola come $\frac{P(x|H_1)}{P(x|H_0)}$. Se il rapporto è $> 1$, il dato "pende" verso la prima ipotesi.
> *   **Log (Logaritmo):** Si applica il logaritmo naturale ($\ln$) per due motivi principali:
>     *   **Stabilità numerica:** Trasforma moltiplicazioni di probabilità molto piccole in somme, evitando errori di calcolo.
>     *   **Semplificazione matematica:** Il logaritmo "annulla" l'esponenziale delle distribuzioni Gaussiane, trasformando i calcoli in forme lineari molto più semplici.
>
> #### **La soglia di decisione**
> Il bello dell'LLR è che la regola di decisione diventa una semplice sottrazione:
>
> $$\text{LLR}(x) = \ln P(x|H_1) - \ln P(x|H_0)$$
>
> Per decidere, confronti il risultato con una soglia $\tau$:
> *   Se $\text{LLR}(x) > \tau \rightarrow$ Scegli **$H_1$**
> *   Se $\text{LLR}(x) < \tau \rightarrow$ Scegli **$H_0$**
>
> In questa soglia $\tau$ andrai a "nascondere" il **prior** (le tue conoscenze pregresse) e il **costo degli errori**. Se l'applicazione cambia, sposti solo $\tau$ senza dover ricalcolare o riaddestrare il modello.

---

## 6. Distribuzione di Bernoulli nella LR

### Teoria
$$P(X=x) = p^x(1-p)^{1-x}, \quad x \in \{0,1\}$$

### Dubbio
*"Perché appare Bernoulli nella Logistic Regression?"*

### Risposta
La classe C è binaria (0 o 1) e P(C=1|x) = σ(wᵀx+b) — esattamente la struttura di una Bernoulli con p = σ(wᵀx+b). Riconoscerla permette di scrivere la likelihood dell'intero dataset in forma compatta come prodotto di Bernoulli, prendere il log e ottenere la binary cross-entropy che si minimizza con gradient descent.

---

## 7. MLE vs Gradient Descent

### Teoria
- **MLE**: obiettivo — trova i parametri che massimizzano la likelihood
- **Gradient Descent**: metodo numerico per raggiungere quell'obiettivo

### Dubbio
*"Qual è la differenza tra MLE e gradient descent? Quando si usa uno o l'altro?"*

### Risposta
MLE è l'obiettivo, gradient descent è lo strumento. Per il Gaussian Classifier MLE ha soluzione analitica diretta (derivate=0, risolvi, hai μ* e Σ*). Per la LR la loss non ha forma chiusa → non puoi risolvere analiticamente → usi gradient descent per muoverti iterativamente verso il minimo. Per gli LLM è lo stesso ma con miliardi di parametri.

---

## 8. Cross-Entropy e Funzione di Loss

### Teoria
$$H(P,Q) = -\mathbb{E}_{P(x)}[\log Q(x)] = -\sum_{x} P(x)\log Q(x)$$

Nel caso binario: `H(cᵢ, yᵢ) = -[cᵢ log yᵢ + (1-cᵢ) log(1-yᵢ)]`

### Dubbio
*"Cosa sono P e Q nel contesto della LR? Come si collega a Evaluator e Recognizer?"*

### Risposta
- **P = Evaluator E** = etichette vere = distribuzione "dura" (0 o 1)
- **Q = Recognizer R** = predizioni del modello = distribuzione "morbida" (σ(wᵀx+b) ∈ (0,1))

Minimizzare H(P,Q) significa fare in modo che le probabilità predette dal modello si avvicinino il più possibile alle etichette reali. Massimizzare la likelihood ≡ minimizzare la cross-entropy — sono la stessa cosa con segno cambiato.

---

## 9. Logistic Loss e il Prodotto zᵢsᵢ

### Teoria
Recodifica: zᵢ ∈ {-1,+1} al posto di cᵢ ∈ {0,1}. Score: sᵢ = wᵀxᵢ+b.

$$l(x) = \log(1+e^{-x}), \quad J(w,b) = \sum_i l(z_i s_i)$$

### Dubbio
*"Perché tutti questi passaggi algebrici? Cosa rappresenta zᵢsᵢ?"*

### Risposta
I passaggi servono a unificare i due casi (cᵢ=0 e cᵢ=1) in una formula sola — più comoda per calcolare il gradiente in gradient descent. Il prodotto zᵢsᵢ è il segnale di correttezza della predizione: stesso segno → predizione giusta → zᵢsᵢ > 0 → costo basso. Segno opposto → predizione sbagliata → zᵢsᵢ < 0 → costo alto, cresce linearmente. Il punto x=0 (confine decisionale) ha costo log(2)≈0.69 — massima incertezza.

---

## 10. Classi Linearmente Separabili e Regolarizzazione

### Teoria
Se le classi sono perfettamente separabili, moltiplicare w per α>1 non cambia il confine ma riduce sempre la loss → ||w||→∞ → nessuna convergenza.

$$R(w,b) = \frac{\lambda}{2}||w||^2 + \sum_i \log(1+e^{-z_i(w^Tx_i+b)})$$

### Dubbio
*"È come il gradient exploding nelle DNN?"*

### Risposta
No — sono problemi superficialmente simili ma diversi. Il gradient exploding nelle DNN nasce dalla moltiplicazione in cascata dei gradienti durante backpropagation (problema durante l'ottimizzazione). Qui invece la loss non ha minimo finito per ragioni geometriche — w cresce monotonicamente, non oscilla. La regolarizzazione λ||w||² crea un "pavimento" che forza l'esistenza di un minimo finito. λ è un iperparametro — si sceglie via cross-validation, non con gradient descent (altrimenti la soluzione ottimale sarebbe λ=0).

---

## 11. Prior — Teoria, Empirico, Applicativo

### Teoria
Il prior P(C=c) risponde a: *"prima di guardare i dati, quanto è probabile la classe c?"*

| Tipo | Definizione | Quando si usa |
|---|---|---|
| Teorico/applicativo | La vera frequenza nel dominio reale | Ideale ma spesso ignoto |
| Empirico | `nT/(nT+nF)` dal dataset | Quando non si conosce il teorico |
| Applicativo | Prior specifico dell'applicazione target | Può differire dal training |

### Dubbio
*"Non ho chiaro il prior — come si collega ai diversi modelli?"*

### Risposta
Il prior è la conoscenza sul dominio prima di vedere i dati. Esempio: per rilevare frodi, il prior reale è 0.001 (frodi rarissime), ma il training set può essere bilanciato 50/50 artificialmente. Ogni modello gestisce il prior diversamente:

- **Generative Gaussian Model**: prior **separato** — entra solo nella soglia, puoi cambiarlo senza riaddestrare
- **Logistic Regression**: prior **mescolato dentro w e b** durante il training — se cambia, devi correggere
- **LDA discriminativo**: prior non entra esplicitamente nella derivazione

---

## 12. DCF, minDCF e Miscalibrazione

### Teoria
- **DCF**: costo reale con la soglia scelta per l'applicazione
- **minDCF**: costo minimo ottenibile provando tutte le soglie → misura la qualità intrinseca del modello
- **DCF - minDCF**: costo della miscalibrazione

### Dubbio
*"A quali modelli si applica DCF/minDCF e perché il Gaussian Model è più robusto?"*

### Risposta
Si applica a qualsiasi modello che produce uno score (LDA, Gaussian Model, LR — non PCA). Il Gaussian Model è più robusto perché il suo LLR non contiene il prior — se cambia il prior applicativo, cambi solo la soglia e DCF resta vicino a minDCF. La LR invece ha il prior mescolato dentro lo score → score spostato → DCF può divergere da minDCF.

**Le due soluzioni per la LR:**
1. Sottrai il log-odds empirico (veloce, corregge soglia ma non orientamento di w): `sllr = wᵀx + b - log(nT/nF)`
2. Prior-weighted LR (riaddestrare con prior target, corregge anche w)

---

## 13. Assunzioni dei Modelli — Confronto Finale

### Dubbio
*"Quale modello assume cosa? Quando uno è meglio dell'altro?"*

### Risposta

| | Gaussian Model | LDA (disc.) | Logistic Regression |
|---|---|---|---|
| Assunzione sui dati | P(X\|C) gaussiana | Nessuna | Nessuna |
| Assunzione sul confine | Lineare (tied Σ) | Lineare | Lineare |
| Prior | Separato, flessibile | Non esplicito | Mescolato dentro |
| Funziona male se... | Dati non gaussiani | Non sep. linearmente | Prior training ≠ applicativo |
| Può generare dati? | ✅ | ❌ | ❌ |

Il Gaussian Model è più rigido sui dati (deve essere gaussiano) ma più flessibile sul prior. La LR è più flessibile sui dati ma rigida sul prior. LDA è nel mezzo — nessuna assunzione sui dati, nessuna gestione esplicita del prior.

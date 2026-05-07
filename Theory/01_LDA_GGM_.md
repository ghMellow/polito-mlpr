# Lacune e Chiarimenti — Gaussian Classifier & Generative Models
> Documento di supporto agli appunti: raccoglie i dubbi emersi durante la lezione e le spiegazioni che li hanno risolti.

---

## 1. Discriminative vs Generative Models

**Teoria**

- **Discriminativo**: impara direttamente $P(C \mid X)$ — il confine tra le classi
- **Generativo**: modella la distribuzione congiunta $P(X, C) = P(X \mid C) \cdot P(C)$, poi usa Bayes per classificare

**Dubbio**

*"LDA e PCA sono esempi di modelli discriminativi?"*

**Risposta**

No. PCA è non supervisionato (non conosce le classi) e legato a un modello generativo (PPCA). LDA è **generativo** nella sua forma classica: stima $P(X \mid C)$ e $P(C)$ e poi usa Bayes. Viene spesso confuso per discriminativo perché massimizza la separazione tra classi, ma il meccanismo sottostante è generativo.

Esempi chiari di discriminativi: regressione logistica, SVM, reti neurali.

---

## 2. Prodotto tra feature e confronto tra classi

**Teoria**

Nel Naive Bayes con più feature si usa l'ipotesi di **indipendenza condizionale**:
$$f_{X|C}(x \mid c) = \prod_i f(x_i \mid c)$$

**Dubbio**

*"Quindi moltiplico le feature indipendenti e dopodiché sommo quando ho il confronto tra classi? Il professore mostrava vari esempi ad altezze diverse scegliendo solo femmina o maschio per questo?"*

**Risposta**

Esatto. Il prodotto combina le evidenze *dentro* una classe fissa. Il confronto sceglie la classe vincente tra tutte. La somma = 1 serve solo se vuoi una probabilità reale, non per decidere la classe.

| Operazione | Dove si applica | Perché |
|---|---|---|
| Prodotto $\prod$ | Tra le feature, dentro una classe fissa | Indipendenza condizionale |
| Confronto / max | Tra le classi | Scegliere la classe più probabile |
| Somma = 1 | Su tutte le classi | Normalizzazione di Bayes (solo se serve probabilità reale) |

Il professore fissava sempre una classe perché il prodotto ha senso solo dentro una classe fissa.

---

## 3. Versione logaritmica

**Teoria**

Il log trasforma il prodotto in somma, evitando underflow numerico:
$$\log \prod_i f(x_i \mid c) = \sum_i \log f(x_i \mid c)$$

**Dubbio**

*"Come funziona il confronto tra classi se tutto diventa somma?"*

**Risposta**

Il confronto non cambia: si calcolano due somme separate (una per classe) e si prende il max:
$$\hat{c} = \arg\max_c \left[ \sum_i \log f(x_i \mid c) + \log P(C=c) \right]$$

Il log è una funzione monotona crescente — se $a > b$ allora $\log a > \log b$ — quindi il confronto dà sempre lo stesso risultato della versione senza log. La normalizzazione serve solo se vuoi una probabilità interpretabile, non per decidere.

---

## 4. Gaussian Classifier — la formula log-gaussiana

**Teoria**

$$\log \mathcal{N}(x \mid \mu, \Sigma) = -\frac{D}{2}\log 2\pi - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)$$

I tre termini sono: costante (uguale per tutte le classi, si cancella nel confronto), forma della distribuzione, distanza di Mahalanobis dal centro.

**Dubbio**

*"È una formula nuova o è l'assunzione che abbiamo fatto all'inizio?"*

**Risposta**

Non è una formula nuova. È semplicemente il logaritmo della gaussiana multivariata — l'assunzione $X \mid C \sim \mathcal{N}(\mu_c, \Sigma_c)$ scritta in forma esplicita. Il termine dominante è la distanza di Mahalanobis: vince la classe il cui centro $\mu_c$ è più vicino a $x$, pesando anche la forma della distribuzione.

---

## 5. Sufficient Statistics

**Teoria**

La log-likelihood dipende dai dati solo attraverso tre quantità:

| Statistica | Formula | Serve per |
|---|---|---|
| $Z_c = N_c$ | Numero campioni della classe $c$ | Stimare $P(C=c)$ |
| $F_c = \sum x_i$ | Somma dei campioni | Stimare $\mu_c$ |
| $S_c = \sum x_i x_i^T$ | Somma prodotti esterni | Stimare $\Sigma_c$ |

**Dubbio**

*"Cosa devo riportare di importante? Il professore ha detto che non ce le chiede..."*

**Risposta**

Le sufficient statistics sono il *perché* bastano media e covarianza — non serve memorizzare la derivata. Il filo logico che conta è: assunzione gaussiana → classificatore argmax della log-gaussiana → per usarlo stimi $\mu_c$ e $\Sigma_c$ dal training set → bastano le sufficient statistics. Le formule esplicite si citano ma non si derivano.

---

## 6. Classificazione binaria e log-ratio

**Teoria**

Nel caso binario $C \in \{h_1, h_0\}$ le due probabilità sommano a 1, quindi:
$$\log r(x) = \log \frac{P(C=h_1 \mid x)}{P(C=h_0 \mid x)}$$

**Dubbio**

*"Sta riconfermando che usa la somma sapendo che fa 1 per confrontare le due classi?"*

**Risposta**

Sì. Nel caso binario conoscere una probabilità equivale a conoscere l'altra. Il rapporto è più comodo perché confrontare con zero è più semplice che confrontare con uno: $\log r > 0$ → classe $h_1$, $\log r < 0$ → classe $h_0$. Non è una cosa nuova — è lo stesso confronto tra classi riscritto in forma compatta.

---

## 7. Tied Covariance, LDA e QDA

**Teoria**

Ogni classe ha il suo $\mu_c$ ma tutte condividono la stessa $\Sigma$:
$$f_{X|C}(x \mid c) = \mathcal{N}(x \mid \mu_c, \Sigma)$$

| Modello | Covarianza | Parametri | Confine |
|---|---|---|---|
| QDA (MVG) | $\Sigma_c$ libera per classe | $O(K \cdot p^2)$ | Quadratico |
| LDA (Tied) | $\Sigma$ unica condivisa | $O(p^2)$ | Lineare |
| Naive Bayes | $\Sigma$ diagonale | $O(K \cdot p)$ | Lineare |

**Dubbio**

*"Perché usare la tied e non le altre?"*

**Risposta**

- vs QDA: troppi parametri con $p$ grande — con immagini 100×100 si arriva a miliardi di parametri, impossibile stimarli bene con pochi dati
- vs Naive: Naive ignora le correlazioni tra feature — se le feature sono correlate (es. pixel vicini) le performance calano
- Tied: via di mezzo — cattura le correlazioni ma con una sola $\Sigma$ stimata usando tutti i dati di tutte le classi → stima più robusta

**Dubbio**

*"Cos'è il class-independent noise e perché giustifica la tied?"*

**Risposta**

Il modello $x = \mu_c + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \Lambda^{-1})$ dice che ogni campione è il centro della classe più un rumore uguale per tutte le classi (es. illuminazione, posa nel face recognition). Poiché il rumore non dipende da $c$, la forma della distribuzione è uguale per tutte → $\Sigma$ condivisa. È la giustificazione fisica della tied covariance. In pratica $\Sigma$ empirica contiene segnale e rumore mescolati — il modello del rumore è la motivazione concettuale, non un modo per separarli.

---

## 8. LDA = MVG Tied — dimostrazione

**Teoria**

Calcolando il log-likelihood ratio nel caso tied, i termini $-\frac{D}{2}\log 2\pi$ e $-\frac{1}{2}\log|\Sigma|$ si cancellano (uguali al numeratore e denominatore). Rimane:
$$llr(x) = x^T b + c \quad \text{con} \quad b = \Lambda(\mu_1 - \mu_0)$$

**Dubbio**

*"Perché LDA è lineare e QDA quadratico?"*

**Risposta**

Con tied: $\Lambda_1 = \Lambda_0$ → il termine quadratico $x^T A x$ con $A = -\frac{1}{2}(\Lambda_1 - \Lambda_0)$ si annulla → confine lineare. Con QDA: $\Sigma_c$ diverse → i termini $\log|\Sigma_c|$ non si cancellano → rimane $x^T A x$ → confine quadratico. Geometricamente: classi con stessa forma → confine è una retta. Classi con forme diverse → la zona di intersezione è una curva.

---

## 9. T di Student vs Gaussiana

**Teoria**

| | Gaussiana $\mathcal{N}(\mu, \sigma^2)$ | T di Student $\mathcal{T}(\mu, \sigma^2, \nu)$ |
|---|---|---|
| Code | Decadimento esponenziale | Decadimento polinomiale (heavy tails) |
| Outlier | Molto sensibile | Robusta |
| Parametri | $\mu, \sigma^2$ | $\mu, \sigma^2, \nu$ (gradi di libertà) |

**Dubbio**

*"Cosa significa che la T non decade come fa la gaussiana?"*

**Risposta**

La gaussiana assegna probabilità quasi zero a valori lontani dalla media → un outlier può distruggere la stima di $\mu$ e $\Sigma$. La T di Student assegna ancora probabilità ragionevole ai punti lontani → gli outlier influenzano meno il modello. Si può usare in sostituzione della gaussiana in tutti e tre i modelli (QDA, LDA, Naive) pagando un parametro extra $\nu$. Quando $\nu \to \infty$ la T converge alla gaussiana.

---

## 10. Connessione con CNN (Deep Models)

**Dubbio**

*"CNN e MVG Tied fanno la stessa cosa? Anche loro calcolano una proiezione lineare aggiornando un'unica matrice di pesi."*

**Risposta**

No. La differenza chiave è la non linearità. MVG Tied fa una sola trasformazione lineare nello spazio originale. CNN invece applica trasformazioni lineari seguite da ReLU in cascata — ogni layer ha la sua matrice, non una condivisa. La ReLU permette di imparare rappresentazioni sempre più astratte, cosa impossibile per MVG.

CNN non risolve confini complessi con un confine quadratico: li aggira trasformando lo spazio finché un confine lineare basta. L'ultimo layer di una CNN fa sostanzialmente una classificazione lineare simile a MVG nello spazio trasformato.

```
Confine lineare            → LDA / Tied
Confine quadratico         → QDA / MVG
Confine non lineare        → CNN (trasforma lo spazio)
```

---

*Documento generato a partire dalla discussione in chat — lezione del 09-10 Aprile 2026.*

# Summary Card — Linear Discriminant Analysis (LDA)

*(Rigenerata come testo da `_exam/02_LDA.tex` per il confronto Prompt 2. Formato indicizzato per asse d'esame.)*

| Asse | Contenuto chiave |
|---|---|
| **Motivation / goal** | Tecnica **lineare supervisionata** per dimensionality reduction + classificazione. Trova direzioni che massimizzano la separazione tra classi rispetto alla dispersione interna. A differenza di PCA usa le label. |
| **Mathematical formulation** | Fisher ratio $L(\mathbf{w}) = \dfrac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$. Scatter: $S_B=\frac1N\sum_c n_c(\boldsymbol\mu_c-\boldsymbol\mu)(\cdot)^T$ (between), $S_W=\frac1N\sum_c\sum_i(\mathbf{x}_{c,i}-\boldsymbol\mu_c)(\cdot)^T$ (within). |
| **Training** | Problema agli autovalori generalizzato $S_W^{-1}S_B\mathbf{w}=\lambda\mathbf{w}$; $\lambda=L(\mathbf{w})$ → si prende l'autovettore col $\lambda$ massimo. Multi-direzione: $\max_W \mathrm{Tr}((W^TS_WW)^{-1}(W^TS_BW))$. |
| **Inference** | Proiezione $\tilde{\mathbf{x}}=W^T\mathbf{x}$. Caso binario in forma chiusa: $\mathbf{w}\propto S_W^{-1}(\boldsymbol\mu_2-\boldsymbol\mu_1)$. |
| **Decision rule (binaria)** | Assegna $C_1$ se $\mathbf{w}^T\mathbf{x}_t < t$, con soglia **midpoint** $t=\frac{m_1+m_2}{2}$, $m_c=\mathbf{w}^T\boldsymbol\mu_c$. Equivale a un nearest-centroid nello spazio proiettato. Ottima con priori uguali; se i priori differiscono la soglia va spostata. |
| **Decision rule (multiclasse)** | LDA usata come **riduzione di dimensionalità** (≤ $K-1$ direzioni), poi un classificatore a valle. |
| **Assumptions** | Distribuzioni intra-classe **gaussiane con stessa covarianza**; soglia midpoint ottima sotto priori uguali. |
| **Variants** | Forma chiusa binaria vs eigen multiclasse; pipeline **PCA→LDA**. |
| **Limitations** | Assunzione gaussiana a covarianza comune spesso violata; $S_W$ singolare se $N<D$; al più $K-1$ direzioni discriminanti. |

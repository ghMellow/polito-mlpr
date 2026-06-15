# Summary Card — Generative Gaussian Models / Multivariate Gaussian classifier (MVG)

*(Rigenerata come testo da `_exam/04_GGM.tex` per il confronto Prompt 2. Formato indicizzato per asse d'esame.)*

| Asse | Contenuto chiave |
|---|---|
| **Motivation / goal** | Classificazione **generativa** closed-set: si modella la densità class-conditional e si applica Bayes per il posterior. Sfrutta le proprietà della gaussiana per regole trattabili. |
| **Mathematical formulation** | $(X\mid C=c)\sim\mathcal{N}(\boldsymbol\mu_c,\boldsymbol\Sigma_c)$ <br> Posterior $\propto f_{X\mid C}(\mathbf{x}\mid c)P(C=c)$ <br> Log-lik: $-\frac{D}{2}\log2\pi-\frac12\log\lvert\boldsymbol\Sigma_c\rvert-\frac12(\mathbf{x}-\boldsymbol\mu_c)^T\boldsymbol\Sigma_c^{-1}(\mathbf{x}-\boldsymbol\mu_c)$ |
| **Training** | ML **classe per classe**: <br> $\boldsymbol\mu_c^*=\frac1{N_c}\sum_{i:c_i=c}\mathbf{x}_i$ <br> $\boldsymbol\Sigma_c^*=\frac1{N_c}\sum_{i:c_i=c}(\mathbf{x}_i-\boldsymbol\mu_c^*)(\cdot)^T$ |
| **Inference** | Binario: LLR vs soglia da log-odds, $\mathrm{llr}(\mathbf{x}_t)=\log\frac{f(\mathbf{x}_t\mid c_1)}{f(\mathbf{x}_t\mid c_0)}\gtrless-\log\frac{P(c_1)}{P(c_0)}$ <br> Multiclasse: $\arg\max_h[\log f(\mathbf{x}_t\mid h)+\log P(C=h)]$ |
| **Decision rule (binaria)** | LLR **quadratica** $\mathbf{x}^TA\mathbf{x}+\mathbf{x}^T\mathbf{b}+c$, $A=-\frac12(\boldsymbol\Lambda_1-\boldsymbol\Lambda_0)$ ($\boldsymbol\Lambda=\boldsymbol\Sigma^{-1}$) <br> Boundary **quadratico** (QDA); diventa **lineare** se $\boldsymbol\Sigma_1=\boldsymbol\Sigma_0$ (tied) |
| **Decision rule (multiclasse)** | $\arg\max$ del log-posterior, $K$ classi, nessun limite $K-1$ <br> Il prior è aggiustabile a deployment |
| **Assumptions** | Densità class-conditional **multivariata gaussiana** <br> $\boldsymbol\Sigma$ invertibile ($N>D$) |
| **Variants** | **Full (QDA)**: $\boldsymbol\Sigma_c$ per classe, boundary quadratico, molti parametri <br> **Tied**: $\boldsymbol\Sigma$ condivisa $=\frac1N\sum_c\sum_i(\mathbf{x}_i-\boldsymbol\mu_c^*)(\cdot)^T$, boundary **lineare**, più stabile <br> **Naive Bayes**: $\boldsymbol\Sigma$ diagonale (feature indipendenti) |
| **Limitations** | Assunzione gaussiana forte <br> full $\boldsymbol\Sigma$ richiede $\frac{D(D+1)}{2}$ parametri <br> serve $\boldsymbol\Sigma$ invertibile ($N>D$) → spesso PCA come preprocessing |

LLR = log likelihood ratio

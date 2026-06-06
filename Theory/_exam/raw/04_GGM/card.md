# Summary Card — Generative Gaussian Models / Multivariate Gaussian classifier (MVG)

*(Rigenerata come testo da `_exam/04_GGM.tex` per il confronto Prompt 2. Formato indicizzato per asse d'esame.)*

| Asse | Contenuto chiave |
|---|---|
| **Motivation / goal** | Classificazione **generativa** closed-set: si modella la densità class-conditional e si applica Bayes per il posterior. Sfrutta le proprietà della gaussiana per regole trattabili. |
| **Mathematical formulation** | $(X\mid C=c)\sim\mathcal{N}(\boldsymbol\mu_c,\boldsymbol\Sigma_c)$. Posterior $\propto f_{X|C}(\mathbf{x}|c)P(C=c)$. Log-lik: $-\frac{D}{2}\log2\pi-\frac12\log|\boldsymbol\Sigma_c|-\frac12(\mathbf{x}-\boldsymbol\mu_c)^T\boldsymbol\Sigma_c^{-1}(\mathbf{x}-\boldsymbol\mu_c)$. |
| **Training** | ML **classe per classe**: $\boldsymbol\mu_c^*=\frac1{N_c}\sum_{i:c_i=c}\mathbf{x}_i$, $\boldsymbol\Sigma_c^*=\frac1{N_c}\sum_{i:c_i=c}(\mathbf{x}_i-\boldsymbol\mu_c^*)(\cdot)^T$. |
| **Inference** | Binario: LLR vs soglia da log-odds, $\mathrm{llr}(\mathbf{x}_t)=\log\frac{f(\mathbf{x}_t|c_1)}{f(\mathbf{x}_t|c_0)}\gtrless-\log\frac{P(c_1)}{P(c_0)}$. Multiclasse: $\arg\max_h[\log f(\mathbf{x}_t|h)+\log P(C=h)]$. |
| **Decision rule (binaria)** | LLR **quadratica** $\mathbf{x}^TA\mathbf{x}+\mathbf{x}^T\mathbf{b}+c$, $A=-\frac12(\boldsymbol\Lambda_1-\boldsymbol\Lambda_0)$ ($\boldsymbol\Lambda=\boldsymbol\Sigma^{-1}$). Boundary **quadratico** (QDA); diventa **lineare** se $\boldsymbol\Sigma_1=\boldsymbol\Sigma_0$ (tied). |
| **Decision rule (multiclasse)** | $\arg\max$ del log-posterior, $K$ classi, nessun limite $K-1$. Il prior è aggiustabile a deployment. |
| **Assumptions** | Densità class-conditional **multivariata gaussiana**; $\boldsymbol\Sigma$ invertibile ($N>D$). |
| **Variants** | **Full (QDA)**: $\boldsymbol\Sigma_c$ per classe, boundary quadratico, molti parametri. **Tied**: $\boldsymbol\Sigma$ condivisa $=\frac1N\sum_c\sum_i(\mathbf{x}_i-\boldsymbol\mu_c^*)(\cdot)^T$, boundary **lineare**, più stabile. **Naive Bayes**: $\boldsymbol\Sigma$ diagonale (feature indipendenti). |
| **Limitations** | Assunzione gaussiana forte; full $\boldsymbol\Sigma$ richiede $\frac{D(D+1)}{2}$ parametri; serve $\boldsymbol\Sigma$ invertibile ($N>D$) → spesso PCA come preprocessing. |

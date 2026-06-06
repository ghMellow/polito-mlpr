# Summary Card — Support Vector Machines (SVM)

*(Final summary card del Prompt 1, formato indicizzato per asse d'esame — input riusabile dal Prompt 2. Confronto canonico con [[06_LR]].)*

| Asse | Contenuto chiave |
|---|---|
| **Motivation / goal** | Classificatore binario **non probabilistico** che dà interpretazione **geometrica** alla regolarizzazione di LR: tra gli infiniti iperpiani separatori sceglie quello a **margine massimo**. Permette separazione **non-lineare senza espansione esplicita** (kernel). Output NON è un posterior. |
| **Mathematical formulation** | Margine $=\min_i d(\mathbf{x}_i)$, $d=\frac{|\mathbf{w}^T\mathbf{x}+b|}{\|\mathbf{w}\|}$, encoding $z_i\in\{\pm1\}$. Forma canonica (vincolo attivo $\min_i z_i(\mathbf{w}^T\mathbf{x}_i+b)=1$): **primal** $\min\frac12\|\mathbf{w}\|^2$ s.t. $z_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1$ → **QP convesso**. |
| **Training** | Via **Lagrangiana → duale**: $\max_{\boldsymbol\alpha}\boldsymbol\alpha^T\mathbf{1}-\frac12\boldsymbol\alpha^T\mathbf{H}\boldsymbol\alpha$, $H_{ij}=z_iz_j\mathbf{x}_i^T\mathbf{x}_j$; s.t. $\sum_i\alpha_iz_i=0$ e $\alpha_i\ge0$ (hard) / $0\le\alpha_i\le C$ (soft). $\mathbf{w}=\sum_i\alpha_iz_i\mathbf{x}_i$. **KKT + complementary slackness** $\alpha_i[z_i(\mathbf{w}^T\mathbf{x}_i+b)-1]=0$ → **support vectors** (solo punti sul margine hanno $\alpha_i\ne0$). Duality gap $=0$ all'ottimo. **Niente forma chiusa**. |
| **Inference** | Score primal $s=\mathbf{w}^T\mathbf{x}_t+b$ ($O(D)$); dual $s=\sum_{i:\alpha_i>0}\alpha_iz_i\,k(\mathbf{x}_i,\mathbf{x}_t)+b$ ($O(\#SV)$). Dipende **solo dai dot-product**. |
| **Decision rule (binaria)** | $s(\mathbf{x}_t)\gtrless0$, iperpiano $\perp\mathbf{w}$. Lo score **non è un LLR** → niente soglia bayesiana nativa; serve calibrazione per i posterior. |
| **Decision rule (multiclasse)** | Nativamente **binario**; estensione multiclasse **difficile** (one-vs-one / one-vs-all, fuori dalla formulazione). |
| **Assumptions** | Nessuna assunzione sulla densità delle feature; nel caso hard-margin si assume separabilità lineare (rilassata dal soft margin). Boundary lineare nello spazio (espanso) delle feature. |
| **Variants** | **Hard margin** (separabili); **Soft margin** (slack $\xi_i\ge0$, $\min\frac12\|\mathbf{w}\|^2+C\sum_i\xi_i$, box $0\le\alpha_i\le C$); **primal con hinge loss** $\max(0,1-z_is)$; **Kernel SVM** (polinomiale $(\mathbf{x}_1^T\mathbf{x}_2+1)^d$, RBF $e^{-\gamma\|\mathbf{x}_1-\mathbf{x}_2\|^2}$; Mercer); **class-balanced** $C_T=C\frac{\pi_T}{\pi_T^{emp}}$, $C_F=C\frac{\pi_F}{\pi_F^{emp}}$. |
| **Limitations** | Score **senza interpretazione probabilistica** (serve calibrazione); **non invariante** a trasformazioni affini (center+whiten); $C$/iperparametri del kernel via **cross-validation**; **multiclasse difficile**; gestione sbilanciamento solo via costi $C_i$ (sempre senza posterior). |

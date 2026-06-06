# Summary Card — Logistic Regression (LR)

*(Final summary card del Prompt 1, formato indicizzato per asse d'esame — input riusabile dal Prompt 2.)*

| Asse | Contenuto chiave |
|---|---|
| **Motivation / goal** | Classificatore **discriminativo**: modella *direttamente* il posterior $P(C\mid X)$ senza modellare $f_X(x)$. Semplice, interpretabile, posterior ben calibrati (sul prior del training). |
| **Mathematical formulation** | Log-odds lineare ⇒ $P(C{=}h_1\mid x,\mathbf{w},b)=\sigma(\mathbf{w}^Tx+b)$, $\sigma(t)=\frac1{1+e^{-t}}$. Score $s=\mathbf{w}^Tx+b=\log\frac{P(1\mid x)}{P(0\mid x)}$. Parametri $\theta=(\mathbf{w},b)$. |
| **Training** | ML sulle label $\equiv$ min **cross-entropy** $\equiv$ **empirical risk minimization**. Obiettivo: $J(\mathbf{w},b)=\sum_i\log(1+e^{-z_i s_i})$, $z_i=2c_i-1$. **Niente forma chiusa** → solver numerico (loss + gradiente). |
| **Inference** | Calcola $s=\mathbf{w}^Tx+b$; posterior $\sigma(s)$. |
| **Decision rule (binaria)** | $s\gtrless 0$ (iperpiano $\perp\mathbf{w}$). Per applicazione $(\pi_T,C)$: recalibra $s_{llr}=s-\log\frac{n_T}{n_F}$ e decidi $s_{llr}\gtrless\log\frac{\pi_T}{1-\pi_T}$. |
| **Decision rule (multiclasse)** | **Softmax**: $P(C{=}k\mid x)=\frac{e^{\mathbf{w}_k^Tx+b_k}}{\sum_j e^{\mathbf{w}_j^Tx+b_j}}$, $\arg\max_k$. Obiettivo = softmax loss (cross-entropy multiclasse); modello over-parametrizzato. |
| **Assumptions** | Log-odds lineare in $x$ (boundary lineare); osservazioni i.i.d.; **nessuna** assunzione sulla densità delle feature. |
| **Variants** | **Regularized LR** ($+\frac\lambda2\|w\|^2$); **prior-weighted LR** (per $\pi_T$ noto); **expanded-feature/quadratic LR** (feature map $\phi(x)$ → boundary non-lineari); **softmax/multinomial LR**. |
| **Limitations** | Su classi separabili la loss non ha minimo ($\|w\|\to\infty$) → serve regolarizzazione; il modello regolarizzato non è invariante a trasformazioni lineari (serve pre-processing); score legato al prior empirico (serve recalibrazione); boundary lineare salvo feature expansion (che fa esplodere la dimensione). |

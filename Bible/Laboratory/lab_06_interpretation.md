---
title: "qual è la differenza tra GDE grafico con linea unita e MLE grafico a barre con..."
source: "https://www.perplexity.ai/search/qual-e-la-differenza-tra-gde-g-wO40Y7z8RciMzrQd5c1JAg"
author:
published:
created: 2026-05-10
description: "Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question."
tags:
  - "clippings"
---

## in questo laboratorio invece continuiamo il discorso andando a provare la classificazione: per il caso multiclasse la postirior prob f(c|x) si prende guardando il max valore delle likelihood f(x|c) mentre nel caso binario si fa un rapporto e la threshold è determinata dal valore del prior (qui nel lab mette 0 ma non ho capito come la calcola? come si calcola?)

Hai capito bene la struttura generale! Chiarisco subito il punto sulla threshold nel caso binario.

## Caso multiclasse — argmax delle posterior

Hai ragione: nel caso multiclasse si calcola la posterior $P(c|x)$ per ogni classe e si assegna al campione la classe con valore massimo [lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf?AWSAccessKeyId=ASIA2F3EMEYEUZGLSWTY&Signature=rPW4608M8TAFY9UIvPVycjh77r4%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCIAPLfluf%2FxP4w9exxa03bbCbMv81tLyBpBuHp7ap68mHAiEAmMVHPmiP7J6G6HgxoROBBCrxH8LqEs8BNSJnn4JNpwUq8wQIBBABGgw2OTk3NTMzMDk3MDUiDLQMzPXJ1LnV8igg1CrQBKMCEjXeKUMdYDO%2FnO2Fv9J6cY8gl6e7W2mzvoWyMpLLPbzsCJd97T%2BxmVoDcWdteNypqFg3M5LmZo3yzvcxnxO1SnT3Yv%2Fb2df6NXn0iwFRuqIUD0ZWC%2BCLnnFP3njxl76FFmSKTHFfpVAm5U2wg%2FCJnwUESzN9x2bI5h4wowkoOMJgb%2BkWCjbUAyB4Oj9LAdDQfHNcebeQeMAo4hMkhNZmmmQPibBIh9gskrMFhcLfOiTH5n2HBh1e31YkqVUSMgDrLKuZckMCldkV1TxaeHRWPPrt5%2FFOUewbptHVuunQ8H7tuU6UalHWXeIEs7snw43ymp8GB%2BbuCYsUNOJJ4vcYB6P18u%2BvspuGdD3DqvuhEWATwWNH7Wk06shktAn27wjrUM2THDQJGbVZKQUBWAnX%2BmIWVVvimEYxaD5KRu%2FIOR1HiBR7%2B6LdYD%2BdzKubGxWNL9C7ZpXYEuIEXWGoUTWVFJUKyyHMOs6N5wcbaQjA5y9UnkQ5XhAasQYymf7nfYIcqooJIfQC0hlJVK9CLrQN0d5OaOwo1k5VUopNIbV0ZEcrFRhh%2BZP%2FE387i9MtTTV6d1ElXgguKkRTpNMTE97p%2BZJsfQ4WS5mNspuMdieciZi5huBXRq%2FzPCXEo5yQHABL6H1HOdJ8x%2Fx%2Fix9wQd8OEJkmDdnbwCaWrzjpezEAmeBWIjxt7s8NmEVO7pLyPrExijkQvpQ0tkzbDQ993mHJmaLBIEcjckYIXokkJrJjSQ1KZnnkwmh8CRT0bUUubS6I0b3rbAbKYN5AfEEFnSEwjqqD0AY6mAGHxMDMWg4l2aTPB%2BRTmTL%2F4hLm98s4NTvFfTFjSQelgdmYn2BP%2Fct60J4TVjqwDdvr8noSxAbnu7egWOxkQcOHeoNUESpO%2B6wG6%2Fy3xW1f7LjIn69gqT01smtCuRkZfjbwi42wQTe9WvCNfWZbGEmLGK2gn9a%2B8BeRv4bvSCOWr0hIS%2BUYbFOB%2BLutbOVZj%2Buyjfmr2o3rbQ%3D%3D&Expires=1778441775). In pratica:

$$
\hat{c} = \arg\max_c P(C = c | X = x_t) = \arg\max_c \, f_{X|C}(x_t|c) \cdot P(C=c)
$$

Se i prior sono uguali per tutte le classi (1/3 nel lab), il fattore $P(C=c)$ è costante e non cambia l'argmax, quindi si riduce a prendere il massimo della likelihood.

## Caso binario — LLR e threshold

Nel caso binario si calcola il **Log-Likelihood Ratio (LLR)**:

$$
s(x_t) = \log \frac{f_{X|C}(x_t|1)}{f_{X|C}(x_t|0)} = \log f_{X|C}(x_t|1) - \log f_{X|C}(x_t|0)
$$

E si confronta con una soglia $t$. La regola di classificazione ottima (derivata dal teorema di Bayes) è:

$$
\text{predict class 1 se} \quad s(x_t) \geq t, \quad \text{dove} \quad t = -\log \frac{P(C=1)}{P(C=0)}
$$

## Perché la threshold è 0 con prior uniformi?

La formula della threshold è:[lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf?AWSAccessKeyId=ASIA2F3EMEYEUZGLSWTY&Signature=rPW4608M8TAFY9UIvPVycjh77r4%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCIAPLfluf%2FxP4w9exxa03bbCbMv81tLyBpBuHp7ap68mHAiEAmMVHPmiP7J6G6HgxoROBBCrxH8LqEs8BNSJnn4JNpwUq8wQIBBABGgw2OTk3NTMzMDk3MDUiDLQMzPXJ1LnV8igg1CrQBKMCEjXeKUMdYDO%2FnO2Fv9J6cY8gl6e7W2mzvoWyMpLLPbzsCJd97T%2BxmVoDcWdteNypqFg3M5LmZo3yzvcxnxO1SnT3Yv%2Fb2df6NXn0iwFRuqIUD0ZWC%2BCLnnFP3njxl76FFmSKTHFfpVAm5U2wg%2FCJnwUESzN9x2bI5h4wowkoOMJgb%2BkWCjbUAyB4Oj9LAdDQfHNcebeQeMAo4hMkhNZmmmQPibBIh9gskrMFhcLfOiTH5n2HBh1e31YkqVUSMgDrLKuZckMCldkV1TxaeHRWPPrt5%2FFOUewbptHVuunQ8H7tuU6UalHWXeIEs7snw43ymp8GB%2BbuCYsUNOJJ4vcYB6P18u%2BvspuGdD3DqvuhEWATwWNH7Wk06shktAn27wjrUM2THDQJGbVZKQUBWAnX%2BmIWVVvimEYxaD5KRu%2FIOR1HiBR7%2B6LdYD%2BdzKubGxWNL9C7ZpXYEuIEXWGoUTWVFJUKyyHMOs6N5wcbaQjA5y9UnkQ5XhAasQYymf7nfYIcqooJIfQC0hlJVK9CLrQN0d5OaOwo1k5VUopNIbV0ZEcrFRhh%2BZP%2FE387i9MtTTV6d1ElXgguKkRTpNMTE97p%2BZJsfQ4WS5mNspuMdieciZi5huBXRq%2FzPCXEo5yQHABL6H1HOdJ8x%2Fx%2Fix9wQd8OEJkmDdnbwCaWrzjpezEAmeBWIjxt7s8NmEVO7pLyPrExijkQvpQ0tkzbDQ993mHJmaLBIEcjckYIXokkJrJjSQ1KZnnkwmh8CRT0bUUubS6I0b3rbAbKYN5AfEEFnSEwjqqD0AY6mAGHxMDMWg4l2aTPB%2BRTmTL%2F4hLm98s4NTvFfTFjSQelgdmYn2BP%2Fct60J4TVjqwDdvr8noSxAbnu7egWOxkQcOHeoNUESpO%2B6wG6%2Fy3xW1f7LjIn69gqT01smtCuRkZfjbwi42wQTe9WvCNfWZbGEmLGK2gn9a%2B8BeRv4bvSCOWr0hIS%2BUYbFOB%2BLutbOVZj%2Buyjfmr2o3rbQ%3D%3D&Expires=1778441775)

$$
t = -\log \frac{P(C=1)}{P(C=0)}
$$

Con prior **uniformi** $P(C=1) = P(C=0) = \frac{1}{2}$, il rapporto vale:

$$
\frac{P(C=1)}{P(C=0)} = \frac{1/2}{1/2} = 1
$$

E quindi:

$$
t = -\log(1) = 0
$$

Ecco perché il laboratorio dice semplicemente $t = 0$: non è una scelta arbitraria, è la conseguenza matematica diretta dell'ipotesi di prior equiprobabili. Se invece i prior fossero sbilanciati (es. $P(C=1) = 0.8$ e $P(C=0) = 0.2$), la threshold diventerebbe $t = -\log(0.8/0.2) = -\log(4) \approx -1.39$, spostando la decisione verso la classe meno frequente.[lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf?AWSAccessKeyId=ASIA2F3EMEYEUZGLSWTY&Signature=rPW4608M8TAFY9UIvPVycjh77r4%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCIAPLfluf%2FxP4w9exxa03bbCbMv81tLyBpBuHp7ap68mHAiEAmMVHPmiP7J6G6HgxoROBBCrxH8LqEs8BNSJnn4JNpwUq8wQIBBABGgw2OTk3NTMzMDk3MDUiDLQMzPXJ1LnV8igg1CrQBKMCEjXeKUMdYDO%2FnO2Fv9J6cY8gl6e7W2mzvoWyMpLLPbzsCJd97T%2BxmVoDcWdteNypqFg3M5LmZo3yzvcxnxO1SnT3Yv%2Fb2df6NXn0iwFRuqIUD0ZWC%2BCLnnFP3njxl76FFmSKTHFfpVAm5U2wg%2FCJnwUESzN9x2bI5h4wowkoOMJgb%2BkWCjbUAyB4Oj9LAdDQfHNcebeQeMAo4hMkhNZmmmQPibBIh9gskrMFhcLfOiTH5n2HBh1e31YkqVUSMgDrLKuZckMCldkV1TxaeHRWPPrt5%2FFOUewbptHVuunQ8H7tuU6UalHWXeIEs7snw43ymp8GB%2BbuCYsUNOJJ4vcYB6P18u%2BvspuGdD3DqvuhEWATwWNH7Wk06shktAn27wjrUM2THDQJGbVZKQUBWAnX%2BmIWVVvimEYxaD5KRu%2FIOR1HiBR7%2B6LdYD%2BdzKubGxWNL9C7ZpXYEuIEXWGoUTWVFJUKyyHMOs6N5wcbaQjA5y9UnkQ5XhAasQYymf7nfYIcqooJIfQC0hlJVK9CLrQN0d5OaOwo1k5VUopNIbV0ZEcrFRhh%2BZP%2FE387i9MtTTV6d1ElXgguKkRTpNMTE97p%2BZJsfQ4WS5mNspuMdieciZi5huBXRq%2FzPCXEo5yQHABL6H1HOdJ8x%2Fx%2Fix9wQd8OEJkmDdnbwCaWrzjpezEAmeBWIjxt7s8NmEVO7pLyPrExijkQvpQ0tkzbDQ993mHJmaLBIEcjckYIXokkJrJjSQ1KZnnkwmh8CRT0bUUubS6I0b3rbAbKYN5AfEEFnSEwjqqD0AY6mAGHxMDMWg4l2aTPB%2BRTmTL%2F4hLm98s4NTvFfTFjSQelgdmYn2BP%2Fct60J4TVjqwDdvr8noSxAbnu7egWOxkQcOHeoNUESpO%2B6wG6%2Fy3xW1f7LjIn69gqT01smtCuRkZfjbwi42wQTe9WvCNfWZbGEmLGK2gn9a%2B8BeRv4bvSCOWr0hIS%2BUYbFOB%2BLutbOVZj%2Buyjfmr2o3rbQ%3D%3D&Expires=1778441775)

Preparato usando Claude Sonnet 4.6 Thinking

il collegamento tra LDA e MGM sta nella formulazione giusto che la media è uguale e la varianza anche notando che trasformando la sommatoria in Nc sono equivalenti

Sì, hai centrato il punto! Il collegamento tra **LDA** e il **Tied MVG** (Multivariate Gaussian con covarianza condivisa) è proprio nella formulazione matematica. Vediamolo in dettaglio.

## La covarianza "within-class" di LDA = Σ\* del Tied MVG

In LDA, la matrice di covarianza within-class è:

$$
\Sigma_W = \frac{1}{N} \sum_c \sum_i (x_{c,i} - \mu_c^*)(x_{c,i} - \mu_c^*)^T
$$

Nel Tied MVG, il laboratorio mostra che la stessa matrice può essere riscritta come:[lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf)

$$
\Sigma^* = \frac{1}{N} \sum_c N_c \Sigma_c^*
$$

dove $\Sigma_c^*$ è la covarianza ML per la classe $c$ del modello MVG standard. Se espandi $\Sigma_c^* = \frac{1}{N_c} \sum_i (x_{c,i} - \mu_c^*)(x_{c,i} - \mu_c^*)^T$, le due espressioni sono **identiche** — il fattore $N_c$ si semplifica.

## Perché i due modelli danno lo stesso risultato?

Con prior uniformi nel caso binario, il LLR del Tied MVG diventa:

$$
s(x_t) = \log \frac{\mathcal{N}(x_t|\mu_1, \Sigma^*)}{\mathcal{N}(x_t|\mu_0, \Sigma^*)}
$$

Poiché la covarianza $\Sigma^*$ è **la stessa per entrambe le classi**, i termini di normalizzazione si cancellano e rimane solo una forma quadratica lineare in $x_t$. Questa è esattamente la **funzione discriminante lineare di Fisher** — da qui il nome LDA.[datascience-project](https://datascience-project.eu/private-data/word/50_50_ldaita.pdf) [lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf)

## In sintesi

| Aspetto | LDA | Tied MVG |
| --- | --- | --- |
| **Media per classe** | $\mu_c$ empirica | $\mu_c^*$ ML (identica) |
| **Matrice di covarianza** | $\Sigma_W$ within-class | $\Sigma^* = \frac{1}{N}\sum_c N_c \Sigma_c^*$ (identica) |
| **Regola di decisione** | funzione discriminante lineare | LLR con soglia $t=0$ |
| **Equivalenza** | ✅ con prior uniformi | ✅ con prior uniformi |

Il laboratorio ti chiede proprio di **verificare empiricamente** che le etichette predette dai due modelli coincidano, come conferma di questa equivalenza formale.[lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf)

Preparato usando Claude Sonnet 4.6 Thinking

quindi solo quando il prior è uniforme (ergo uguale a 0 o quando il valore combacia?) e il caso è binario?

Ottima domanda, facciamo chiarezza!

## L'equivalenza LDA ↔ Tied MVG vale solo con prior uniformi

Sì, l'equivalenza è soggetta a **due condizioni**:

1. **Prior uniformi** — cioè $P(C=1) = P(C=0) = \frac{1}{2}$, che porta la threshold a $t = -\log(1) = 0$
2. **Covarianza condivisa (Tied)** — il MVG deve usare la stessa $\Sigma^*$ per tutte le classi, altrimenti la regola di decisione diventa quadratica, non lineare

Il caso binario è solo il contesto in cui il laboratorio lo verifica esplicitamente, ma il concetto si estende: anche nel multiclasse con prior uniformi e covarianza tied, le frontiere di decisione sono lineari esattamente come in LDA.[lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf)

## Perché "prior uniforme" ≠ "threshold uguale a 0 sempre"

La threshold è 0 **solo perché** i prior sono uguali. Non è che "0 è un caso speciale" — è semplicemente il risultato di:

$$
t = -\log \frac{P(C=1)}{P(C=0)} = -\log \frac{0.5}{0.5} = -\log(1) = 0
$$

Se i prior fossero diversi, ad esempio $P(C=1) = 0.7$ e $P(C=0) = 0.3$, la threshold cambierebbe:

$$
t = -\log \frac{0.7}{0.3} \approx -0.847
$$

E in quel caso LDA e Tied MVG **non sarebbero più equivalenti**, perché LDA classicamente non incorpora prior sbilanciati nella sua regola di decisione standard, mentre il Tied MVG sì tramite la threshold.[lab\_06\_Generative\_Gaussian\_Models.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/146803486/e81abcb2-079d-4447-b9f6-b755371b6dcb/lab_06_Generative_Gaussian_Models.pdf)

## In sintesi

| Condizione | LDA ↔ Tied MVG equivalenti? |
| --- | --- |
| Prior uniformi + covarianza tied | ✅ Sì |
| Prior sbilanciati | ❌ No (threshold diversa) |
| MVG standard (covarianza non tied) | ❌ No (frontiera quadratica) |

Quindi la risposta breve è: **sì, solo con prior uniformi e nel caso tied**, indipendentemente dal fatto che sia binario o multiclasse — anche se nel multiclasse è più difficile da verificare direttamente.

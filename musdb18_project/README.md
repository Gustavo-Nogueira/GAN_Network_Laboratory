## Sobre o Projeto

Este projeto tem como objetivo a **reconstrução do stem vocal** de músicas a partir de misturas completas (mixture) utilizando uma Conditional Deep Convolutional Generative Adversarial Network (**cDCGAN**), com treinamento e avaliação baseados no dataset **MUSDB18**. Em outras palavras, este trabalho explora o uso de modelos generativos adversariais condicionais para aprender o mapeamento entre o espectrograma da mistura musical e o espectrograma correspondente ao vocal isolado.

## Pré-processamento

Na etapa de pré-processamento, os sinais de áudio foram convertidos do domínio do tempo para o domínio tempo–frequência por meio da STFT (Short-Time Fourier Transform). Esse processo transforma cada música em espectrogramas, que representam como as frequências do sinal variam ao longo do tempo.

## Arquitetura

A cDCGAN foi escolhida porque consegue equilibrar **boa qualidade de reconstrução** da voz com **baixo custo computacional**. Com o treinamento adversarial, a cDCGAN aprende não só a aproximar o vocal original, mas também a fazer com que ele **soe como um vocal real**, melhorando o timbre e reduzindo artefatos. O uso da condição (a mistura musical como entrada) garante que a voz gerada esteja alinhada com a música.
 
Além disso, por ser baseada em redes convolucionais, a arquitetura é mais leve e simples do que modelos mais recentes e complexos, exigindo menos poder de processamento e permitindo treinamento mais rápido.

Por último, a essa abordagem foi escolhida por também ser a mesma utilizada no projeto do dataset MNIST, assim tornando possível a análise de como esse tipo de modelo se comporta com diferentes tipos de dados, bem como permitindo o aproveitamento e aprofundamento dos conceitos vistos anteriormente.

## Ambiente e Ferramentas

Para realizar o treinamento foi utilizado uma máquina local com GPU Nvidia RTX 5050.

O projeto foi implementado em **PyTorch**, utilizado para definir as arquiteturas do gerador e do discriminador, realizar operações com tensores e executar o processo de treinamento da cDCGAN. 

## Treinamento

O treinamento  dos modelos foi realizado por meio do script python _train.py_. Para treinar, instale as dependências do arquivo _requirements.txt_, ajuste as constantes de configuração presentes na função main (caso necessário) e rode: 

```
python3 train.py
```

## Inferência

Para rodar a inferência basta instalar as dependências do _requirements.txt_ e rodar o comando de referência a seguir:

```
python3 inference.py --model models/best_model_vocals.pth --input musdb18_samples/inputs/sample_1.mp3 --output musdb18_samples/outputs/sample_1.mp3 --device cpu
```

## Métricas Utilizadas

Para avaliar a qualidade da reconstrução de vocal do Generator será utilizada a métrica SDR (Signal-to-Distortion Ratio). A SDR é usada para avaliar a qualidade de sinais reconstruídos em tarefas como **separação de fontes** e **reconstrução de áudio**. Ela mede o quanto o sinal estimado (por exemplo, o vocal reconstruído pelo modelo) se aproxima do sinal original de referência, considerando todas as distorções presentes.

Além disso, a SDR não mede apenas “o quanto sobrou de outros instrumentos”, mas sim a **distorção total** no sinal reconstruído, assim penalizando vazamento de instrumentos, ruídos, artefatos, etc.

Uma SDR alta indica que o vocal estimado está **próximo do original**, com menos interferência instrumental e menos artefatos, então quanto mais alta, maior a qualidade.

Devido às limitações de tempo (treinamento lento), foi possível treinar apenas **42 épocas** do modelo, então imagino que o modelo ainda tem margem para melhorar. Apesar disso, o resultado quanto à métrica SDR foi de **6,29 dB**, considerada como uma **qualidade razoável** segundo a literatura.

## Logs do Treinamento

O arquivo _train.log_ contém os logs do treinamento, contando com os losses e SDR de validação.

## Resultados

### Amostra 1

**Mixture:** https://github.com/Gustavo-Nogueira/GAN_Network_Laboratory/raw/refs/heads/main/musdb18_project/musdb18_samples/inputs/sample_1.mp3

**Vocal:** https://github.com/Gustavo-Nogueira/GAN_Network_Laboratory/raw/refs/heads/main/musdb18_project/musdb18_samples/outputs/sample_1.mp3

### Amostra 2

**Mixture:** https://github.com/Gustavo-Nogueira/GAN_Network_Laboratory/raw/refs/heads/main/musdb18_project/musdb18_samples/inputs/sample_2.mp3

**Vocal:** https://github.com/Gustavo-Nogueira/GAN_Network_Laboratory/raw/refs/heads/main/musdb18_project/musdb18_samples/outputs/sample_2.mp3

### Amostra 3

**Mixture:** https://github.com/Gustavo-Nogueira/GAN_Network_Laboratory/raw/refs/heads/main/musdb18_project/musdb18_samples/inputs/sample_3.mp3

**Vocal:** https://github.com/Gustavo-Nogueira/GAN_Network_Laboratory/raw/refs/heads/main/musdb18_project/musdb18_samples/outputs/sample_3.mp3

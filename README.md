# GAN_Network_Laboratory

Processo Seletivo para Desenvolvedor de Inteligência Artificial FPFTech.

## Instruções Iniciais

Para a parte do dataset MNIST, acesse a pasta mnist_project (arquivo README na raiz).

Para a parte do dataset MUSDB18, acesse a pasta musdb18_project (arquivo README na raiz).

## Discussão sobre diferenças entre modelagem generativa de imagens e áudio

Modelos generativos para imagens e para áudio lidam com tipos de informação diferentes. Imagens são dados espaciais (altura e largura), enquanto o áudio é um sinal que evolui no tempo. Mesmo quando o som é transformado em espectrograma (parecendo uma imagem), ele ainda depende da ordem temporal e da continuidade do sinal, o que torna o problema mais delicado.

Outra diferença é a percepção humana. Pequenos erros em imagens muitas vezes passam despercebidos, mas no áudio, principalmente na voz, falhas pequenas podem gerar ruídos, distorções ou um som artificial.

Por fim, imagens já estão no formato final ao serem geradas. No áudio, quando se utilizam espectrogramas, é necessário um passo adicional de reconstrução para o domínio do tempo por meio da ISTFT, o que pode introduzir erros extras. Por isso, gerar áudio de boa qualidade costuma ser mais sensível e complexo do que gerar imagens.

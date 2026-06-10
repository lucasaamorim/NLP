# Uso de Redes Neurais para Part of Speech Tagging e Parsing de Gramática de Constituintes

- **Jeremias Pinheiro de Araujo Andrade [@?]()**
- **Lucas Apolonio de Amorim [(@lucasaamorim)](https://github.com/lucasaamorim)**
- **Moisés Ferreira de Lima [(@moisesferreira123)](https://github.com/moisesferreira123)**

Este repositório contém implementações de modelos para POS Tagging e Parsing de Gramática de Constituintes de sentenças. As tags e o dataset são provenientes do Penn [Treebank](https://en.wikipedia.org/wiki/Treebank).

## Execução
Para treinar um modelo:
```bash
...
```

Para usar um modelo:
```bash
...
```

## Detalhes de Implementação

### Pré-processamento de Dados
**TODO:** Decidir se as sentenças terão algum tipo de normalização (deixar todo o texto minúsculo, remover acentos, etc.) além da adição de padding

As sentenças e outputs são normalizados para o mesmo comprimento através da concatenação de padding. No caso dos taggers, todos os inputs e outputs tem o mesmo comprimento da sentença mais longa (medido em número de palavras), já no Parser, os inputs tem esse mesmo comprimento mas o output tem o comprimento da maior Árvore Gramatical.

### Tokenização dos Dados de Entrada e Embeddings
**TODO:** Ver se vamos usar embeddings que são palavras inteiras ou subwords, caso sejam subwords, comentar sobre como foram resolvidos problemas com relação a tagging de palavras que foram quebradas em dois ou mais tokens.

Em geral, os embeddings continuam sendo treinados pelos modelos, mas são inicializados utilizando outros embeddings pré-treinados e, portanto, utilizam a tokenização proveniente dessas embeddings (com modificações eventuais para acomodar tags e possíveis palavras ausentes).

### Stack (Tecnologias)
Todos os modelos serão desenvolvidos utilizando TensorFlow/Keras em Python.

### Hiperparâmetros e Otimizadores
**TODO:** Decidir Otimizador utilizado, learning rate, loss function, batch size e critério de parada.

### Ambiente Computacional
O treinamento dos modelos foi realizado em uma máquina com as seguintes especificações:
...

## Modelos Implementados

### RNN (Moisés)
- **Tagger Baseado em RNN convencional:**
- **Tagger Baseado em LSTM:**

### Transformer (Lucas)
- **Tagger usando Encoder-Only:**
- **Parser Generativo usando Decoder-Only:**
- **Parser Generativo usando Encoder-Decoder:**

### Pré-Treinado (Jeremias)
- **Tagging usando uma LLM pré treinada (0-shot):**
- **Tagging usando uma LLM pré treinada + exemplos estáticos (few-shot):**
- **Tagging usando uma LLM pré treinada + exemplos dinâmicamente selecionados (RAG):**

## Avaliação
### POS Tagging
Para o modelos de tagging, foi utilizada a Acurácia. Também foi gerada uma Matriz de Confusão para cada modelo para permitir uma visualização geral do desempenho na forma de um mapa de calor (heatmap).

### Parsing da Gramática de Constituites
Para avaliar o desempenho do analisador sintático, são utilizadas as métricas do padrão **PARSEVAL** (através da biblioteca `evalb` ou similar em Python):
* **Brackets Precision:** Proporção de constituintes preditos pelo modelo que estão corretos de acordo com a árvore real.
* **Brackets Recall:** Proporção de constituintes da árvore real que foram identificados corretamente pelo modelo.
* **F1-Score de Constituintes:** A média harmônica entre a precisão e o recall dos constituintes.
* **Crossing Brackets:** O número médio de constituintes preditos que se cruzam/sobrepõem incorretamente com os constituintes reais.

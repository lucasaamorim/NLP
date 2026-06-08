<!-- OBS: Isso é um esboço e tudo aqui está sujeito a mudanças, sinta-se a vontade para alterar -->
<!-- Fiz em ingles pq acho menos pior doq traduzir alguns termos que eu usei para pt-br -->
# Using Neural Networks for Part of Speech Tagging

This repository has implementations for models that do Part of Speech Tagging and Constituency Grammar Parsing of sentences. Tags and Dataset are from the Penn [Treebank](https://en.wikipedia.org/wiki/Treebank).

## Implementation Details
### Data Pre-Processing
<!-- A pasta no drive está vazia, procurar pelos dados já tratados ou escrever um "parser" simples para extrair os dados relevantes -->
...

### Tokenization of the input data and Embeddings
The tokens used are the ones defined in [GloVe](https://nlp.stanford.edu/projects/glove/), as their embeddings are used to initialize the embeddings on all models, with eventual changes to also accommodate for the tags from the Penn Treebank.

In general, the embeddings are still trained by the models but are initialized using other pre-trained embeddings and therefore use the tokenization from these embeddings (with eventual changes to accomodate tags and possible missing words).

### Stack
<!-- Ir colocando as outras bibliotecas usadas ao longo do tempo -->
All models are developed using TensorFlow/Keras in Python.

### Hyperparameters and Optimizers
...

### Computational Environment
Model training was done in a Machine with the following specs
<!-- Preencher isso daqui com as especificações do hardware usado no treino -->

<!-- TODO: Elaborar uma spec curta do que vai ser implementado e como vai ser implementado -->
## Models Implemented

### RNN POS Tagger


### LSTM POS Tagger

<!-- Ou faz mais sentido Encoder Only? Discutir quais archs fazem sentido utilizar -->
### Decoder-only Transformer POS Tagger


### Encoder-Decoder Transformer Constituency Grammar Parser
For this architecture specifically 

### RAG
<!-- TODO: Decidir qual modelo usar e criar o prompt -->
While not being a model at all, we also took a mainstream, pre-trained model and tried both 0-shot prompting and few-shot prompting. Testing with both the traditional Penn Treebank tags and with random tag names to reduce the impact of training data memorization.

Also actual RAG with embedding of the sentece and searching for relevant sentences from the "traning" (non-hidden) dataset was used.

## Evaluation
<!-- Parsing da Árvore de Derivação não pode ser avaliada com as métricas tradicionais de ML, diferente de POS tagging, por isso foram usadas métricas diferentes entre as tasks --->
### POS Tagging
For Part of Speech Tagging, the eval metrics used for evaluating the models were Accuracy, Precision and F1 Score, a Confusion Matrix for Each Model was also built for a visual overview of the performance of each model as a heatmap.

### Constituency Grammar Parsing
TODO: Search for what metrics to use

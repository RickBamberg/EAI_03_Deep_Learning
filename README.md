# EAI_03 - Deep Learning

## 📚 Sobre este Módulo

Este módulo introduz **Deep Learning**, levando você de conceitos fundamentais de redes neurais até arquiteturas avançadas como CNN, RNN, LSTM e GRU. Você implementará redes neurais do zero, entenderá como funcionam internamente e aplicará em projetos reais.

## 🎯 Objetivos de Aprendizagem

Ao finalizar este módulo, você será capaz de:

- ✅ Compreender a arquitetura de Redes Neurais Artificiais (ANN)
- ✅ Implementar redes neurais com TensorFlow/Keras
- ✅ Entender funções de ativação, backpropagation e otimizadores
- ✅ Construir CNNs para classificação de imagens
- ✅ Aplicar RNNs, LSTMs e GRUs para dados sequenciais
- ✅ Visualizar o aprendizado de redes neurais
- ✅ Implementar técnicas de regularização (Dropout, L2)
- ✅ Desenvolver projetos completos de Deep Learning

## 📂 Estrutura do Módulo

```
EAI_03_Deep_Learning/
├── README.md                          ← Este arquivo
├── AGENT_CONTEXT.md                   ← Contexto técnico
│
├── Conceitos/
│   ├── README.md                      ← Conceitos teóricos
│   ├── 1_conceito_deep_learning.ipynb ← Fundamentos DL
│   ├── conceito_ann.ipynb             ← Redes Neurais Artificiais
│   ├── conceito_cnn.ipynb             ← Redes Convolucionais
│   ├── conceito_rnn.ipynb             ← Redes Recorrentes
│   ├── conceito_lstm.ipynb            ← Long Short-Term Memory
│   ├── conceito_gru.ipynb             ← Gated Recurrent Unit
│   ├── conceito_mlp_regressao.ipynb   ← MLP para Regressão
│   └── Estrutura_ANN.ipynb            ← Estrutura completa
│
├── Modelos_Base/
│   ├── README.md                      ← Templates reutilizáveis
│   ├── Estrutura_ANN.ipynb            ← Template ANN atual
│   ├── Estrutura_ANN_Ant.ipynb        ← Versão anterior
│   └── Compara_Deep.ipynb             ← Comparação de arquiteturas
│
├── Projetos_Estudos/
│   ├── README.md                      ← Projetos educacionais
│   │
│   ├── CNN_Classificador/
│   │   ├── cnn_classificador_mnist.ipynb
│   │   └── visualizacao_aprendizado_cnn.ipynb
│   │
│   ├── MNIST_Classificador/
│   │   └── mnist_classificador.ipynb
│   │
│   ├── Regressao_MLP/
│   │   └── regressao_mlp.ipynb
│   │
│   └── Regularizacao_Visual/
│       └── visualizacao_regularizacao.ipynb
│
└── Projetos_Reais/
    ├── README.md                      ← Projetos de produção
    │
    ├── ArtClassifier/                 ← Classificador de arte
    │   ├── README.md
    │   ├── AGENT_CONTEXT.md
    │   └── [app deployado...]
    │
    └── Previsao_Acoes/                ← Previsão de ações
        ├── README.md
        ├── AGENT_CONTEXT.md
        └── [app deployado...]
```

## 📖 Conteúdo Detalhado

### 🔹 Conceitos

Notebooks teóricos focados em **entender** antes de implementar.

#### **1_conceito_deep_learning.ipynb**
**Fundamentos de Deep Learning**

**Tópicos**:
- O que é uma Rede Neural Artificial (ANN)
- Arquitetura: entrada → ocultas → saída
- Fórmula do neurônio: `y = f(Σ(w·x) + b)`
- Funções de ativação (Sigmoid, Tanh, ReLU, Softmax)
- Forward propagation
- Backpropagation e gradiente descendente
- Overfitting e técnicas de regularização

**Visualizações**:
- Gráficos das 4 principais funções de ativação
- Comparação visual entre elas

**Conceitos-chave**:
- Perceptron como bloco básico
- Não-linearidade através de ativações
- Ajuste de pesos via backpropagation

---

#### **conceito_ann.ipynb**
**Redes Neurais Artificiais Detalhadas**

**Estrutura**:
- Camada de entrada (recebe dados)
- Camadas ocultas (processamento)
- Camada de saída (predição)

**Funções de ativação**:
- **Sigmoid**: `σ(x) = 1/(1+e^(-x))` → saída (0,1)
- **Tanh**: `tanh(x)` → saída (-1,1), zero-centrada
- **ReLU**: `max(0,x)` → padrão para ocultas
- **Softmax**: probabilidades para multiclasse

**Processos**:
- **Forward propagation**: dados fluem entrada→saída
- **Backpropagation**: erro propaga saída→entrada, ajusta pesos

**Overfitting**:
- Problema: modelo memoriza treino
- Soluções: Dropout, L2, EarlyStopping, mais dados

---

#### **conceito_cnn.ipynb**
**Redes Neurais Convolucionais**

**Por quê CNNs?**
- Especializadas em dados espaciais (imagens)
- Extraem padrões locais (bordas, texturas, formas)
- Preservam estrutura espacial

**Componentes**:

1. **Camada Convolucional (Conv2D)**:
   - Filtros deslizantes (kernels)
   - Parâmetros: `filters`, `kernel_size`, `activation`, `padding`
   - Exemplo: `Conv2D(32, (3,3), activation='relu')`

2. **Pooling**:
   - Reduz dimensionalidade
   - `MaxPooling2D((2,2))` → pega valor máximo de 2x2
   - Reduz overfit e custo computacional

3. **Flatten + Dense**:
   - `Flatten()` → tensor 3D → vetor 1D
   - `Dense()` → camadas totalmente conectadas

**Arquitetura típica**:
```
Input (28,28,1)
  ↓
Conv2D(32, (3,3), relu)
  ↓
MaxPooling2D((2,2))
  ↓
Conv2D(64, (3,3), relu)
  ↓
MaxPooling2D((2,2))
  ↓
Flatten()
  ↓
Dense(64, relu)
  ↓
Dense(10, softmax)
```

---

#### **conceito_rnn.ipynb**
**Redes Neurais Recorrentes**

**Para quê RNNs?**
- Dados sequenciais (texto, séries temporais, áudio)
- Mantém "memória" de estados anteriores
- Conexões recorrentes (output volta como input)

**Características**:
- Compartilhamento de pesos no tempo
- Hidden state captura contexto
- Problema: gradient vanishing em sequências longas

**Aplicações**:
- Previsão de séries temporais
- Processamento de linguagem natural
- Reconhecimento de fala

---

#### **conceito_lstm.ipynb**
**Long Short-Term Memory**

**Por quê LSTM?**
- Resolve gradient vanishing do RNN
- Mantém memória de longo prazo
- 3 gates controlam fluxo de informação

**Estrutura**:
- **Forget gate**: o que esquecer do estado anterior
- **Input gate**: o que adicionar ao estado
- **Output gate**: o que expor como output

**Quando usar**:
- Sequências longas (>50 timesteps)
- Dependências de longo alcance
- Tradução automática, geração de texto

---

#### **conceito_gru.ipynb**
**Gated Recurrent Unit**

**GRU vs LSTM**:
- Versão simplificada do LSTM
- 2 gates (update, reset) em vez de 3
- Menos parâmetros → mais rápido
- Performance similar ao LSTM

**Quando usar**:
- Datasets menores
- Recursos computacionais limitados
- Treinamento mais rápido

---

#### **conceito_mlp_regressao.ipynb**
**MLP para Regressão**

**Diferenças para classificação**:
- **Saída**: 1 neurônio sem ativação (valor contínuo)
- **Loss**: MSE (Mean Squared Error) ou MAE
- **Métricas**: MAE, RMSE

**Estrutura**:
```python
Input(shape=(n_features,))
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(1)  # sem ativação para regressão
```

**Compilação**:
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

---

### 🔹 Modelos_Base

Templates prontos para copiar e adaptar.

#### **Estrutura_ANN.ipynb**
Template completo com:
- Criação de dados sintéticos
- Arquiteturas pré-configuradas (ANN, CNN, RNN)
- Pipeline de treinamento
- Avaliação e visualização
- Callbacks (EarlyStopping, ModelCheckpoint)

#### **Compara_Deep.ipynb**
Comparação de múltiplas arquiteturas:
- MLP simples
- CNN
- RNN/LSTM
- Performance em diferentes tipos de dados

---

### 🔹 Projetos_Estudos

Projetos educacionais para praticar conceitos.

#### **CNN_Classificador/**
**Classificador MNIST com CNN**

**Notebooks**:
1. `cnn_classificador_mnist.ipynb` - Implementação completa
2. `visualizacao_aprendizado_cnn.ipynb` - Visualização de filtros

**Dataset**: MNIST (70k dígitos manuscritos 28x28)

**Arquitetura**:
- 2 blocos Conv2D + MaxPooling
- Flatten + Dense
- Softmax para 10 classes

**Conceitos aplicados**:
- Data augmentation
- Dropout
- Batch normalization

---

#### **MNIST_Classificador/**
**Classificador básico com MLP**

**Objetivo**: Baseline sem convoluções

**Comparação**: MLP vs CNN no mesmo dataset

**Resultado esperado**: CNN > MLP para imagens

---

#### **Regressao_MLP/**
**Previsão de valores contínuos**

**Aplicações**:
- Previsão de preços
- Estimativa de consumo
- Qualquer valor numérico

**Diferença**: Loss MSE, saída sem ativação

---

#### **Regularizacao_Visual/**
**Visualização de técnicas de regularização**

**Técnicas exploradas**:
- **Dropout**: desativa neurônios aleatoriamente
- **L2 Regularization**: penaliza pesos grandes
- **EarlyStopping**: para antes de overfit
- **Batch Normalization**: normaliza ativações

**Visualizações**:
- Curvas de treino/validação com e sem regularização
- Comparação visual de overfit

---

### 🔹 Projetos_Reais

Projetos completos com deployment.

#### **ArtClassifier**
Classificador de estilos artísticos.

**Ver**: [Projetos_Reais/ArtClassifier/README.md](Projetos_Reais/ArtClassifier/README.md)

---

#### **Previsao_Acoes**
Sistema de previsão de preços de ações com LSTM.

**Ver**: [Projetos_Reais/Previsao_Acoes/README.md](Projetos_Reais/Previsao_Acoes/README.md)

---

## 🚀 Como Usar Este Módulo

### Pré-requisitos

```bash
# TensorFlow/Keras
pip install tensorflow

# Utilitários
pip install numpy pandas matplotlib seaborn

# Para projetos específicos
# Ver requirements.txt de cada projeto
```

### Ordem Recomendada de Estudo

**Fase 1 - Conceitos (1 semana)**
1. `1_conceito_deep_learning.ipynb` - Base teórica
2. `conceito_ann.ipynb` - ANNs em detalhes
3. `conceito_cnn.ipynb` - CNNs
4. `conceito_rnn.ipynb` - RNNs
5. `conceito_lstm.ipynb` - LSTMs
6. `conceito_gru.ipynb` - GRUs
7. `conceito_mlp_regressao.ipynb` - Regressão com DL

**Fase 2 - Prática (1-2 semanas)**
8. Explore `Modelos_Base/` - Templates
9. Complete projetos de `Projetos_Estudos/`:
   - MNIST_Classificador (MLP)
   - CNN_Classificador (CNN + visualizações)
   - Regressao_MLP (aplicação diferente)
   - Regularizacao_Visual (evitar overfit)

**Fase 3 - Projetos Reais (2-3 semanas)**
10. Escolha um projeto de `Projetos_Reais/`
11. Complete do início ao fim
12. Deploy local ou na nuvem

### Executando os Notebooks

```bash
# Entre no diretório
cd EAI_03_Deep_Learning/Conceitos

# Inicie o Jupyter
jupyter notebook

# Ou use Google Colab para GPU grátis
```

## 💡 Conceitos-Chave Aprendidos

### Arquiteturas

| Tipo | Uso | Exemplo |
|------|-----|---------|
| **MLP** | Dados tabulares | Previsão de preços |
| **CNN** | Imagens | Classificação de imagens |
| **RNN** | Sequências curtas | Análise de sentimento |
| **LSTM** | Sequências longas | Tradução, geração de texto |
| **GRU** | Sequências (mais rápido) | Previsão de séries temporais |

### Funções de Ativação

```python
# Camadas ocultas
activation='relu'  # Padrão (rápido, efetivo)

# Saída - Classificação binária
activation='sigmoid'  # Probabilidade 0-1

# Saída - Classificação multiclasse
activation='softmax'  # Probabilidades que somam 1

# Saída - Regressão
# SEM ativação (valor contínuo)
```

### Loss Functions

```python
# Classificação binária
loss='binary_crossentropy'

# Classificação multiclasse
loss='categorical_crossentropy'  # one-hot encoded
loss='sparse_categorical_crossentropy'  # inteiros

# Regressão
loss='mse'  # Mean Squared Error
loss='mae'  # Mean Absolute Error
```

### Otimizadores

```python
# Padrão (funciona bem na maioria dos casos)
optimizer='adam'

# Outros
optimizer='sgd'  # Stochastic Gradient Descent
optimizer='rmsprop'  # Bom para RNNs
```

### Callbacks Essenciais

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Para quando validação parar de melhorar
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Salva melhor modelo
checkpoint = ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_accuracy'
)

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          callbacks=[early_stop, checkpoint])
```

### Regularização

```python
from tensorflow.keras import layers

# Dropout (desativa 50% dos neurônios)
layers.Dropout(0.5)

# L2 regularization
layers.Dense(64, activation='relu', 
             kernel_regularizer='l2')

# Batch Normalization
layers.BatchNormalization()
```

## 🔗 Conexão com Outros Módulos

### De EAI_02 (Machine Learning)
- **Pipeline similar**: dados → modelo → avaliação
- **Métricas iguais**: accuracy, precision, recall
- **Quando usar DL vs ML**:
  - DL: imagens, texto, áudio, dados complexos
  - ML: dados tabulares, interpretabilidade importante

### Para EAI_04 (NLP Clássico)
- **Embeddings**: representação vetorial de palavras
- **RNNs**: base para processamento de texto
- **LSTM**: análise de sentimento, classificação

### Para EAI_05 (NLP com Transformers)
- **Attention mechanism**: evolução do LSTM
- **BERT**: baseado em Transformers (não RNN)
- **Fine-tuning**: transfer learning de modelos pré-treinados

### Para EAI_06 (Visão Computacional)
- **CNNs**: base para detecção de objetos, segmentação
- **Transfer learning**: VGG, ResNet, EfficientNet
- **YOLO**: CNN especializada

### Para EAI_08 (MLOps)
- **Modelos deste módulo**: serão deployados
- **TensorFlow Serving**: servir modelos .h5
- **Monitoramento**: drift detection em produção

## 📝 Notas Importantes

### Boas Práticas

1. **Sempre normalize os dados**: `(X - mean) / std`
2. **Use validation split**: 10-20% para monitorar overfit
3. **Comece simples**: MLP → CNN → RNN
4. **Visualize o treinamento**: curvas de loss/accuracy
5. **Salve checkpoints**: não perca o melhor modelo
6. **Use GPU**: TensorFlow/Keras detectam automaticamente

### Armadilhas Comuns

- ❌ Não normalizar dados
- ❌ Esquecer validation split
- ❌ Rede muito complexa (overfit)
- ❌ Learning rate muito alto (não converge)
- ❌ Usar RNN para imagens (use CNN)
- ❌ Usar CNN para sequências (use RNN/LSTM)

### Debugging

**Loss não diminui**:
- Learning rate muito alto → reduza
- Dados não normalizados → normalize
- Rede muito simples → adicione camadas

**Overfit (treino 99%, val 70%)**:
- Adicione Dropout
- Use regularização L2
- Aumente dataset (data augmentation)
- Reduza complexidade da rede

**Underfitting (treino 60%, val 58%)**:
- Rede muito simples → adicione camadas
- Poucos epochs → treine mais
- Features insuficientes → feature engineering

## 🎓 Recursos Complementares

### Cursos
- **Deep Learning Specialization** - Andrew Ng (Coursera)
- **Fast.ai** - Practical Deep Learning
- **Stanford CS231n** - CNNs for Visual Recognition

### Livros
- "Deep Learning" - Goodfellow, Bengio, Courville
- "Hands-On Machine Learning" - Aurélien Géron (Part 2)
- "Deep Learning with Python" - François Chollet (criador do Keras)

### Documentação
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/) (alternativa)

### Papers Clássicos
- **LeNet** (1998) - Primeira CNN
- **AlexNet** (2012) - Venceu ImageNet
- **VGG** (2014) - Arquitetura profunda
- **ResNet** (2015) - Skip connections
- **LSTM** (1997) - Hochreiter & Schmidhuber

## ✅ Checklist de Progresso

### Conceitos
- [ ] Entendo arquitetura de ANNs
- [ ] Conheço as principais funções de ativação
- [ ] Compreendo forward/backpropagation
- [ ] Sei quando usar CNN vs RNN vs MLP
- [ ] Entendo LSTM e GRU

### Prática
- [ ] Implementei minha primeira ANN
- [ ] Treinei uma CNN para MNIST
- [ ] Usei LSTM para séries temporais
- [ ] Apliquei técnicas de regularização
- [ ] Visualizei o aprendizado da rede

### Projetos
- [ ] Completei pelo menos 2 projetos de estudos
- [ ] Explorei 1 projeto real
- [ ] Criei meu próprio projeto de DL

## 🤝 Contribuindo

Encontrou um erro ou tem uma sugestão? Abra uma issue ou envie um pull request!

---

**Próximo Módulo**: [EAI_04 - NLP Clássico](../EAI_04_NLP_Classico)

**Anterior**: [EAI_02 - Machine Learning](../EAI_02_Machine_Learning)

---

*Desenvolvido como parte do projeto "Especialista em IA"*

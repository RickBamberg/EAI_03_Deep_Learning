# Conceitos - Deep Learning

## 📌 Sobre

Esta pasta contém **notebooks conceituais** que explicam os fundamentos de Deep Learning, desde redes neurais básicas (ANN) até arquiteturas avançadas (CNN, RNN, LSTM, GRU).

**Objetivo**: Fornecer uma base sólida teórica e prática antes de aplicar Deep Learning em projetos reais.

---

## 🎯 Por Que Estudar Conceitos?

Deep Learning é mais complexo que ML clássico:
- Mais hiperparâmetros para ajustar
- Arquiteturas específicas para cada tipo de dado
- Necessidade de compreender backpropagation e gradientes
- Trade-offs entre profundidade e overfitting

**Estudar conceitos evita**:
- ❌ Usar arquiteturas erradas para o problema
- ❌ Configurações que não convergem
- ❌ Overfitting por falta de regularização
- ❌ Perda de tempo tentando cegamente

---

## 📂 Notebooks Disponíveis

### 1️⃣ **1_conceito_deep_learning.ipynb** (Fundamentos)

**Tópicos**:
- O que é Deep Learning vs Machine Learning
- Arquitetura de uma ANN (Input → Hidden → Output)
- Neurônio artificial: `y = f(Σ(w·x) + b)`
- 4 funções de ativação principais:
  - **Sigmoid**: `σ(x) = 1/(1+e^-x)` (0 a 1)
  - **Tanh**: `tanh(x)` (-1 a 1)
  - **ReLU**: `max(0, x)` (padrão para hidden layers)
  - **Softmax**: para classificação multiclasse
- Forward propagation (cálculo da saída)
- Backpropagation (ajuste de pesos)
- Overfitting e soluções (Dropout, L2, EarlyStopping)

**Para Quem**: Iniciantes em Deep Learning

**Duração**: ~30 minutos

---

### 2️⃣ **conceito_ann.ipynb** (Redes Neurais Artificiais)

**Tópicos**:
- Perceptron como unidade básica
- Camadas de entrada, ocultas e saída
- Funções de ativação e quando usar cada uma
- Forward propagation passo a passo
- Backpropagation e descida do gradiente
- Regularização:
  - **Dropout**: desativa neurônios aleatoriamente
  - **L2**: penaliza pesos grandes
  - **EarlyStopping**: para quando validação piora

**Código Exemplo**:
```python
# ANN básica com Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Para Quem**: Quem quer entender a estrutura interna de ANNs

---

### 3️⃣ **conceito_cnn.ipynb** (Redes Convolucionais)

**Tópicos**:
- Por que CNNs para imagens?
- Camadas de convolução (Conv2D)
  - Filtros/kernels (ex: 32 filtros de 3×3)
  - Stride e padding
- Camadas de pooling (MaxPooling2D)
  - Reduz dimensionalidade
  - Preserva features importantes
- Arquitetura típica:
  ```
  Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense
  ```

**Código Exemplo**:
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

**Para Quem**: Quem vai trabalhar com imagens

**Aplicações**: Classificação de imagens, detecção de objetos, segmentação

---

### 4️⃣ **conceito_rnn.ipynb** (Redes Recorrentes)

**Tópicos**:
- Por que RNNs para sequências?
- Hidden state (memória interna)
- Fórmula: `h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b)`
- Problema: vanishing gradient em sequências longas
- Aplicações:
  - Séries temporais
  - Processamento de linguagem natural (NLP)
  - Reconhecimento de fala

**Código Exemplo**:
```python
model = Sequential([
    SimpleRNN(64, input_shape=(timesteps, features), return_sequences=True),
    SimpleRNN(32),
    Dense(1)  # para regressão
])
```

**Para Quem**: Quem vai trabalhar com dados sequenciais

**Limitação**: Não funciona bem para sequências muito longas (>50 timesteps)

---

### 5️⃣ **conceito_lstm.ipynb** (Long Short-Term Memory)

**Tópicos**:
- Solução para vanishing gradient em RNNs
- 3 gates (portas):
  - **Forget gate** (f_t): o que esquecer
  - **Input gate** (i_t): o que adicionar
  - **Output gate** (o_t): o que expor
- Cell state (c_t): "esteira transportadora" de informação
- Fórmulas detalhadas de cada gate

**Quando usar LSTM?**:
- Sequências longas (>50 timesteps)
- Dependências de longo prazo são importantes
- Exemplos: tradução automática, geração de texto, previsão de séries temporais

**Código Exemplo**:
```python
model = Sequential([
    LSTM(128, input_shape=(timesteps, features), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(num_classes, activation='softmax')
])
```

**Para Quem**: Quem trabalha com NLP ou séries temporais complexas

---

### 6️⃣ **conceito_gru.ipynb** (Gated Recurrent Unit)

**Tópicos**:
- Versão simplificada do LSTM
- 2 gates (em vez de 3):
  - **Update gate** (z_t): quanto do passado manter
  - **Reset gate** (r_t): quanto do passado esquecer
- Menos parâmetros → treina mais rápido
- Performance similar ao LSTM em muitos casos

**LSTM vs GRU**:
| Aspecto | LSTM | GRU |
|---------|------|-----|
| Parâmetros | Mais | Menos |
| Velocidade | Mais lento | Mais rápido |
| Performance | Ligeiramente melhor | Similar |
| Quando usar | Dados grandes, sequências muito longas | Dados menores, protótipos rápidos |

**Código Exemplo**:
```python
model = Sequential([
    GRU(128, input_shape=(timesteps, features), return_sequences=True),
    GRU(64),
    Dense(num_classes, activation='softmax')
])
```

**Para Quem**: Quem quer alternativa mais leve ao LSTM

---

### 7️⃣ **conceito_mlp_regressao.ipynb** (MLP para Regressão)

**Tópicos**:
- Multi-Layer Perceptron aplicado a regressão
- Diferenças vs classificação:
  - **Saída**: 1 neurônio sem ativação (valor contínuo)
  - **Loss**: MSE ou MAE (não categorical_crossentropy)
  - **Métricas**: MAE, RMSE

**Código Exemplo**:
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(1)  # sem ativação para saída contínua
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

**Para Quem**: Quem vai prever valores numéricos (preços, temperaturas, etc.)

---

## 🗺️ Ordem de Estudo Recomendada

### Iniciante (Nunca viu Deep Learning)
```
1. 1_conceito_deep_learning.ipynb      (fundamentos gerais)
2. conceito_ann.ipynb                  (estrutura de redes neurais)
3. conceito_mlp_regressao.ipynb        (caso especial: regressão)
4. conceito_cnn.ipynb                  (para imagens)
5. conceito_rnn.ipynb                  (para sequências)
6. conceito_lstm.ipynb                 (sequências longas)
7. conceito_gru.ipynb                  (alternativa ao LSTM)
```

### Intermediário (Já conhece ML)
```
1. 1_conceito_deep_learning.ipynb      (revisão rápida)
2. conceito_cnn.ipynb                  (arquiteturas visuais)
3. conceito_lstm.ipynb / conceito_gru.ipynb (sequências)
4. conceito_mlp_regressao.ipynb        (se for fazer regressão)
```

### Avançado (Revisão Rápida)
```
- Pular direto para os projetos
- Consultar conceitos específicos quando necessário
```

---

## 📊 Comparação de Arquiteturas

| Arquitetura | Tipo de Dado | Quando Usar | Exemplo |
|-------------|--------------|-------------|---------|
| **ANN/MLP** | Tabular | Dados estruturados | Classificação de clientes |
| **CNN** | Imagens | Estrutura espacial | Reconhecimento de dígitos (MNIST) |
| **RNN** | Sequências curtas | Dependências temporais simples | Análise de sentimentos |
| **LSTM** | Sequências longas | Dependências de longo prazo | Tradução automática |
| **GRU** | Sequências | Alternativa mais leve ao LSTM | Previsão de vendas |

---

## 🔑 Conceitos-Chave

### Funções de Ativação

| Função | Range | Uso Principal | Vantagem |
|--------|-------|---------------|----------|
| **Sigmoid** | (0, 1) | Output binário | Probabilidades |
| **Tanh** | (-1, 1) | Hidden layers | Zero-centrado |
| **ReLU** | [0, ∞) | Hidden layers (padrão) | Rápido, evita vanishing |
| **Softmax** | (0, 1) soma=1 | Output multiclasse | Probabilidades normalizadas |

### Loss Functions

| Problema | Loss | Métrica |
|----------|------|---------|
| Classificação binária | `binary_crossentropy` | Accuracy |
| Classificação multiclasse | `categorical_crossentropy` | Accuracy |
| Regressão | `mse`, `mae` | MAE, RMSE |

### Regularização

| Técnica | Como Funciona | Quando Usar |
|---------|---------------|-------------|
| **Dropout** | Desativa neurônios aleatoriamente | Overfitting em redes densas |
| **L2 (Ridge)** | Penaliza pesos grandes | Overfitting geral |
| **EarlyStopping** | Para quando validação piora | Sempre recomendado |
| **BatchNormalization** | Normaliza ativações por batch | Treino mais estável |

---

## 💻 Código Base Reutilizável

### Template ANN/MLP
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Definir modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinar
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
```

### Template CNN
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Template LSTM
```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 🎯 Checklist de Aprendizado

### Conceitos Fundamentais
- [ ] Entendo a diferença entre ANN, CNN, RNN
- [ ] Sei quando usar cada função de ativação
- [ ] Compreendo forward e backpropagation
- [ ] Entendo o problema de vanishing gradient
- [ ] Sei como evitar overfitting

### Arquiteturas
- [ ] Sei construir uma ANN básica
- [ ] Entendo camadas Conv2D e MaxPooling
- [ ] Compreendo como LSTMs mantêm memória
- [ ] Sei a diferença entre LSTM e GRU

### Prática
- [ ] Executei todos os 7 notebooks
- [ ] Modifiquei hiperparâmetros e observei o impacto
- [ ] Apliquei conceitos em um projeto próprio

---

## 📚 Recursos Complementares

### Cursos Online
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning) - Andrew Ng
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)
- [MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com/)

### Livros
- "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Neural Networks and Deep Learning" - Michael Nielsen (gratuito online)
- "Hands-On Machine Learning" - Aurélien Géron

### Papers Fundamentais
- LeCun et al. "Gradient-Based Learning Applied to Document Recognition" (1998) - LeNet/CNN
- Hochreiter & Schmidhuber "Long Short-Term Memory" (1997) - LSTM
- Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder" (2014) - GRU

### Ferramentas
- [TensorFlow Playground](https://playground.tensorflow.org/) - Visualizar redes neurais
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Entender CNNs visualmente
- [Distill.pub](https://distill.pub/) - Artigos visuais sobre DL

---

## 🔧 Troubleshooting Comum

### Problema: Modelo não converge
**Soluções**:
- Reduzir learning rate
- Normalizar dados (StandardScaler)
- Verificar se targets estão corretos
- Simplificar arquitetura

### Problema: Overfitting (treino 95%, validação 70%)
**Soluções**:
- Adicionar Dropout
- Usar EarlyStopping
- Aumentar dados (data augmentation em imagens)
- Regularização L2

### Problema: Underfitting (treino e validação baixos)
**Soluções**:
- Aumentar capacidade da rede (mais camadas/neurônios)
- Treinar por mais épocas
- Reduzir regularização

### Problema: Vanishing gradient (LSTM não aprende)
**Soluções**:
- Verificar normalização dos dados
- Usar gradient clipping
- Reduzir número de timesteps

---

## 💡 Dicas de Estudo

1. **Execute os notebooks célula por célula**
   - Não apenas leia, execute e observe os outputs

2. **Modifique e experimente**
   - Mude número de camadas, neurônios, funções de ativação
   - Observe como as métricas mudam

3. **Visualize**
   - Use `model.summary()` para ver a arquitetura
   - Plote loss e accuracy durante treino

4. **Compare**
   - Treine mesma arquitetura com/sem Dropout
   - Compare CNN vs MLP em imagens

5. **Documente**
   - Anote o que funcionou e o que não funcionou
   - Crie seu próprio cheat sheet

---

## 🚀 Próximos Passos

Após dominar os conceitos:

1. **Ir para Modelos_Base** - Templates prontos para usar
2. **Ir para Projetos_Estudos** - Aplicações práticas guiadas
3. **Ir para Projetos_Reais** - Projetos completos deployados

---

**Lembre-se**: Deep Learning é 20% teoria, 80% experimentação. Os conceitos dão a base, mas a prática constrói a intuição!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_03*

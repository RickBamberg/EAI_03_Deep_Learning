# AGENT_CONTEXT.md - EAI_03 Deep Learning

> **Propósito**: Contexto estruturado para agentes de IA responderem questões sobre Deep Learning  
> **Última atualização**: Janeiro 2026

## RESUMO EXECUTIVO

**Módulo**: EAI_03 - Deep Learning  
**Objetivo**: Introduzir redes neurais profundas (ANN, CNN, RNN, LSTM, GRU)  
**Abordagem**: Teoria → Implementação → Visualização → Projetos  
**Nível**: Intermediário-Avançado  
**Framework**: TensorFlow/Keras  
**Bibliotecas**: tensorflow, numpy, matplotlib, seaborn

---

## ESTRUTURA DE ARQUIVOS

```
EAI_03/
├── Conceitos/              [7 notebooks teóricos]
├── Modelos_Base/           [3 templates, comparação]
├── Projetos_Estudos/       [4 projetos educacionais]
└── Projetos_Reais/         [2 projetos com deployment]
```

---

## CONCEITOS - NOTEBOOKS TEÓRICOS

### 1_conceito_deep_learning.ipynb

**Conteúdo**:

**1. ANN (Artificial Neural Network)**
```
Arquitetura: Input → Hidden Layers → Output
Neurônio: y = f(Σ(w·x) + b)
```

**2. Funções de Ativação**

| Função | Fórmula | Range | Uso |
|--------|---------|-------|-----|
| Sigmoid | `1/(1+e^(-x))` | (0, 1) | Saída binária |
| Tanh | `(e^x - e^(-x))/(e^x + e^(-x))` | (-1, 1) | Ocultas (zero-centered) |
| ReLU | `max(0, x)` | [0, ∞) | **Padrão para ocultas** |
| Softmax | `e^(x_i)/Σe^(x_j)` | (0, 1), soma=1 | Saída multiclasse |

**3. Forward Propagation**
```
Para cada camada l:
  z[l] = W[l] · a[l-1] + b[l]
  a[l] = f(z[l])
```

**4. Backpropagation**
```
Calcular gradientes da loss em relação aos pesos:
  dL/dW = dL/dz · dz/dW
Atualizar pesos:
  W_new = W_old - learning_rate * dL/dW
```

**5. Overfitting**
- **Sintoma**: treino accuracy >> val accuracy
- **Soluções**: Dropout, L2, Early Stopping, mais dados

**Visualizações**: 4 gráficos comparando ativações

**Bibliotecas**: numpy, matplotlib

---

### conceito_ann.ipynb

**Estrutura completa de ANN**:

**Camadas**:
- **Input**: recebe features
- **Hidden**: processamento (1+ camadas)
- **Output**: predição

**Perceptron**:
- Unidade básica
- Inputs → pesos → soma → ativação → output

**Funções de ativação** (mesmas de 1_conceito)

**Forward propagation**:
```
Camada por camada:
  z = w·x + b
  a = f(z)
```

**Backpropagation**:
- Erro → derivadas → atualiza pesos
- Chain rule para propagar gradientes
- Múltiplas épocas

**Overfitting**:
- **Dropout**: desativa neurônios aleatoriamente
- **L2**: penaliza pesos grandes
- **EarlyStopping**: para quando val piora

**Formato**: Markdown puro (sem código executável)

---

### conceito_cnn.ipynb

**1. O que é CNN**
- Especializada em dados espaciais (imagens)
- Filtros convolucionais extraem padrões locais
- Preserva estrutura espacial

**2. Camada Convolucional (Conv2D)**
```python
layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')
```

**Parâmetros**:
- `filters`: número de filtros (32, 64, 128...)
- `kernel_size`: tamanho do filtro (3x3, 5x5)
- `activation`: geralmente 'relu'
- `padding`: 'same' (mantém tamanho) ou 'valid' (reduz)

**Funcionamento**:
- Filtro desliza sobre imagem
- Produto escalar em cada posição
- Gera feature map

**3. Pooling (MaxPooling2D)**
```python
layers.MaxPooling2D(pool_size=(2,2))
```

**Efeito**:
- Reduz dimensionalidade 2x
- Mantém informações importantes (max)
- Reduz overfit e custo computacional

**4. Flatten + Dense**
```python
layers.Flatten()  # 3D → 1D
layers.Dense(64, activation='relu')
layers.Dense(10, activation='softmax')
```

**5. Arquitetura típica**
```
Input (28, 28, 1)
  ↓
Conv2D(32, (3,3), relu) → (26, 26, 32)
MaxPooling2D((2,2))     → (13, 13, 32)
  ↓
Conv2D(64, (3,3), relu) → (11, 11, 64)
MaxPooling2D((2,2))     → (5, 5, 64)
  ↓
Flatten()               → (1600,)
Dense(64, relu)         → (64,)
Dense(10, softmax)      → (10,)
```

**Exemplo de código completo**:
```python
from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### conceito_rnn.ipynb

**1. O que é RNN**
- Especializada em dados sequenciais
- Mantém hidden state (memória)
- Conexões recorrentes (output → input)

**Estrutura**:
```
x_t → [RNN Cell] → h_t
       ↑      ↓
       └──────┘  (hidden state recorrente)
```

**Fórmula**:
```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
y_t = W_hy · h_t
```

**Aplicações**:
- Séries temporais (previsão de ações)
- NLP (análise de sentimento)
- Reconhecimento de fala

**Problema**: 
- **Vanishing gradient**: gradientes desaparecem em sequências longas
- **Solução**: LSTM, GRU

**Exemplo Keras**:
```python
from tensorflow.keras.layers import SimpleRNN

model = Sequential([
    SimpleRNN(64, input_shape=(timesteps, features)),
    Dense(1)
])
```

---

### conceito_lstm.ipynb

**1. Por quê LSTM?**
- Resolve vanishing gradient do RNN
- Mantém memória de longo prazo
- 3 gates controlam fluxo

**2. Estrutura**

**Cell State (c_t)**:
- "Esteira transportadora" de informação
- Flui quase inalterada

**3 Gates**:

**Forget Gate** (o que esquecer):
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Input Gate** (o que adicionar):
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
```

**Output Gate** (o que expor):
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)
```

**Update Cell State**:
```
c_t = f_t * c_{t-1} + i_t * c̃_t
```

**Quando usar**:
- Sequências longas (>50 timesteps)
- Dependências de longo alcance
- Tradução, geração de texto

**Exemplo Keras**:
```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(64, input_shape=(timesteps, features), return_sequences=True),
    LSTM(32),
    Dense(1)
])
```

---

### conceito_gru.ipynb

**1. GRU vs LSTM**
- Versão simplificada do LSTM
- 2 gates em vez de 3
- Menos parâmetros → mais rápido
- Performance similar

**2. Estrutura**

**Update Gate** (quanto atualizar):
```
z_t = σ(W_z · [h_{t-1}, x_t])
```

**Reset Gate** (quanto esquecer):
```
r_t = σ(W_r · [h_{t-1}, x_t])
```

**Candidate hidden state**:
```
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
```

**Final hidden state**:
```
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

**Comparação LSTM vs GRU**:

| Aspecto | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Parâmetros | Mais | Menos |
| Velocidade | Mais lento | Mais rápido |
| Performance | Melhor em sequências muito longas | Similar na maioria |

**Quando usar GRU**:
- Datasets menores
- Recursos limitados
- Necessidade de velocidade

**Exemplo Keras**:
```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(64, input_shape=(timesteps, features)),
    Dense(1)
])
```

---

### conceito_mlp_regressao.ipynb

**MLP para Regressão**

**Diferenças para classificação**:

| Aspecto | Classificação | Regressão |
|---------|---------------|-----------|
| Saída | softmax/sigmoid | **nenhuma ativação** |
| Neurônios saída | n_classes | **1** |
| Loss | categorical_crossentropy | **mse** ou **mae** |
| Métrica | accuracy | **mae**, **rmse** |

**Estrutura**:
```python
model = Sequential([
    Input(shape=(n_features,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # SEM ativação!
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

**Loss functions**:
- **MSE**: Mean Squared Error = `(1/n) Σ(y_true - y_pred)²`
- **MAE**: Mean Absolute Error = `(1/n) Σ|y_true - y_pred|`

**Métricas**:
- **MAE**: erro médio absoluto
- **RMSE**: `√MSE` → mesma unidade do target

**Exemplo completo**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Dados de exemplo (previsão de preços)
X_train.shape  # (1000, 10) - 10 features
y_train.shape  # (1000,)    - preços contínuos

model = Sequential([
    Input(shape=(10,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # previsão de preço
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, 
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32)
```

---

## MODELOS BASE - TEMPLATES

### Estrutura_ANN.ipynb

**Template completo para novos projetos**

**Seções**:

1. **Imports**
```python
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

2. **Preparação de dados**
```python
# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape para CNN (se necessário)
X_train = X_train.reshape(-1, 28, 28, 1)
```

3. **Arquiteturas pré-configuradas**

**MLP**:
```python
model = models.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])
```

**CNN**:
```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

**LSTM**:
```python
model = models.Sequential([
    layers.LSTM(64, input_shape=(timesteps, features), return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(1)
])
```

4. **Callbacks**
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_accuracy'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)

callbacks = [early_stop, checkpoint, reduce_lr]
```

5. **Compilação**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

6. **Treinamento**
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

7. **Visualização**
```python
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.show()
```

8. **Avaliação**
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
```

---

### Compara_Deep.ipynb

**Comparação de múltiplas arquiteturas**

**Modelos comparados**:
1. MLP simples
2. MLP profundo
3. CNN
4. RNN/LSTM

**Métricas**:
- Accuracy
- Loss
- Tempo de treinamento
- Número de parâmetros

**Visualização**:
- Tabela comparativa
- Gráficos de convergência
- Matriz de confusão de cada

---

## PROJETOS ESTUDOS - DETALHES

### CNN_Classificador/

**cnn_classificador_mnist.ipynb**

**Dataset**: MNIST
- 60k treino, 10k teste
- 28x28 grayscale
- 10 classes (dígitos 0-9)

**Arquitetura**:
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

**Técnicas aplicadas**:
- Data augmentation (rotação, shift, zoom)
- Dropout
- Batch normalization (opcional)

**Resultado esperado**: ~99% accuracy

**visualizacao_aprendizado_cnn.ipynb**

**Visualizações**:
1. **Filtros da primeira camada Conv2D**
2. **Feature maps** de cada camada
3. **Ativações** para uma imagem específica
4. **Curvas de aprendizado**

**Código de visualização de filtros**:
```python
# Obter pesos da primeira camada
filters, biases = model.layers[0].get_weights()

# Normalizar filtros para visualização
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plotar
n_filters = 32
plt.figure(figsize=(15,15))
for i in range(n_filters):
    plt.subplot(6, 6, i+1)
    plt.imshow(filters[:,:,0,i], cmap='gray')
    plt.axis('off')
plt.show()
```

---

### MNIST_Classificador/

**mnist_classificador.ipynb**

**Objetivo**: Baseline com MLP (sem convoluções)

**Arquitetura**:
```python
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

**Comparação**:
- MLP: ~97-98% accuracy
- CNN: ~99% accuracy
- **Conclusão**: CNN superior para imagens

---

### Regressao_MLP/

**regressao_mlp.ipynb**

**Objetivo**: Prever valores contínuos

**Dataset típico**: Boston Housing, California Housing, ou sintético

**Arquitetura**:
```python
model = Sequential([
    Input(shape=(n_features,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # sem ativação
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

**Avaliação**:
```python
# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
```

---

### Regularizacao_Visual/

**visualizacao_regularizacao.ipynb**

**Técnicas demonstradas**:

**1. Dropout**
```python
Dense(128, activation='relu'),
Dropout(0.5),  # desativa 50%
```

**Efeito**: Reduz overfit, força redundância

**2. L2 Regularization**
```python
Dense(128, activation='relu', kernel_regularizer=l2(0.01))
```

**Efeito**: Penaliza pesos grandes

**3. Early Stopping**
```python
EarlyStopping(monitor='val_loss', patience=10)
```

**Efeito**: Para antes de overfit

**4. Batch Normalization**
```python
Dense(128, activation='relu'),
BatchNormalization(),
```

**Efeito**: Normaliza ativações, acelera treinamento

**Visualizações**:
- Curvas de treino/val com e sem regularização
- Comparação lado a lado
- Histogramas de pesos

---

## KERAS/TENSORFLOW - REFERÊNCIA RÁPIDA

### Importações
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, GRU
```

### Camadas Comuns

**Dense** (Fully Connected):
```python
layers.Dense(units=64, activation='relu', kernel_regularizer=l2(0.01))
```

**Conv2D** (Convolução 2D):
```python
layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')
```

**MaxPooling2D**:
```python
layers.MaxPooling2D(pool_size=(2,2))
```

**Flatten**:
```python
layers.Flatten()
```

**Dropout**:
```python
layers.Dropout(rate=0.5)
```

**BatchNormalization**:
```python
layers.BatchNormalization()
```

**LSTM**:
```python
layers.LSTM(units=64, return_sequences=True)
```

**GRU**:
```python
layers.GRU(units=64, return_sequences=False)
```

### Funções de Loss

**Classificação binária**:
```python
loss='binary_crossentropy'
```

**Classificação multiclasse (one-hot)**:
```python
loss='categorical_crossentropy'
```

**Classificação multiclasse (inteiros)**:
```python
loss='sparse_categorical_crossentropy'
```

**Regressão**:
```python
loss='mse'  # ou 'mae'
```

### Otimizadores

```python
optimizer='adam'  # padrão (bom na maioria)
optimizer='sgd'   # Stochastic Gradient Descent
optimizer='rmsprop'  # bom para RNNs
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
```

### Callbacks

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_accuracy')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
tensorboard = TensorBoard(log_dir='./logs')
```

### Métricas

```python
metrics=['accuracy']  # classificação
metrics=['mae']  # regressão
metrics=['mse', 'mae']  # múltiplas
```

---

## PERGUNTAS FREQUENTES - RESPOSTAS PARA AGENTES

**Q: Quando usar CNN vs RNN vs MLP?**
A: 
- **MLP**: Dados tabulares, features independentes
- **CNN**: Imagens, dados espaciais (altura x largura)
- **RNN/LSTM**: Sequências, séries temporais, texto

**Q: Por que ReLU é padrão para camadas ocultas?**
A: (1) Não satura como sigmoid/tanh, (2) Computacionalmente eficiente (max(0,x)), (3) Reduz vanishing gradient, (4) Funciona bem empiricamente.

**Q: Diferença entre Dropout e L2?**
A: 
- **Dropout**: desativa neurônios aleatoriamente durante treino, força redundância
- **L2**: adiciona termo `λ·||w||²` à loss, penaliza pesos grandes
- Dropout geralmente mais efetivo, pode usar ambos

**Q: Quando usar LSTM vs GRU?**
A: 
- **LSTM**: sequências muito longas, memória de longo prazo crítica
- **GRU**: mais rápido, menos parâmetros, performance similar na maioria
- Teste ambos, comece com GRU

**Q: Como escolher número de camadas e neurônios?**
A: 
- **Comece simples**: 1-2 camadas ocultas, 64-128 neurônios
- **Aumente gradualmente** se underfit
- **Reduza ou regularize** se overfit
- Não há fórmula mágica, depende do problema

**Q: Por que normalizar dados?**
A: (1) Gradientes mais estáveis, (2) Convergência mais rápida, (3) Evita dominância de features com escalas grandes, (4) Ativações não saturam.

**Q: Diferença entre `return_sequences=True` e `False` em LSTM?**
A:
- **True**: retorna output para cada timestep (sequence-to-sequence)
- **False**: retorna apenas último output (sequence-to-one)
- Use True se empilhar LSTMs, False na última camada

**Q: Como diagnosticar overfit?**
A: Treino accuracy >> Val accuracy. Curvas divergem. Soluções: Dropout, L2, mais dados, Early Stopping, simplificar rede.

**Q: Por que usar `sparse_categorical_crossentropy` vs `categorical_crossentropy`?**
A:
- **sparse**: labels são inteiros (0, 1, 2...)
- **categorical**: labels são one-hot encoded ([1,0,0], [0,1,0]...)
- Funcionalmente idênticos, sparse economiza memória

**Q: Diferença entre `Sequential` e `Functional API`?**
A:
- **Sequential**: modelos lineares (camada após camada)
- **Functional**: modelos com múltiplas entradas/saídas, skip connections, DAGs
- Sequential é mais simples, use quando possível

---

## DEBUGGING - PROBLEMAS COMUNS

**Loss é NaN**:
- Learning rate muito alto → reduza (1e-4, 1e-5)
- Explosão de gradientes → clip gradients
- Dados não normalizados → normalize

**Loss não diminui**:
- Learning rate muito baixo → aumente
- Rede muito simples → adicione camadas
- Features ruins → feature engineering
- Bug nos dados → verifique shapes

**Overfit severo (treino 99%, val 60%)**:
- Adicione Dropout (0.3-0.5)
- Use L2 regularization (0.01)
- Aumente dataset (data augmentation)
- Reduza complexidade da rede
- Early stopping

**Underfit (treino 65%, val 63%)**:
- Rede muito simples → adicione camadas/neurônios
- Poucos epochs → treine mais
- Learning rate muito alto → reduza
- Features insuficientes → adicione features

**LSTM/GRU muito lento**:
- Use GRU em vez de LSTM
- Reduza batch size
- Use GPU (CUDA)
- Reduza unidades LSTM

**OOM (Out of Memory)**:
- Reduza batch size
- Reduza tamanho do modelo
- Use model.fit com generator
- Use mixed precision training

---

## TAGS DE BUSCA

`#deep-learning` `#ann` `#cnn` `#rnn` `#lstm` `#gru` `#tensorflow` `#keras` `#neural-networks` `#backpropagation` `#activation-functions` `#regularization` `#dropout` `#batch-normalization` `#overfitting` `#underfitting` `#convolutional` `#recurrent` `#sequential-models` `#image-classification` `#time-series` `#mnist`

---

**Versão**: 1.0  
**Compatibilidade**: Agentes de IA com conhecimento de Deep Learning e TensorFlow/Keras  
**Uso recomendado**: Responder perguntas sobre arquiteturas, implementação, debugging, ou seleção de modelos em Deep Learning

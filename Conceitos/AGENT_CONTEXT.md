# AGENT_CONTEXT.md - Conceitos de Deep Learning

> **Propósito**: Contexto técnico estruturado para agentes sobre fundamentos de Deep Learning  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Referência conceitual completa

## RESUMO EXECUTIVO

**Objetivo**: Fornecer base teórica e matemática sólida de Deep Learning antes de implementação prática  
**Notebooks**: 7 notebooks conceituais (ANN, CNN, RNN, LSTM, GRU, MLP Regressão)  
**Público-alvo**: Iniciantes e intermediários em Deep Learning  
**Uso**: Consulta rápida de fórmulas, arquiteturas e melhores práticas

---

## ESTRUTURA DOS NOTEBOOKS

### 1. 1_conceito_deep_learning.ipynb

**Foco**: Fundamentos gerais de Deep Learning

**Conteúdo**:

#### Arquitetura de ANN
```
Input Layer → Hidden Layer(s) → Output Layer

Cada neurônio:
y = f(Σ(w·x) + b)

onde:
- w = pesos (weights)
- x = entradas (inputs)
- b = viés (bias)
- f = função de ativação
```

#### Funções de Ativação (4 principais)

**1. Sigmoid**:
```python
σ(x) = 1 / (1 + e^(-x))

Range: (0, 1)
Derivada: σ'(x) = σ(x) · (1 - σ(x))
Uso: Output de classificação binária
Problema: Vanishing gradient
```

**2. Tanh**:
```python
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Range: (-1, 1)
Derivada: tanh'(x) = 1 - tanh²(x)
Uso: Hidden layers (melhor que sigmoid)
Vantagem: Zero-centrado
```

**3. ReLU** (Rectified Linear Unit):
```python
ReLU(x) = max(0, x)

Range: [0, ∞)
Derivada: 1 se x > 0, 0 se x ≤ 0
Uso: Padrão para hidden layers
Vantagens: Rápido, evita vanishing gradient
Problema: Dead neurons (ReLU negativo sempre 0)
```

**4. Softmax**:
```python
softmax(x_i) = e^(x_i) / Σ(e^(x_j))

Range: (0, 1) com Σ = 1
Uso: Output de classificação multiclasse
Interpretação: Probabilidades normalizadas
```

#### Forward Propagation

**Fórmulas por camada**:
```python
# Camada l
z[l] = W[l] · a[l-1] + b[l]
a[l] = f(z[l])

onde:
- W[l] = matriz de pesos da camada l
- a[l-1] = ativações da camada anterior
- b[l] = vetor de bias
- f = função de ativação
```

**Exemplo (3 camadas)**:
```python
# Input → Hidden 1
z[1] = W[1] · x + b[1]
a[1] = ReLU(z[1])

# Hidden 1 → Hidden 2
z[2] = W[2] · a[1] + b[2]
a[2] = ReLU(z[2])

# Hidden 2 → Output
z[3] = W[3] · a[2] + b[3]
a[3] = softmax(z[3])  # para classificação
```

#### Backpropagation

**Objetivo**: Atualizar pesos para minimizar erro

**Fórmulas**:
```python
# 1. Calcular erro
L = loss_function(y_true, y_pred)

# 2. Gradiente da camada de saída
dL/dz[L] = a[L] - y  # para softmax + cross-entropy

# 3. Propagação para trás
dL/dz[l] = (W[l+1]^T · dL/dz[l+1]) ⊙ f'(z[l])

# 4. Gradientes dos pesos e bias
dL/dW[l] = dL/dz[l] · a[l-1]^T
dL/db[l] = dL/dz[l]

# 5. Atualização (gradient descent)
W[l] = W[l] - α · dL/dW[l]
b[l] = b[l] - α · dL/db[l]

onde:
- α = learning rate
- ⊙ = multiplicação elemento a elemento (Hadamard)
```

#### Overfitting

**Sinais**:
- Treino: 99% accuracy
- Validação: 75% accuracy
- Gap grande entre treino e validação

**Soluções**:

**1. Dropout**:
```python
from tensorflow.keras.layers import Dropout

Dense(128, activation='relu'),
Dropout(0.5),  # Desativa 50% dos neurônios
Dense(64, activation='relu')
```

**2. L2 Regularization**:
```python
from tensorflow.keras.regularizers import l2

Dense(128, activation='relu', kernel_regularizer=l2(0.01))
```

**3. EarlyStopping**:
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

---

### 2. conceito_ann.ipynb

**Foco**: Aprofundamento em Redes Neurais Artificiais

**Conteúdo Adicional**:

#### Perceptron

```python
# Perceptron simples
y = 1 if (Σ(w_i · x_i) + b) > 0 else 0

# Limitação: Apenas problemas linearmente separáveis
# Solução: Multi-Layer Perceptron (MLP)
```

#### Camadas de uma ANN

```
Input Layer:
- Tamanho = número de features
- Sem ativação

Hidden Layer(s):
- Tamanho = hiperparâmetro (ex: 128, 64, 32)
- Ativação = ReLU (padrão)

Output Layer:
- Tamanho = número de classes (classificação) ou 1 (regressão)
- Ativação = softmax (multiclasse), sigmoid (binária), nenhuma (regressão)
```

#### Regularização Detalhada

**Dropout - Como Funciona**:
```python
# Durante treino
mask = np.random.binomial(1, keep_prob, size=neurons)
activations = activations * mask / keep_prob

# Durante teste
# Usa todos os neurônios (sem dropout)
```

**L2 (Ridge) - Fórmula**:
```python
Loss_total = Loss_original + λ · Σ(w²)

# Gradiente modificado
dL/dw = dL/dw_original + 2λw

# Efeito: Pesos tendem a ficar menores
```

**BatchNormalization**:
```python
# Normaliza por batch
μ_batch = mean(x_batch)
σ_batch = std(x_batch)
x_norm = (x - μ_batch) / (σ_batch + ε)

# Aprende escala e shift
y = γ · x_norm + β
```

---

### 3. conceito_cnn.ipynb

**Foco**: Redes Convolucionais para Imagens

**Conteúdo**:

#### Por Que CNN para Imagens?

**MLPs ignoram estrutura espacial**:
```python
# MLP: Flatten image (28×28 = 784 inputs)
# Perde informação de vizinhança

# CNN: Mantém estrutura 2D
# Aprende features locais (bordas, texturas)
```

#### Camada Convolucional (Conv2D)

**Operação de Convolução**:
```python
# Filtro/Kernel desliza sobre imagem
output[i,j] = Σ Σ (input[i+m, j+n] · kernel[m,n]) + bias

# Exemplo: Filtro 3×3
kernel = [[-1, -1, -1],
          [ 0,  0,  0],
          [ 1,  1,  1]]  # Detecta bordas horizontais
```

**Parâmetros**:
```python
Conv2D(
    filters=32,          # Número de filtros (features)
    kernel_size=(3,3),   # Tamanho do filtro
    strides=(1,1),       # Passo do deslizamento
    padding='valid',     # 'valid' ou 'same'
    activation='relu',
    input_shape=(28,28,1)
)

# Output shape
# H_out = (H_in - kernel_h + 2*padding) / stride + 1
# Ex: (28 - 3 + 0) / 1 + 1 = 26
```

#### Camada de Pooling (MaxPooling2D)

**Operação**:
```python
# MaxPooling 2×2 seleciona valor máximo
[[1, 2],   → max = 4
 [3, 4]]

# Reduz dimensionalidade
# 28×28 → 14×14 (pooling 2×2)
```

**Código**:
```python
MaxPooling2D(
    pool_size=(2,2),  # Tamanho da janela
    strides=None      # Default = pool_size
)
```

#### Arquitetura CNN Típica

```python
model = Sequential([
    # Bloco 1
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    # Output: (26, 26, 32)
    MaxPooling2D((2,2)),
    # Output: (13, 13, 32)
    
    # Bloco 2
    Conv2D(64, (3,3), activation='relu'),
    # Output: (11, 11, 64)
    MaxPooling2D((2,2)),
    # Output: (5, 5, 64)
    
    # Bloco 3
    Conv2D(64, (3,3), activation='relu'),
    # Output: (3, 3, 64)
    
    # Classificador
    Flatten(),
    # Output: (3*3*64 = 576)
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

**Por que funciona**:
- Primeiras camadas: Features simples (bordas, cores)
- Camadas médias: Features intermediárias (texturas)
- Camadas finais: Features complexas (partes de objetos)

---

### 4. conceito_rnn.ipynb

**Foco**: Redes Recorrentes para Sequências

**Conteúdo**:

#### Por Que RNN?

**Problema com ANNs**:
```python
# ANN: Inputs independentes
y = f(x)

# Sequência: Inputs dependem do passado
# "O céu está ____" → precisa contexto
```

#### Arquitetura RNN

**Hidden State (Memória)**:
```python
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
y_t = W_hy · h_t + b_y

onde:
- h_t = hidden state no tempo t (memória)
- h_{t-1} = hidden state anterior
- x_t = input no tempo t
- y_t = output no tempo t
- W_hh = pesos hidden-to-hidden
- W_xh = pesos input-to-hidden
- W_hy = pesos hidden-to-output
```

**Visualização**:
```
t=0:  x_0 → [RNN] → h_0 → y_0
              ↓
t=1:  x_1 → [RNN] → h_1 → y_1
              ↓
t=2:  x_2 → [RNN] → h_2 → y_2
```

#### Problema: Vanishing Gradient

**Backpropagation Through Time (BPTT)**:
```python
# Gradiente propaga no tempo
dL/dh_1 = dL/dh_t · ∏(dh_i/dh_{i-1})

# Tanh derivada ≤ 1
# Produto de muitos valores < 1 → gradiente desaparece
# Resultado: Não aprende dependências longas (>10 steps)
```

**Solução**: LSTM ou GRU

#### Código Keras

```python
from tensorflow.keras.layers import SimpleRNN

model = Sequential([
    SimpleRNN(64, input_shape=(timesteps, features), return_sequences=True),
    SimpleRNN(32),
    Dense(num_classes, activation='softmax')
])
```

---

### 5. conceito_lstm.ipynb

**Foco**: Long Short-Term Memory

**Conteúdo**:

#### Arquitetura LSTM

**Cell State** (c_t): "Esteira transportadora" de informação

**3 Gates**:

**1. Forget Gate** (o que esquecer):
```python
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# Decide o que remover do cell state
# Valores próximos de 0 = esquecer
# Valores próximos de 1 = manter
```

**2. Input Gate** (o que adicionar):
```python
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

# i_t: Decide quais valores atualizar
# C̃_t: Novos valores candidatos
```

**3. Output Gate** (o que expor):
```python
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)

# Decide o que do cell state vira output
```

**Cell State Update**:
```python
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

# Esquece parte antiga + Adiciona parte nova
```

#### Por Que LSTM Funciona?

**Cell state flui sem multiplicações**:
```python
# RNN: h_t depende de multiplicações → vanishing gradient
# LSTM: C_t tem adições → gradiente flui melhor
```

#### Quando Usar LSTM?

- Sequências longas (>50 timesteps)
- Dependências de longo prazo importantes
- Exemplos: Tradução, geração de texto, séries temporais longas

#### Código Keras

```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
```

---

### 6. conceito_gru.ipynb

**Foco**: Gated Recurrent Unit

**Conteúdo**:

#### Diferença para LSTM

**LSTM**: 3 gates + cell state separado  
**GRU**: 2 gates + cell state = hidden state

#### Arquitetura GRU

**2 Gates**:

**1. Update Gate** (z_t):
```python
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

# Decide quanto do passado manter
# Similar a forget + input do LSTM combinados
```

**2. Reset Gate** (r_t):
```python
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

# Decide quanto do passado esquecer ao calcular novo estado
```

**Hidden State Update**:
```python
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

# Interpolação entre estado antigo e novo
```

#### LSTM vs GRU - Comparação

| Aspecto | LSTM | GRU |
|---------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (update, reset) |
| **Parâmetros** | Mais (4 matrizes de peso) | Menos (3 matrizes) |
| **Velocidade Treino** | Mais lento | Mais rápido |
| **Velocidade Inferência** | Mais lento | Mais rápido |
| **Performance** | Ligeiramente melhor | Similar |
| **Memória** | Mais | Menos |

#### Quando Usar GRU?

- Datasets menores (menos parâmetros = menos overfit)
- Prototipagem rápida
- Recursos computacionais limitados
- Performance similar ao LSTM é suficiente

#### Código Keras

```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(timesteps, features)),
    GRU(64),
    Dense(num_classes, activation='softmax')
])
```

---

### 7. conceito_mlp_regressao.ipynb

**Foco**: MLP para Prever Valores Contínuos

**Conteúdo**:

#### Diferenças vs Classificação

| Aspecto | Classificação | Regressão |
|---------|---------------|-----------|
| **Output** | Probabilidades de classes | Valor contínuo |
| **Neurônios Saída** | num_classes | 1 |
| **Ativação Saída** | softmax/sigmoid | **Nenhuma** |
| **Loss** | categorical_crossentropy | **mse**, **mae** |
| **Métricas** | accuracy, precision, recall | **MAE**, **RMSE**, **R²** |

#### Arquitetura para Regressão

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(1)  # SEM ativação!
])

model.compile(
    optimizer='adam',
    loss='mse',  # ou 'mae'
    metrics=['mae']
)
```

#### Funções de Perda

**MSE (Mean Squared Error)**:
```python
MSE = (1/n) · Σ(y_true - y_pred)²

# Penaliza erros grandes mais fortemente
# Sensível a outliers
```

**MAE (Mean Absolute Error)**:
```python
MAE = (1/n) · Σ|y_true - y_pred|

# Penalização linear
# Robusto a outliers
```

**Huber Loss** (híbrido):
```python
# MSE para erros pequenos, MAE para grandes
# Melhor dos dois mundos
```

#### Métricas de Avaliação

**MAE**:
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
# Erro médio em unidades originais
```

**RMSE** (Root Mean Squared Error):
```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# Mesma escala que o target, penaliza erros grandes
```

**R²** (Coeficiente de Determinação):
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)

# R² = 1 - (SS_res / SS_tot)
# Range: (-∞, 1]
# 1 = predição perfeita
# 0 = modelo tão bom quanto média
# < 0 = modelo pior que média
```

---

## TABELA COMPARATIVA DE ARQUITETURAS

| Arquitetura | Tipo de Dado | Camadas Principais | Quando Usar | Exemplo |
|-------------|--------------|-------------------|-------------|---------|
| **MLP/ANN** | Tabular | Dense | Dados estruturados | Classificação de clientes |
| **CNN** | Imagens | Conv2D, MaxPooling2D | Estrutura espacial | MNIST, ImageNet |
| **RNN** | Sequências curtas | SimpleRNN | Séries temporais simples | Análise sentimento |
| **LSTM** | Sequências longas | LSTM | Dependências longo prazo | Tradução, geração texto |
| **GRU** | Sequências | GRU | Alternativa LSTM mais leve | Previsão vendas |

---

## MELHORES PRÁTICAS POR ARQUITETURA

### MLP
```python
# Estrutura típica
Dense(128, activation='relu'),
BatchNormalization(),
Dropout(0.3),
Dense(64, activation='relu'),
BatchNormalization(),
Dropout(0.2),
Dense(num_classes, activation='softmax')

# Dicas:
# - Começar com 1-2 hidden layers
# - Dropout 0.2-0.5
# - BatchNorm ajuda convergência
```

### CNN
```python
# Estrutura típica
Conv2D(32, (3,3), activation='relu'),
BatchNormalization(),
MaxPooling2D((2,2)),
Dropout(0.25),

Conv2D(64, (3,3), activation='relu'),
BatchNormalization(),
MaxPooling2D((2,2)),
Dropout(0.25),

Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(num_classes, activation='softmax')

# Dicas:
# - Aumentar filtros a cada camada (32→64→128)
# - Pooling reduz dimensões
# - Dropout maior antes do classificador (0.5)
```

### LSTM/GRU
```python
# Estrutura típica
LSTM(128, return_sequences=True),
Dropout(0.3),
LSTM(64),
Dropout(0.2),
Dense(num_classes, activation='softmax')

# Dicas:
# - return_sequences=True se próxima camada é LSTM
# - Dropout 0.2-0.3 (menor que MLP/CNN)
# - Menos camadas que CNN (2-3 suficiente)
```

---

## TROUBLESHOOTING - RESPOSTAS PARA AGENTES

**Q: Modelo não converge (loss estável)?**
A: (1) Reduzir learning rate: `Adam(learning_rate=0.0001)`, (2) Normalizar dados: `StandardScaler()`, (3) Verificar arquitetura (pode estar muito simples ou complexa).

**Q: Overfit severo (treino 99%, val 70%)?**
A: (1) Adicionar Dropout(0.5), (2) Usar EarlyStopping(patience=10), (3) Regularização L2: `kernel_regularizer=l2(0.01)`, (4) Reduzir complexidade (menos camadas/neurônios).

**Q: Quando usar LSTM vs GRU?**
A: LSTM para sequências muito longas (>100 steps) e datasets grandes. GRU para prototipagem rápida, datasets menores, ou quando recursos são limitados. Performance geralmente similar.

**Q: CNN vs MLP para imagens?**
A: Sempre CNN. MLP ignora estrutura espacial (flatten perde informação de vizinhança). CNN aprende features locais (bordas→texturas→objetos). Melhoria típica: +5-10% accuracy.

**Q: Por que ReLU é padrão?**
A: (1) Evita vanishing gradient (derivada é 1 para x>0), (2) Rápido de computar, (3) Esparso (muitos neurônios com 0), (4) Funciona bem na prática. Alternativas: Leaky ReLU, ELU.

**Q: Qual loss para regressão?**
A: MSE se quiser penalizar erros grandes mais fortemente. MAE se quiser robustez a outliers. Huber loss para híbrido. Todas têm gradientes bem comportados.

**Q: BatchNormalization antes ou depois de ativação?**
A: Debate aberto. Original (paper): antes. Prática comum: depois. Testar ambos. Geralmente: `Dense → BatchNorm → Activation`.

**Q: Quantas épocas treinar?**
A: Use EarlyStopping com patience=10-30. Deixe rodar até parar automaticamente. Sem EarlyStopping: 50-100 épocas geralmente suficiente, mas varia muito.

---

## CÓDIGO COMPLETO DE REFERÊNCIA

### Template MLP Completo
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

# Preparar dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
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
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Treinar
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Avaliar
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

## TAGS DE BUSCA

`#deep-learning` `#ann` `#cnn` `#rnn` `#lstm` `#gru` `#mlp` `#backpropagation` `#funcoes-ativacao` `#regularizacao` `#overfitting` `#forward-propagation` `#gradient-descent` `#keras` `#tensorflow` `#redes-neurais` `#conceitos-fundamentais`

---

**Versão**: 1.0  
**Compatibilidade**: Agentes de IA com conhecimento de Deep Learning  
**Uso recomendado**: Consulta rápida de fórmulas, arquiteturas e troubleshooting

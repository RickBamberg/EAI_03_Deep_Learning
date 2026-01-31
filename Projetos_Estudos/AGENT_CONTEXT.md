# AGENT_CONTEXT.md - Projetos de Estudo Deep Learning

> **Propósito**: Contexto técnico de projetos práticos de aprendizado  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Projetos didáticos completos

## RESUMO EXECUTIVO

**Objetivo**: Aplicar conceitos de DL em problemas clássicos conhecidos  
**Projetos**: 4 (MNIST MLP, CNN, Regressão MLP, Regularização Visual)  
**Nível**: Iniciante a intermediário  
**Foco**: Processo de desenvolvimento, não resultado final

---

## PROJETO 1: MNIST_CLASSIFICADOR (MLP)

### Especificações Técnicas

**Dataset**:
```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Shape
X_train: (60000, 28, 28)
X_test:  (10000, 28, 28)
y_train/y_test: valores 0-9
```

### Pipeline Completo

```python
# 1. Preprocessamento
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Arquitetura MLP
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# 3. Compilação
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# 5. Treinamento
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 6. Avaliação
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.4f}")  # Esperado: ~0.97-0.98
```

### Resultados Esperados
```
Epoch 10-20: Convergência
Final Accuracy: 97-98%
Test Loss: ~0.10

Matriz de Confusão:
- Dígitos mais confundidos: 4 com 9, 3 com 5
- Classes bem separadas: 0, 1, 6
```

### Parâmetros do Modelo
```
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0

Layer 1 (Dense 128): 784*128 + 128 = 100,480
Layer 2 (Dense 64):  128*64 + 64 = 8,256
Layer 3 (Dense 10):  64*10 + 10 = 650
```

---

## PROJETO 2: CNN_CLASSIFICADOR (MNIST)

### Arquitetura CNN

```python
# Preprocessamento diferente (mantém estrutura 2D)
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Modelo CNN
model = Sequential([
    # Bloco Convolucional 1
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    # Output: (26, 26, 32)
    MaxPooling2D((2,2)),
    # Output: (13, 13, 32)
    
    # Bloco Convolucional 2
    Conv2D(64, (3,3), activation='relu'),
    # Output: (11, 11, 64)
    MaxPooling2D((2,2)),
    # Output: (5, 5, 64)
    
    # Bloco Convolucional 3
    Conv2D(64, (3,3), activation='relu'),
    # Output: (3, 3, 64)
    
    # Classificador
    Flatten(),  # (3*3*64 = 576)
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Comparação MLP vs CNN

| Métrica | MLP | CNN | Diferença |
|---------|-----|-----|-----------|
| **Test Accuracy** | ~97% | **~99%** | +2% |
| **Total Params** | 109k | **93k** | -16% |
| **Tempo Treino** | ~5 min | ~10 min | +100% |
| **Tempo Predição** | Mais rápido | Mais lento | - |

**Por Que CNN é Melhor?**:
- Aprende features espaciais (bordas, cantos)
- Invariância a translação
- Compartilhamento de pesos (menos parâmetros)

### Visualização de Filtros

```python
# Visualizar filtros da 1ª camada Conv2D
filters, biases = model.layers[0].get_weights()
# filters.shape: (3, 3, 1, 32)

# Plotar primeiros 16 filtros
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[:, :, 0, i], cmap='gray')
    ax.axis('off')
plt.show()
```

---

## PROJETO 3: REGRESSAO_MLP

### Diferenças para Classificação

```python
# Para REGRESSÃO
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(1)  # ← SEM ativação (valor contínuo)
])

model.compile(
    optimizer='adam',
    loss='mse',  # ← NÃO categorical_crossentropy
    metrics=['mae']  # ← MAE em vez de accuracy
)

# Treinamento idêntico
history = model.fit(X_train, y_train, validation_split=0.2, ...)

# Avaliação
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

### Dataset Exemplo: Boston Housing

```python
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

# Features: 13 (CRIM, ZN, INDUS, ...)
# Target: MEDV (preço médio da casa)

# Normalização CRÍTICA para regressão
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Treinar com dados escalados
model.fit(X_train_scaled, y_train_scaled, ...)

# Predição: Reverter escala
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
```

### Métricas de Regressão

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
# Erro médio em unidades originais
# Interpretável: "Erro médio de $3,500"

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Penaliza erros grandes mais fortemente
# Mesma escala que target

# R² (Coeficiente de Determinação)
r2 = r2_score(y_test, y_pred)
# Range: (-∞, 1]
# 1.0 = perfeito
# 0.0 = modelo tão bom quanto média
# <0 = pior que média

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
```

---

## PROJETO 4: REGULARIZACAO_VISUAL

### Objetivo Educacional

Demonstrar visualmente o efeito de técnicas de regularização no overfit.

### Experimentos

#### 1. Baseline (Sem Regularização)

```python
model_baseline = Sequential([
    Dense(512, activation='relu', input_shape=(n_features,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Resultado: OVERFITTING
# Train Acc: 99%
# Val Acc:   75%
```

#### 2. Com Dropout

```python
model_dropout = Sequential([
    Dense(512, activation='relu', input_shape=(n_features,)),
    Dropout(0.5),  # ← Desativa 50%
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Resultado: Melhor generalização
# Train Acc: 95%
# Val Acc:   92%
```

#### 3. Com L2 Regularization

```python
from tensorflow.keras.regularizers import l2

model_l2 = Sequential([
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(num_classes, activation='softmax')
])

# Loss modificada
# Loss_total = Loss_CE + 0.01 * Σ(w²)

# Efeito: Pesos menores e mais distribuídos
```

#### 4. Com BatchNormalization

```python
model_bn = Sequential([
    Dense(512, activation='relu'),
    BatchNormalization(),  # ← Normaliza ativações
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])

# Efeito:
# - Convergência mais rápida
# - Permite learning rates maiores
# - Leve efeito regularizador
```

#### 5. Combinação (Melhor Prática)

```python
model_combo = Sequential([
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(num_classes, activation='softmax')
])

# Resultado: Melhor de todos
# Train Acc: 94%
# Val Acc:   93%
# Gap mínimo (1%)
```

### Visualização Comparativa

```python
# Plotar comparação
models = {
    'Baseline': history_baseline,
    'Dropout': history_dropout,
    'L2': history_l2,
    'BatchNorm': history_bn,
    'Combo': history_combo
}

plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
for name, history in models.items():
    plt.plot(history.history['val_loss'], label=name)
plt.title('Validation Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
for name, history in models.items():
    plt.plot(history.history['val_accuracy'], label=name)
plt.title('Validation Accuracy')
plt.legend()

plt.show()
```

### Análise de Pesos

```python
# Comparar distribuição de pesos
import seaborn as sns

weights_baseline = model_baseline.layers[0].get_weights()[0].flatten()
weights_l2 = model_l2.layers[0].get_weights()[0].flatten()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(weights_baseline, bins=50)
plt.title('Baseline (sem L2)')
plt.xlabel('Valor do Peso')

plt.subplot(1, 2, 2)
sns.histplot(weights_l2, bins=50)
plt.title('Com L2 Regularization')
plt.xlabel('Valor do Peso')

plt.show()

# Observação:
# L2 → Pesos menores e mais concentrados perto de 0
```

---

## COMPARAÇÃO DE RESULTADOS

| Projeto | Accuracy | Tempo Treino | Parâmetros | Dificuldade |
|---------|----------|--------------|------------|-------------|
| MNIST MLP | ~97% | ~5 min | 109k | ⭐ Fácil |
| MNIST CNN | ~99% | ~10 min | 93k | ⭐⭐ Médio |
| Regressão MLP | MAE ~3.5 | ~3 min | ~10k | ⭐ Fácil |
| Regularização | N/A (educacional) | ~5 min/modelo | Varia | ⭐⭐ Médio |

---

## EXPERIMENTOS SUGERIDOS - CÓDIGOS

### MNIST MLP - Variar Camadas

```python
# Experimento: 1, 2, 3, 4 camadas ocultas
configs = [
    [128],
    [128, 64],
    [128, 64, 32],
    [256, 128, 64, 32]
]

results = {}
for i, layers in enumerate(configs):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(784,)))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, verbose=0)
    
    results[f'{len(layers)} camadas'] = history.history['val_accuracy'][-1]

# Resultado esperado: 2-3 camadas são ótimas
print(results)
```

### CNN - Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Sem augmentation: ~99.0%
# Com augmentation: ~99.3%

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

history_aug = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50
)
```

---

## FAQ TÉCNICO - PROJETOS ESTUDOS

**Q: Por que CNN é melhor para MNIST mesmo sendo "simples"?**
A: MNIST tem estrutura espacial (pixels vizinhos importam). CNN aprende features locais (bordas, curvas) que compõem dígitos. MLP trata cada pixel independentemente.

**Q: Devo sempre normalizar em regressão?**
A: SIM. Features em escalas diferentes (ex: idade 0-100, salário 20k-200k) causam gradientes desbalanceados. Normalizar acelera convergência e melhora performance.

**Q: Dropout durante teste?**
A: NÃO. Keras desativa automaticamente durante `model.predict()`. Durante treino, neurônios são desativados aleatoriamente. Durante teste, todos neurônios são usados.

**Q: BatchNorm antes ou depois de ativação?**
A: Debate aberto. Paper original: antes. Prática comum: depois (`Dense → BatchNorm → Activation`). Experimentar ambos.

**Q: Como saber se está overfitting?**
A: Gap grande entre treino e validação. Ex: train_acc=99%, val_acc=75% = OVERFIT. Solução: Dropout, L2, EarlyStopping.

**Q: MNIST CNN converge em quantas épocas?**
A: Tipicamente 10-20 épocas. Usar EarlyStopping(patience=10) para parar automaticamente.

**Q: R² negativo é possível?**
A: SIM. Significa que modelo é pior que simplesmente prever a média dos valores. Revisar arquitetura, normalização, ou features.

---

## CHECKLIST DE CONCLUSÃO

### Após Completar os 4 Projetos:

**Conceitos Dominados**:
- [ ] Entendo diferença MLP vs CNN em imagens
- [ ] Sei quando usar regressão vs classificação
- [ ] Compreendo overfitting e regularização
- [ ] Entendo callbacks (EarlyStopping)

**Habilidades Práticas**:
- [ ] Treinei pelo menos 1 modelo do zero
- [ ] Plotei curvas de loss/accuracy
- [ ] Interpretei matriz de confusão
- [ ] Salvei e carreguei modelo

**Experimentação**:
- [ ] Variei arquiteturas e observei impacto
- [ ] Comparei com/sem Dropout
- [ ] Usei diferentes loss functions

---

## TAGS DE BUSCA

`#mnist` `#mlp` `#cnn` `#regressao` `#regularizacao` `#dropout` `#batchnormalization` `#l2` `#projetos-praticos` `#keras` `#tensorflow` `#didatico`

---

**Versão**: 1.0  
**Uso recomendado**: Executar notebooks, experimentar, e consolidar conceitos antes de projetos reais

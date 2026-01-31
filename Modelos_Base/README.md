# Modelos Base - Deep Learning

## 📌 Sobre

Esta pasta contém **templates completos e reutilizáveis** de arquiteturas de Deep Learning prontas para uso. São estruturas pré-configuradas com melhores práticas para iniciar rapidamente novos projetos.

**Objetivo**: Acelerar desenvolvimento fornecendo código testado e otimizado para diferentes tipos de problemas.

---

## 🎯 Diferença: Conceitos vs Modelos_Base

| Aspecto | Conceitos | Modelos_Base |
|---------|-----------|--------------|
| **Foco** | Explicações teóricas | Código pronto para usar |
| **Conteúdo** | Fundamentos e fórmulas | Arquiteturas completas |
| **Uso** | Aprender | Copiar e adaptar |
| **Exemplos** | Pequenos e didáticos | Completos e prontos |

---

## 📂 Arquivos Disponíveis

### 📘 **Estrutura_ANN.ipynb** ⭐

**Descrição**: Template completo com arquiteturas pré-configuradas para ANN, CNN e LSTM.

**Conteúdo**:

#### 1. Imports Organizados
```python
# Data
import numpy as np
import pandas as pd

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, 
    Flatten, LSTM, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
```

#### 2. Preparação de Dados
```python
def preparar_dados_classificacao(X, y, test_size=0.2):
    """
    Prepara dados para classificação
    
    Returns:
        X_train, X_test, y_train, y_test (normalizados)
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encoding do target
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    return X_train, X_test, y_train_cat, y_test_cat, scaler

def preparar_dados_imagem(X, y, img_height=28, img_width=28):
    """
    Prepara dados de imagem para CNN
    
    Returns:
        X_train, X_test, y_train, y_test (reshape para CNN)
    """
    # Normalizar pixels [0, 255] → [0, 1]
    X = X.astype('float32') / 255.0
    
    # Reshape para CNN (samples, height, width, channels)
    X = X.reshape(-1, img_height, img_width, 1)
    
    # Split e one-hot
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    return X_train, X_test, y_train_cat, y_test_cat
```

#### 3. Arquitetura MLP (Multi-Layer Perceptron)
```python
def criar_mlp(input_shape, num_classes, hidden_layers=[128, 64], dropout_rate=0.3):
    """
    Cria MLP para classificação
    
    Args:
        input_shape: tuple (n_features,)
        num_classes: int
        hidden_layers: list de int (neurônios por camada)
        dropout_rate: float
    
    Returns:
        model: Sequential
    """
    model = Sequential()
    
    # Primeira camada oculta
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Camadas ocultas adicionais
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Camada de saída
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Exemplo de uso:
# model = criar_mlp(input_shape=(784,), num_classes=10, hidden_layers=[256, 128, 64])
```

#### 4. Arquitetura CNN (Convolutional Neural Network)
```python
def criar_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Cria CNN para classificação de imagens
    
    Args:
        input_shape: tuple (height, width, channels)
        num_classes: int
    
    Returns:
        model: Sequential
    """
    model = Sequential([
        # Bloco Convolucional 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Bloco Convolucional 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Bloco Convolucional 3
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        
        # Flatten e Dense
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Exemplo de uso:
# model = criar_cnn(input_shape=(28, 28, 1), num_classes=10)
```

#### 5. Arquitetura LSTM (Long Short-Term Memory)
```python
def criar_lstm(input_shape, num_classes, lstm_units=[128, 64], dropout_rate=0.3):
    """
    Cria LSTM para classificação de sequências
    
    Args:
        input_shape: tuple (timesteps, features)
        num_classes: int
        lstm_units: list de int (unidades por camada LSTM)
        dropout_rate: float
    
    Returns:
        model: Sequential
    """
    model = Sequential()
    
    # Primeira camada LSTM
    model.add(LSTM(
        lstm_units[0], 
        return_sequences=len(lstm_units) > 1,  # True se há mais camadas
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))
    
    # Camadas LSTM adicionais
    for i, units in enumerate(lstm_units[1:]):
        return_seq = i < len(lstm_units) - 2  # True se não é última camada
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    
    # Camada de saída
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Exemplo de uso:
# model = criar_lstm(input_shape=(100, 50), num_classes=5, lstm_units=[128, 64])
```

#### 6. Callbacks Configurados
```python
def criar_callbacks(model_path='best_model.keras', patience=10):
    """
    Cria callbacks padrão para treinamento
    
    Args:
        model_path: str (onde salvar melhor modelo)
        patience: int (épocas sem melhora para parar)
    
    Returns:
        list de callbacks
    """
    callbacks = [
        # Salva melhor modelo
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Para quando validação não melhora
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduz learning rate quando estagnado
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

# Exemplo de uso:
# callbacks = criar_callbacks(model_path='models/mlp_best.keras', patience=15)
```

#### 7. Função de Treinamento
```python
def treinar_modelo(model, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, callbacks=None):
    """
    Treina modelo com configurações padrão
    
    Returns:
        history: History object
    """
    if callbacks is None:
        callbacks = criar_callbacks()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

#### 8. Visualização de Resultados
```python
def plotar_historico(history):
    """
    Plota curvas de loss e accuracy
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Treino')
    ax1.plot(history.history['val_loss'], label='Validação')
    ax1.set_title('Loss por Época')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Treino')
    ax2.plot(history.history['val_accuracy'], label='Validação')
    ax2.set_title('Accuracy por Época')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plotar_confusion_matrix(y_true, y_pred, classes):
    """
    Plota matriz de confusão
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Predição')
    plt.show()
```

---

## 🚀 Como Usar os Templates

### Exemplo 1: Classificação com MLP

```python
# 1. Carregar dados
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

# 2. Preparar
X_train, X_test, y_train, y_test, scaler = preparar_dados_classificacao(X, y)

# 3. Criar modelo
model = criar_mlp(input_shape=(X_train.shape[1],), num_classes=10)
model.summary()

# 4. Callbacks
callbacks = criar_callbacks(model_path='mlp_digits.keras', patience=15)

# 5. Treinar
history = treinar_modelo(
    model, X_train, y_train, X_test, y_test,
    epochs=100, batch_size=32, callbacks=callbacks
)

# 6. Visualizar
plotar_historico(history)

# 7. Avaliar
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)
plotar_confusion_matrix(y_true, y_pred, classes=range(10))
```

### Exemplo 2: Classificação de Imagens com CNN

```python
# 1. Carregar MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Preparar para CNN
X_train, X_test, y_train, y_test = preparar_dados_imagem(
    np.concatenate([X_train, X_test]),
    np.concatenate([y_train, y_test]),
    img_height=28,
    img_width=28
)

# 3. Criar CNN
model = criar_cnn(input_shape=(28, 28, 1), num_classes=10)
model.summary()

# 4. Treinar
callbacks = criar_callbacks('cnn_mnist.keras', patience=10)
history = treinar_modelo(
    model, X_train, y_train, X_test, y_test,
    epochs=50, batch_size=64, callbacks=callbacks
)

# 5. Visualizar
plotar_historico(history)
```

### Exemplo 3: Séries Temporais com LSTM

```python
# 1. Preparar dados sequenciais
# X shape: (samples, timesteps, features)
# y shape: (samples, num_classes)

# 2. Criar LSTM
model = criar_lstm(
    input_shape=(100, 50),  # 100 timesteps, 50 features
    num_classes=5,
    lstm_units=[128, 64],
    dropout_rate=0.3
)

# 3. Treinar
callbacks = criar_callbacks('lstm_series.keras', patience=20)
history = treinar_modelo(
    model, X_train, y_train, X_val, y_val,
    epochs=100, batch_size=32, callbacks=callbacks
)
```

---

## ⚙️ Configurações Recomendadas

### Hiperparâmetros por Tipo de Problema

| Problema | Arquitetura | Hidden Layers | Dropout | Batch Size | Epochs |
|----------|-------------|---------------|---------|------------|--------|
| Tabular pequeno (<10k) | MLP | [64, 32] | 0.3 | 32 | 50-100 |
| Tabular grande (>10k) | MLP | [256, 128, 64] | 0.5 | 64 | 100-200 |
| Imagens pequenas (28×28) | CNN | - | 0.25-0.5 | 64 | 50 |
| Imagens médias (224×224) | CNN | - | 0.5 | 32 | 100 |
| Séries curtas (<50 steps) | LSTM | [64] | 0.2 | 32 | 50 |
| Séries longas (>100 steps) | LSTM | [128, 64] | 0.3 | 64 | 100-200 |

### Callbacks Padrão

```python
# Configuração conservadora (evita overfit)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# Configuração agressiva (busca máxima performance)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.2, patience=10)
]
```

---

## 🔧 Customizações Comuns

### Adicionar Data Augmentation (Imagens)
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Treinar com augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=callbacks
)
```

### Adicionar Class Weights (Dados Desbalanceados)
```python
from sklearn.utils import class_weight

# Calcular pesos
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train.argmax(axis=1)),
    y=y_train.argmax(axis=1)
)
class_weights_dict = dict(enumerate(class_weights))

# Treinar com pesos
history = model.fit(
    X_train, y_train,
    class_weight=class_weights_dict,
    validation_data=(X_test, y_test),
    epochs=100
)
```

### Usar Transfer Learning (CNN)
```python
from tensorflow.keras.applications import VGG16

# Carregar modelo pré-treinado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar camadas

# Adicionar classificador
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 📊 Checklist de Uso

Antes de treinar um modelo:

- [ ] Dados normalizados/escalonados
- [ ] Shape correto (MLP: 2D, CNN: 4D, LSTM: 3D)
- [ ] Target em one-hot encoding (se classificação)
- [ ] Callbacks configurados (EarlyStopping, ModelCheckpoint)
- [ ] Arquitetura apropriada ao problema
- [ ] Loss e métricas corretas

Durante o treino:

- [ ] Monitorar loss de treino e validação
- [ ] Verificar se está convergindo
- [ ] Observar se há overfit (treino >> validação)

Após o treino:

- [ ] Plotar curvas de loss e accuracy
- [ ] Avaliar no conjunto de teste
- [ ] Salvar modelo final
- [ ] Documentar hiperparâmetros usados

---

## 💡 Dicas de Otimização

### Para Melhorar Performance

1. **Aumentar capacidade**:
   - Adicionar mais camadas
   - Aumentar número de neurônios/filtros
   
2. **Regularizar mais**:
   - Aumentar dropout
   - Adicionar L2 regularization
   - Usar data augmentation

3. **Ajustar learning rate**:
   - Começar com `adam` (padrão)
   - Testar `sgd` com momentum
   - Usar ReduceLROnPlateau

4. **Treinar por mais tempo**:
   - Aumentar epochs
   - Reduzir patience do EarlyStopping

### Para Treinar Mais Rápido

1. **Reduzir tamanho do batch**:
   - Menor batch = mais atualizações por época
   
2. **Usar GPU**:
   - Verificar: `tf.config.list_physical_devices('GPU')`
   
3. **Simplificar modelo**:
   - Menos camadas/neurônios
   - Menos parâmetros

---

## 🎯 Quando Usar Cada Arquitetura

```
Dados Tabulares → MLP
    ↓
Imagens → CNN
    ↓
Sequências/Séries Temporais → LSTM/GRU
    ↓
Texto → LSTM/GRU ou Transformers (módulo posterior)
    ↓
Áudio/Vídeo → CNN + LSTM ou modelos especializados
```

---

## 📚 Recursos Adicionais

- **Keras Documentation**: https://keras.io/api/
- **TensorFlow Tutorials**: https://www.tensorflow.org/tutorials
- **Model Zoo**: https://modelzoo.co/ (arquiteturas prontas)

---

**Próximos Passos**: Vá para `../Projetos_Estudos/` para ver esses templates em ação!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_03*

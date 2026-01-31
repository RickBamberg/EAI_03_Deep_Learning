# AGENT_CONTEXT.md - Modelos Base de Deep Learning

> **Propósito**: Templates prontos e reutilizáveis de arquiteturas de Deep Learning  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Código de produção otimizado

## RESUMO EXECUTIVO

**Objetivo**: Fornecer código completo e testado para iniciar rapidamente projetos de DL  
**Arquivo Principal**: Estrutura_ANN.ipynb  
**Arquiteturas**: MLP, CNN, LSTM (3 principais)  
**Diferencial**: Funções parametrizáveis + callbacks + visualização tudo-em-um

---

## ESTRUTURA DO ARQUIVO ESTRUTURA_ANN.IPYNB

### SEÇÃO 1: IMPORTS

```python
# Data Manipulation
import numpy as np
import pandas as pd

# Deep Learning
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Flatten,
    LSTM, GRU, BatchNormalization, Input,
    GlobalAveragePooling2D, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import warnings
warnings.filterwarnings('ignore')
```

---

### SEÇÃO 2: PREPARAÇÃO DE DADOS

#### Função: preparar_dados_classificacao()

```python
def preparar_dados_classificacao(X, y, test_size=0.2, random_state=42, 
                                 scale=True, scaler_type='standard'):
    """
    Prepara dados para classificação com normalização e one-hot encoding
    
    Args:
        X: np.array ou pd.DataFrame - Features
        y: np.array ou pd.Series - Target
        test_size: float - Proporção do teste (0.0-1.0)
        random_state: int - Seed para reprodutibilidade
        scale: bool - Se deve normalizar
        scaler_type: str - 'standard', 'minmax', ou 'none'
    
    Returns:
        tuple: (X_train, X_test, y_train_cat, y_test_cat, scaler, label_encoder)
    
    Exemplo:
        >>> X_train, X_test, y_train, y_test, scaler, le = preparar_dados_classificacao(X, y)
    """
    # Converter para numpy se necessário
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Mantém proporção de classes
    )
    
    # Normalização
    scaler = None
    if scale:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type deve ser 'standard' ou 'minmax'")
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # One-hot encoding do target
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train_encoded, num_classes)
    y_test_cat = to_categorical(y_test_encoded, num_classes)
    
    print(f"✓ Dados preparados:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste:  {X_test.shape[0]} amostras")
    print(f"  Classes: {num_classes} ({le.classes_})")
    
    return X_train, X_test, y_train_cat, y_test_cat, scaler, le
```

#### Função: preparar_dados_imagem()

```python
def preparar_dados_imagem(X, y, img_height=28, img_width=28, channels=1, 
                          test_size=0.2, random_state=42, normalize=True):
    """
    Prepara dados de imagem para CNN
    
    Args:
        X: np.array - Imagens (samples, height, width) ou flat
        y: np.array - Labels
        img_height: int - Altura da imagem
        img_width: int - Largura da imagem
        channels: int - 1 (grayscale) ou 3 (RGB)
        test_size: float
        random_state: int
        normalize: bool - Dividir por 255.0
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    
    Exemplo:
        >>> X_train, X_test, y_train, y_test = preparar_dados_imagem(X, y, 28, 28, 1)
    """
    # Normalizar pixels [0, 255] → [0, 1]
    if normalize:
        X = X.astype('float32') / 255.0
    
    # Reshape para CNN (samples, height, width, channels)
    if X.ndim == 2:  # Se flat (samples, pixels)
        X = X.reshape(-1, img_height, img_width, channels)
    elif X.ndim == 3 and channels == 1:  # Se (samples, height, width)
        X = X.reshape(-1, img_height, img_width, channels)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # One-hot encoding
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"✓ Imagens preparadas:")
    print(f"  Shape treino: {X_train.shape}")
    print(f"  Shape teste:  {X_test.shape}")
    print(f"  Classes: {num_classes}")
    
    return X_train, X_test, y_train_cat, y_test_cat
```

---

### SEÇÃO 3: ARQUITETURAS

#### Função: criar_mlp()

```python
def criar_mlp(input_shape, num_classes, 
              hidden_layers=[128, 64], 
              dropout_rate=0.3,
              use_batchnorm=True,
              l2_reg=0.01,
              optimizer='adam',
              learning_rate=0.001):
    """
    Cria Multi-Layer Perceptron otimizado
    
    Args:
        input_shape: tuple - (n_features,)
        num_classes: int - Número de classes
        hidden_layers: list[int] - Neurônios por camada oculta
        dropout_rate: float - Taxa de dropout (0.0-0.5)
        use_batchnorm: bool - Usar BatchNormalization
        l2_reg: float - Força da regularização L2
        optimizer: str - 'adam', 'sgd', 'rmsprop'
        learning_rate: float - Learning rate
    
    Returns:
        model: Sequential compilado
    
    Exemplo:
        >>> model = criar_mlp((784,), 10, hidden_layers=[256, 128, 64])
        >>> model.summary()
    """
    model = Sequential(name='MLP')
    
    # Primeira camada oculta
    model.add(Dense(
        hidden_layers[0], 
        activation='relu',
        input_shape=input_shape,
        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
        name='dense_1'
    ))
    if use_batchnorm:
        model.add(BatchNormalization(name='bn_1'))
    model.add(Dropout(dropout_rate, name='dropout_1'))
    
    # Camadas ocultas adicionais
    for i, units in enumerate(hidden_layers[1:], start=2):
        model.add(Dense(
            units, 
            activation='relu',
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
            name=f'dense_{i}'
        ))
        if use_batchnorm:
            model.add(BatchNormalization(name=f'bn_{i}'))
        model.add(Dropout(dropout_rate, name=f'dropout_{i}'))
    
    # Camada de saída
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    # Compilar
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = optimizer
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ MLP criado:")
    print(f"  Camadas ocultas: {hidden_layers}")
    print(f"  Total parâmetros: {model.count_params():,}")
    
    return model
```

#### Função: criar_cnn()

```python
def criar_cnn(input_shape=(28, 28, 1), num_classes=10,
              conv_layers=[(32, 3), (64, 3), (64, 3)],
              dense_units=128,
              dropout_rate=0.5,
              use_batchnorm=True,
              pool_size=2):
    """
    Cria Convolutional Neural Network otimizada
    
    Args:
        input_shape: tuple - (height, width, channels)
        num_classes: int
        conv_layers: list[tuple] - [(filters, kernel_size), ...]
        dense_units: int - Neurônios na camada densa
        dropout_rate: float
        use_batchnorm: bool
        pool_size: int - Tamanho do MaxPooling
    
    Returns:
        model: Sequential compilado
    
    Exemplo:
        >>> model = criar_cnn((28,28,1), 10, conv_layers=[(32,3), (64,3), (128,3)])
        >>> model.summary()
    """
    model = Sequential(name='CNN')
    
    # Blocos convolucionais
    for i, (filters, kernel_size) in enumerate(conv_layers, start=1):
        if i == 1:
            model.add(Conv2D(
                filters, (kernel_size, kernel_size),
                activation='relu',
                input_shape=input_shape,
                padding='same',
                name=f'conv_{i}'
            ))
        else:
            model.add(Conv2D(
                filters, (kernel_size, kernel_size),
                activation='relu',
                padding='same',
                name=f'conv_{i}'
            ))
        
        if use_batchnorm:
            model.add(BatchNormalization(name=f'bn_conv_{i}'))
        
        # MaxPooling a cada 1 ou 2 camadas conv
        if i % 1 == 0 and i < len(conv_layers):
            model.add(MaxPooling2D((pool_size, pool_size), name=f'pool_{i}'))
            model.add(Dropout(0.25, name=f'dropout_conv_{i}'))
    
    # Classificador
    model.add(Flatten(name='flatten'))
    model.add(Dense(dense_units, activation='relu', name='dense'))
    if use_batchnorm:
        model.add(BatchNormalization(name='bn_dense'))
    model.add(Dropout(dropout_rate, name='dropout_dense'))
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ CNN criada:")
    print(f"  Blocos conv: {len(conv_layers)}")
    print(f"  Total parâmetros: {model.count_params():,}")
    
    return model
```

#### Função: criar_lstm()

```python
def criar_lstm(input_shape, num_classes,
               lstm_units=[128, 64],
               dropout_rate=0.3,
               recurrent_dropout=0.2,
               bidirectional=False,
               dense_units=32):
    """
    Cria LSTM para classificação de sequências
    
    Args:
        input_shape: tuple - (timesteps, features)
        num_classes: int
        lstm_units: list[int] - Unidades por camada LSTM
        dropout_rate: float - Dropout após LSTM
        recurrent_dropout: float - Dropout recorrente dentro LSTM
        bidirectional: bool - Usar Bidirectional LSTM
        dense_units: int - Neurônios camada densa final
    
    Returns:
        model: Sequential compilado
    
    Exemplo:
        >>> model = criar_lstm((100, 50), 5, lstm_units=[128, 64], bidirectional=True)
        >>> model.summary()
    """
    model = Sequential(name='LSTM')
    
    # Primeira camada LSTM
    return_seq = len(lstm_units) > 1
    
    if bidirectional:
        model.add(Bidirectional(
            LSTM(
                lstm_units[0],
                return_sequences=return_seq,
                dropout=recurrent_dropout,
                name='lstm_1'
            ),
            input_shape=input_shape,
            name='bi_lstm_1'
        ))
    else:
        model.add(LSTM(
            lstm_units[0],
            return_sequences=return_seq,
            dropout=recurrent_dropout,
            input_shape=input_shape,
            name='lstm_1'
        ))
    
    model.add(Dropout(dropout_rate, name='dropout_1'))
    
    # Camadas LSTM adicionais
    for i, units in enumerate(lstm_units[1:], start=2):
        return_seq = i < len(lstm_units)
        
        if bidirectional:
            model.add(Bidirectional(
                LSTM(
                    units,
                    return_sequences=return_seq,
                    dropout=recurrent_dropout,
                    name=f'lstm_{i}'
                ),
                name=f'bi_lstm_{i}'
            ))
        else:
            model.add(LSTM(
                units,
                return_sequences=return_seq,
                dropout=recurrent_dropout,
                name=f'lstm_{i}'
            ))
        
        model.add(Dropout(dropout_rate, name=f'dropout_{i}'))
    
    # Camada densa final (opcional)
    if dense_units > 0:
        model.add(Dense(dense_units, activation='relu', name='dense'))
        model.add(Dropout(dropout_rate, name='dropout_dense'))
    
    # Saída
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ LSTM criada:")
    print(f"  Camadas LSTM: {len(lstm_units)}")
    print(f"  Bidirectional: {bidirectional}")
    print(f"  Total parâmetros: {model.count_params():,}")
    
    return model
```

---

### SEÇÃO 4: CALLBACKS

```python
def criar_callbacks(model_path='models/best_model.keras',
                   patience=15,
                   reduce_lr_patience=5,
                   min_lr=1e-7,
                   monitor='val_loss',
                   save_best_only=True,
                   tensorboard_log_dir='logs/',
                   csv_log_path='training.csv'):
    """
    Cria conjunto padrão de callbacks
    
    Args:
        model_path: str - Onde salvar melhor modelo
        patience: int - Épocas sem melhora para parar
        reduce_lr_patience: int - Épocas para reduzir LR
        min_lr: float - LR mínimo
        monitor: str - Métrica a monitorar
        save_best_only: bool - Salvar apenas se melhorar
        tensorboard_log_dir: str - Dir para TensorBoard
        csv_log_path: str - Path para log CSV
    
    Returns:
        list: Callbacks configurados
    
    Exemplo:
        >>> callbacks = criar_callbacks('models/mlp_best.keras', patience=20)
        >>> history = model.fit(..., callbacks=callbacks)
    """
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    callbacks = [
        # Salva melhor modelo
        ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            save_best_only=save_best_only,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1,
            save_weights_only=False
        ),
        
        # Para quando não melhora
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        ),
        
        # Reduz learning rate quando estagna
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        ),
        
        # TensorBoard para visualização
        TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True
        ),
        
        # Log CSV
        CSVLogger(
            filename=csv_log_path,
            separator=',',
            append=False
        )
    ]
    
    print(f"✓ Callbacks configurados:")
    print(f"  ModelCheckpoint: {model_path}")
    print(f"  EarlyStopping patience: {patience}")
    print(f"  ReduceLROnPlateau patience: {reduce_lr_patience}")
    
    return callbacks
```

---

### SEÇÃO 5: VISUALIZAÇÃO

```python
def plotar_historico(history, save_path=None):
    """
    Plota loss e accuracy durante treinamento
    
    Args:
        history: History object do model.fit()
        save_path: str (opcional) - Salvar figura
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Treino', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validação', linewidth=2)
    ax1.set_title('Loss por Época', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Treino', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    ax2.set_title('Accuracy por Época', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura salva em: {save_path}")
    
    plt.show()

def plotar_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Plota matriz de confusão
    
    Args:
        y_true: np.array - Labels verdadeiros
        y_pred: np.array - Labels preditos
        classes: list - Nomes das classes
        save_path: str (opcional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Contagem'}
    )
    plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Predição', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

---

## EXEMPLO DE USO COMPLETO

```python
# 1. Carregar dados
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

# 2. Preparar
X_train, X_test, y_train, y_test, scaler, le = preparar_dados_classificacao(
    X, y, test_size=0.2, scale=True
)

# 3. Criar modelo MLP
model = criar_mlp(
    input_shape=(X_train.shape[1],),
    num_classes=10,
    hidden_layers=[256, 128, 64],
    dropout_rate=0.3,
    use_batchnorm=True,
    l2_reg=0.01
)

# 4. Ver arquitetura
model.summary()

# 5. Configurar callbacks
callbacks = criar_callbacks(
    model_path='models/digits_mlp.keras',
    patience=20
)

# 6. Treinar
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 7. Plotar resultados
plotar_historico(history, save_path='training_history.png')

# 8. Avaliar
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))
plotar_confusion_matrix(y_true, y_pred, classes=range(10))

# 9. Salvar scaler
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Modelo e scaler salvos!")
```

---

## CONFIGURAÇÕES RECOMENDADAS POR PROBLEMA

### Tabular Pequeno (<10k amostras)
```python
model = criar_mlp(
    input_shape=(n_features,),
    num_classes=num_classes,
    hidden_layers=[64, 32],
    dropout_rate=0.3,
    use_batchnorm=True,
    l2_reg=0.01
)
# Epochs: 50-100, Batch: 32
```

### Tabular Grande (>100k amostras)
```python
model = criar_mlp(
    input_shape=(n_features,),
    num_classes=num_classes,
    hidden_layers=[256, 128, 64],
    dropout_rate=0.5,
    use_batchnorm=True,
    l2_reg=0.01
)
# Epochs: 100-200, Batch: 64-128
```

### Imagens Pequenas (28×28)
```python
model = criar_cnn(
    input_shape=(28, 28, 1),
    num_classes=10,
    conv_layers=[(32, 3), (64, 3), (64, 3)],
    dense_units=128,
    dropout_rate=0.5
)
# Epochs: 50, Batch: 64
```

### Séries Temporais Curtas (<50 steps)
```python
model = criar_lstm(
    input_shape=(timesteps, features),
    num_classes=num_classes,
    lstm_units=[64],
    dropout_rate=0.2,
    bidirectional=False
)
# Epochs: 50-100, Batch: 32
```

### Séries Temporais Longas (>100 steps)
```python
model = criar_lstm(
    input_shape=(timesteps, features),
    num_classes=num_classes,
    lstm_units=[128, 64],
    dropout_rate=0.3,
    bidirectional=True
)
# Epochs: 100-200, Batch: 64
```

---

## FAQ TÉCNICO

**Q: Como carregar modelo salvo?**
```python
from tensorflow.keras.models import load_model
model = load_model('models/best_model.keras')
```

**Q: Como fazer predição em produção?**
```python
import joblib
# Carregar
model = load_model('models/model.keras')
scaler = joblib.load('models/scaler.pkl')

# Processar novo dado
X_new_scaled = scaler.transform(X_new)
y_pred = model.predict(X_new_scaled)
class_pred = y_pred.argmax(axis=1)
```

**Q: Como adicionar Data Augmentation?**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=100
)
```

**Q: Como usar GPU?**
```python
import tensorflow as tf

# Verificar GPUs
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Usar GPU específica
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
```

**Q: Como salvar apenas pesos?**
```python
# Salvar pesos
model.save_weights('models/weights.h5')

# Carregar em modelo idêntico
new_model = criar_mlp(...)
new_model.load_weights('models/weights.h5')
```

---

## TAGS DE BUSCA

`#templates` `#modelos-base` `#mlp` `#cnn` `#lstm` `#keras` `#tensorflow` `#callbacks` `#earlystopping` `#modelcheckpoint` `#visualizacao` `#producao` `#codigo-reutilizavel`

---

**Versão**: 1.0  
**Compatibilidade**: TensorFlow 2.x, Keras  
**Uso recomendado**: Copiar funções e adaptar para seu projeto

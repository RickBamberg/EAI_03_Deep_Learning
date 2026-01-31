# Projetos de Estudo - Deep Learning

## 📌 Sobre

Esta pasta contém **projetos práticos de aprendizado** para aplicar conceitos de Deep Learning em problemas reais, mas com escopo controlado e didático.

**Objetivo**: Praticar implementação de arquiteturas em problemas conhecidos, focando no processo de desenvolvimento mais que no resultado final.

---

## 🎯 Diferença: Projetos_Estudos vs Projetos_Reais

| Aspecto | Projetos_Estudos | Projetos_Reais |
|---------|------------------|----------------|
| **Escopo** | Problemas clássicos | Aplicações deployadas |
| **Complexidade** | Controlada | Alta |
| **Foco** | Aprendizado | Produção |
| **Datasets** | Públicos famosos | Diversos |
| **Deployment** | Opcional | Obrigatório |
| **Documentação** | Básica | Completa |

---

## 📂 Projetos Disponíveis

### 1️⃣ **MNIST_Classificador/** 🔢

**Problema**: Classificar dígitos escritos à mão (0-9)

**Dataset**: 
- **MNIST**: 70,000 imagens 28×28 pixels em escala de cinza
- 60,000 treino + 10,000 teste
- 10 classes (dígitos 0-9)

**Arquitetura**: MLP (Multi-Layer Perceptron)
```python
Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

**Resultado Esperado**: ~97-98% accuracy

**Aprendizado**:
- Primeiro contato com Keras/TensorFlow
- Normalização de imagens (pixels / 255)
- One-hot encoding de targets
- Uso de callbacks (EarlyStopping)

**Arquivos**:
- `mnist_classificador.ipynb` - Notebook principal

**Duração**: ~1-2 horas

---

### 2️⃣ **CNN_Classificador/** 🖼️

**Problema**: Classificar dígitos MNIST usando CNN (Convolutional Neural Network)

**Dataset**: MNIST (mesmo do projeto 1)

**Arquitetura**: CNN
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

**Resultado Esperado**: ~99% accuracy

**Aprendizado**:
- Diferença de performance MLP vs CNN em imagens
- Camadas Conv2D e MaxPooling2D
- Reshape de dados para CNN (samples, height, width, channels)
- Visualização de filtros aprendidos

**Arquivos**:
- `cnn_classificador_mnist.ipynb` - Implementação completa
- `visualizacao_filtros.ipynb` (opcional) - Ver o que CNN aprende

**Duração**: ~2-3 horas

**Comparação com MLP**:
| Métrica | MLP | CNN |
|---------|-----|-----|
| Accuracy | ~97% | ~99% |
| Parâmetros | ~100k | ~93k |
| Tempo de treino | Mais rápido | Mais lento |

---

### 3️⃣ **Regressao_MLP/** 📈

**Problema**: Prever valores contínuos usando MLP

**Dataset**: Sintético ou Boston Housing (preços de casas)

**Arquitetura**: MLP para Regressão
```python
Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(1)  # Saída contínua sem ativação
])
```

**Diferenças vs Classificação**:
- **Saída**: 1 neurônio (valor contínuo)
- **Ativação final**: Nenhuma
- **Loss**: `mse` ou `mae` (não categorical_crossentropy)
- **Métricas**: MAE, RMSE

**Aprendizado**:
- Regressão com redes neurais
- Diferença entre classificação e regressão
- Métricas de regressão (MAE, RMSE, R²)

**Arquivos**:
- `regressao_mlp.ipynb` - Implementação
- `regressao_manual.ipynb` - Comparação com regressão linear

**Duração**: ~1-2 horas

---

### 4️⃣ **Regularizacao_Visual/** 🎨

**Problema**: Demonstrar visualmente efeitos de regularização

**Foco**: Educacional (não é classificação real)

**Técnicas Demonstradas**:

#### Dropout
```python
Dense(128, activation='relu'),
Dropout(0.5),  # Desativa 50% dos neurônios
Dense(64, activation='relu'),
```

**Efeito**: Reduz overfit desativando neurônios aleatoriamente

#### L2 Regularization
```python
Dense(128, activation='relu', kernel_regularizer=l2(0.01))
```

**Efeito**: Penaliza pesos grandes, forçando distribuição mais uniforme

#### BatchNormalization
```python
Dense(128, activation='relu'),
BatchNormalization(),  # Normaliza ativações
```

**Efeito**: Estabiliza e acelera treinamento

**Comparações Visuais**:
- Sem regularização: Overfit (treino 99%, validação 80%)
- Com Dropout: Melhor generalização (treino 95%, validação 92%)
- Com L2: Pesos menores e mais distribuídos
- Com BatchNorm: Convergência mais rápida

**Arquivos**:
- `visualizacao_regularizacao.ipynb` - Comparações lado a lado

**Duração**: ~1 hora

**Aprendizado**:
- Identificar overfit visualmente
- Quando usar cada técnica
- Combinar técnicas de regularização

---

## 🗺️ Ordem de Estudo Recomendada

### Iniciante (Primeiro Contato com Deep Learning)
```
1. MNIST_Classificador       (MLP básico)
2. CNN_Classificador          (introdução a CNNs)
3. Regularizacao_Visual       (evitar overfit)
4. Regressao_MLP              (caso especial)
```

### Intermediário (Já conhece ML)
```
1. CNN_Classificador          (direto para CNNs)
2. Regularizacao_Visual       (técnicas avançadas)
3. Regressao_MLP              (se for fazer regressão)
```

### Avançado (Revisão Rápida)
```
- Executar notebooks rapidamente
- Focar em detalhes de implementação
- Modificar arquiteturas e observar impacto
```

---

## 📊 Comparação de Resultados Esperados

| Projeto | Accuracy/Métrica | Tempo Treino | Dificuldade |
|---------|------------------|--------------|-------------|
| MNIST_Classificador (MLP) | ~97% | ~5 min | ⭐ Fácil |
| CNN_Classificador | ~99% | ~10 min | ⭐⭐ Médio |
| Regressao_MLP | MAE ~3.5 | ~3 min | ⭐ Fácil |
| Regularizacao_Visual | N/A (educacional) | ~5 min | ⭐⭐ Médio |

---

## 🎯 Experimentos Sugeridos

### Para MNIST_Classificador
- [ ] Treinar com diferentes números de camadas
- [ ] Variar número de neurônios (64, 128, 256)
- [ ] Testar diferentes dropout rates (0.1, 0.3, 0.5)
- [ ] Comparar otimizadores (adam, sgd, rmsprop)

### Para CNN_Classificador
- [ ] Adicionar mais camadas convolucionais
- [ ] Variar tamanho de filtros (3×3, 5×5)
- [ ] Testar diferentes números de filtros (16, 32, 64)
- [ ] Implementar data augmentation

### Para Regressao_MLP
- [ ] Comparar com regressão linear
- [ ] Testar diferentes funções de perda (mse, mae, huber)
- [ ] Avaliar impacto de normalização dos dados

### Para Regularizacao_Visual
- [ ] Combinar Dropout + L2
- [ ] Testar BatchNorm em diferentes posições
- [ ] Variar força da regularização

---

## 💻 Setup Inicial

### Instalar Dependências
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

### Verificar GPU (Opcional)
```python
import tensorflow as tf
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
```

### Template Básico de Projeto
```python
# 1. Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 2. Carregar dados
# ...

# 3. Preprocessar
# ...

# 4. Criar modelo
model = Sequential([...])
model.compile(optimizer='adam', loss='...', metrics=['...'])

# 5. Callbacks
callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

# 6. Treinar
history = model.fit(X_train, y_train, validation_split=0.2, 
                   epochs=100, callbacks=callbacks)

# 7. Avaliar
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## 📈 Visualizações Importantes

### Curvas de Aprendizado
```python
import matplotlib.pyplot as plt

def plotar_historico(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Treino')
    ax1.plot(history.history['val_loss'], label='Validação')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Treino')
    ax2.plot(history.history['val_accuracy'], label='Validação')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.show()

plotar_historico(history)
```

### Matriz de Confusão
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()
```

---

## 🔧 Troubleshooting

### Modelo não aprende (loss estável)
**Causas**:
- Learning rate muito alto/baixo
- Dados não normalizados
- Arquitetura inadequada

**Soluções**:
```python
# Reduzir learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), ...)

# Normalizar dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```

### Overfit severo (treino 99%, val 70%)
**Soluções**:
```python
# Adicionar Dropout
model.add(Dropout(0.5))

# Usar EarlyStopping
callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

# Reduzir complexidade
# Menos camadas ou neurônios
```

### Treino muito lento
**Soluções**:
- Aumentar batch size (32 → 64 → 128)
- Simplificar modelo
- Usar GPU se disponível
- Reduzir tamanho do dataset (para testes)

---

## ✅ Checklist de Aprendizado

### Após Completar TODOS os Projetos

**Conceitos**:
- [ ] Entendo diferença entre MLP e CNN
- [ ] Sei quando usar classificação vs regressão
- [ ] Compreendo overfitting e como evitar
- [ ] Entendo callbacks (EarlyStopping, ModelCheckpoint)

**Prática**:
- [ ] Treinei pelo menos 1 modelo do zero
- [ ] Plotei curvas de loss e accuracy
- [ ] Interpretei matriz de confusão
- [ ] Salvei e carreguei um modelo

**Técnicas**:
- [ ] Usei Dropout
- [ ] Usei BatchNormalization
- [ ] Experimentei diferentes arquiteturas
- [ ] Comparei performance de modelos

---

## 🎓 Certificado de Conclusão (Informal)

Quando completar todos os 4 projetos:

```
🏆 CERTIFICADO DE CONCLUSÃO 🏆

[Seu Nome] completou com sucesso:
✓ MNIST_Classificador (MLP)
✓ CNN_Classificador (CNN)
✓ Regressao_MLP
✓ Regularizacao_Visual

Data: __________

Próximos passos: Projetos Reais!
```

---

## 🚀 Próximos Passos

Após dominar estes projetos:

1. **Ir para ../Projetos_Reais/** - Aplicações deployadas
2. **Criar seu próprio projeto** usando templates
3. **Participar de competições Kaggle** (ex: Dogs vs Cats)
4. **Contribuir para projetos open source**

---

## 📚 Recursos Adicionais

**Datasets Similares**:
- Fashion MNIST (roupas)
- CIFAR-10 (imagens coloridas)
- IMDB (análise de sentimento)

**Desafios**:
- [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

**Visualizações**:
- [TensorFlow Playground](https://playground.tensorflow.org/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

---

**Lembre-se**: A melhor forma de aprender Deep Learning é **experimentando**. Mude hiperparâmetros, quebr arquiteturas, e observe os resultados!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_03*

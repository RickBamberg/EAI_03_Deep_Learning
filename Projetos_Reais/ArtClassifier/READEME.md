# 🎨 Classificador de Estilos Artísticos com Deep Learning

Este projeto usa Transfer Learning com a arquitetura MobileNetV2 para classificar imagens de pinturas em diferentes estilos artísticos. Ele inclui um notebook de treinamento/análise e uma interface web feita com Flask para uso interativo.

---

## 🧠 Objetivo

Criar um modelo de Deep Learning capaz de reconhecer o estilo artístico de uma pintura com base em sua aparência visual. Este projeto demonstra o uso de técnicas como:

- Transfer Learning (MobileNetV2)
- Data Augmentation
- Avaliação com Matriz de Confusão
- Interpretação de erros e comportamento do modelo
- Deploy com Flask

---

## 📁 Estrutura do Projeto

art_classifier/  
│  
├── app.py           # Aplicação Flask  
├── requirements.txt # Dependências do projeto  
├── README.md        # Documentação do projeto  
├── .gitignore  
│  
├── /models/ # Modelo e rótulos  
│ ├── art_style_classifier_best.keras  
│ └── class_names.json  
│  
├── /notebooks/  
│ └── ArtClassifier.ipynb # Treinamento e análise  
│  
├── /static/  
│ └── /uploads/ # Imagens enviadas via Flask  
│  
├── /templates/ # HTML da aplicação  
│ ├── index.html  
│ └── result.html  
  

---

## 🖼️ Dataset Utilizado

**Fonte:** [WikiArt - Painter by Numbers](https://www.kaggle.com/datasets/ipythonx/painter-by-numbers)  
O dataset foi organizado em pastas, com cada pasta representando um estilo artístico, por exemplo:


/dataset/
├── impressionism/
├── cubism/
├── surrealism/


Foram escolhidos 6 estilos com número balanceado de imagens.

---

## 🏗️ Treinamento do Modelo

O modelo usa a arquitetura MobileNetV2 com pesos pré-treinados no ImageNet. A "cabeça" foi substituída por camadas específicas para a tarefa.

Principais técnicas utilizadas:

- Redimensionamento para 224x224
- Normalização (pré-processamento da MobileNetV2)
- Data Augmentation (flip, zoom, rotação)
- EarlyStopping e ModelCheckpoint

**Resultado Final:**  
Acurácia no conjunto de validação: **~65%**

---

## 📊 Análise dos Erros e Insights

### Insight Principal: Generalização vs. Memorização

Um exemplo importante foi a obra barroca _View of Scheveningen Sands_, classificada incorretamente como Impressionista.  
Análise revelou que:

> O modelo aprendeu a associar "céus vastos e luminosos" com o estilo impressionista — um erro compreensível, pois se baseia em características visuais gerais, não no contexto histórico.

![View of Scheveningen Sands](caminho/para/uma/imagem/de_exemplo.jpg)

Esse comportamento mostra que o modelo **está generalizando com base em padrões visuais**, não apenas memorizando rótulos.

---

## 🧪 Avaliação com Matriz de Confusão

A matriz de confusão revelou maior confusão entre:

- Barroco vs. Impressionismo (19 erros)
- Expressionismo vs. Expressionismo Abstrato (12 erros)

Esses resultados são coerentes com as semelhanças visuais entre os estilos.

---

## 🌐 Interface Flask

A aplicação permite ao usuário enviar uma imagem de uma pintura e receber o estilo artístico previsto.

### Como usar:

1. Clone o repositório:

   git clone https://github.com/seu_usuario/art-classifier.git
   cd art-classifier

2. Instale as dependências:

   pip install -r requirements.txt

3. Inicie o servidor Flask:

   python app.py

4. Acesse via navegador: 

   http://localhost:5000

📦 Requisitos
Veja requirements.txt para a lista completa.

📄 Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.


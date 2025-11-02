# 💖 Análise de Doenças Cardíacas - Dashboard Interativo

Projeto de análise de dados sobre doenças cardíacas utilizando machine learning supervisionado e não supervisionado, com interface web interativa desenvolvida em Streamlit.

## 📋 Descrição

Este projeto oferece uma aplicação web completa para análise de dados de doenças cardíacas, incluindo:

- **📑 Relatório Automático**: Insights automáticos sobre balanceamento, correlações, outliers e recomendações
- **📊 Análise Exploratória (EDA)**: Visualizações interativas com histogramas, boxplots, violinos, correlações e PCA 3D
- **💖 Modelo Supervisionado**: RandomForest com interface para predição de risco cardíaco
- **🧠 Modelo Não Supervisionado**: KMeans para agrupamento de pacientes por similaridade

## 🚀 Como Iniciar o Projeto

### Pré-requisitos

- Python 3.11
- pip (gerenciador de pacotes Python)

### Passo a Passo

1. **Clone o repositório ou navegue até a pasta do projeto**
   ```bash
   cd caminho/para/o/projeto
   ```

2. **Crie um ambiente virtual (recomendado)**
   ```bash
   python -m venv venv
   ```
   
   **Ative o ambiente virtual:**
   - No Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - No Windows (CMD):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - No Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Certifique-se de que o arquivo `heart.csv` está na pasta do projeto**

5. **Execute a aplicação Streamlit**
   ```bash
   streamlit run app.py
   ```

6. **Acesse a aplicação**
   - O Streamlit abrirá automaticamente no navegador em `http://localhost:8501`
   - Se não abrir automaticamente, copie a URL exibida no terminal e cole no navegador

## 📦 Dependências

O arquivo `requirements.txt` contém todas as bibliotecas necessárias com versões específicas testadas:

- `streamlit==1.39.0`: Framework para criação de aplicações web interativas
- `pandas==2.2.2`: Manipulação e análise de dados
- `numpy==1.26.4`: Operações numéricas
- `scikit-learn==1.4.2`: Machine learning (RandomForest, KMeans, PCA, etc.)
- `scipy==1.11.4`: Operações científicas e estatísticas avançadas
- `matplotlib==3.8.4`: Visualizações básicas
- `seaborn==0.13.2`: Visualizações estatísticas avançadas
- `plotly==5.23.0`: Gráficos interativos 3D e dinâmicos

**Nota:** As versões foram fixadas para garantir compatibilidade e reprodutibilidade do ambiente.

## 📁 Estrutura do Projeto

```
TRABALHO - FINAL/
├── app.py              # Aplicação principal Streamlit
├── heart.csv           # Dataset de doenças cardíacas (obrigatório)
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo
```

## 🎯 Funcionalidades

### 1. Relatório Automático
- Análise de balanceamento do conjunto de dados
- Top correlações com a variável alvo
- Identificação de variáveis categóricas mais relevantes
- Detecção de outliers
- Recomendações de modelagem

### 2. Análise Exploratória
- Estatísticas descritivas
- Distribuições e comparações por diagnóstico
- Matriz de correlação
- Gráficos de dispersão com tendência
- Análise de variáveis categóricas
- Visualização PCA 3D

### 3. Modelo Supervisionado (RandomForest)
- Interface para ajuste de hiperparâmetros (n_estimators, max_depth)
- Predição de risco cardíaco com base em características do paciente
- Métricas de desempenho (Accuracy, ROC AUC, Matriz de Confusão)
- Ajuste automático de limiar (Youden/ROC)
- **Curva ROC interativa** para análise de diferentes limiares
- **Curva Precisão-Recall** para avaliação de desempenho
- **Distribuição de Probabilidades** por classe real
- **Análise de Calibração** do classificador
- **Varredura de Limiar** para otimização de métricas
- **Importância de Atributos** (Permutation Importance) - Top 15 features

### 4. Modelo Não Supervisionado (KMeans)
- Sugestão automática de número de clusters (Silhouette Score)
- Visualização PCA 2D dos clusters
- Cálculo de risco médio por cluster
- Predição de cluster e risco para novos pacientes
- **Análise de Silhouette por Amostra** (gráfico de barras)
- **Perfil dos Clusters** com z-scores normalizados para variáveis numéricas

## 🛠️ Solução de Problemas

### Erro: "FileNotFoundError: Arquivo 'heart.csv' não encontrado"
- Certifique-se de que o arquivo `heart.csv` está na mesma pasta que `app.py`

### Erro ao instalar dependências
- Atualize o pip: `python -m pip install --upgrade pip`
- Instale as dependências novamente: `pip install -r requirements.txt`

### Porta 8501 já em uso
- Feche outras instâncias do Streamlit ou use uma porta diferente:
  ```bash
  streamlit run app.py --server.port 8502
  ```

## 📝 Notas

- O dataset `heart.csv` é necessário para executar a aplicação
- A primeira execução pode levar alguns segundos devido ao processamento inicial
- Os modelos são treinados em tempo real com base no dataset fornecido
- O uso de versões específicas no `requirements.txt` garante reprodutibilidade dos resultados
- Todas as visualizações são interativas e podem ser exploradas diretamente no navegador

## 👤 Autor

Iago Vieira da Silva | Trabalho Final - Análise de Dados de Doenças Cardíacas

---

**Desenvolvido com ❤️ usando Streamlit e scikit-learn**


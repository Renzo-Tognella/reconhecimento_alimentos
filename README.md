# Sistema de Reconhecimento de Alimentos

[![Versão do Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Versão do TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![Licença: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema avançado de reconhecimento de alimentos que identifica diferentes tipos de comida a partir de imagens. O projeto utiliza uma Rede Neural Convolucional (CNN) para classificação, juntamente com técnicas avançadas de processamento de imagem para segmentação e extração de características.

![Demonstração da Detecção de Alimentos](SCR-20250624-rsyw.png)

## Funcionalidades

-   **Segmentação de Alimentos**: Isola os itens alimentares do fundo da imagem usando processamento avançado de imagem.
-   **Detecção de Prato**: Identifica automaticamente o prato na imagem para focar na área relevante.
-   **Reconhecimento de Múltiplos Alimentos**: Classifica vários itens alimentares presentes em uma única imagem.
-   **Modelo CNN Avançado**: Um modelo CNN robusto treinado com técnicas de aumento de dados (data augmentation) específicas para imagens de alimentos.
-   **Interface Web**: Uma interface web simples para fazer upload de uma imagem e obter as predições.
-   **Análise Detalhada de Métricas**: Um conjunto completo de ferramentas para avaliar o desempenho do modelo, incluindo matriz de confusão, curvas ROC e relatórios detalhados.

## Tecnologias Utilizadas

-   **Backend**: Python, Flask
-   **Machine Learning**: TensorFlow, Keras, Scikit-learn
-   **Processamento de Imagem**: OpenCV, Pillow
-   **Frontend**: HTML, CSS (via templates)
-   **Visualização de Dados**: Matplotlib, Seaborn

## Começando

### Pré-requisitos

-   Python 3.9 ou superior
-   Gerenciador de pacotes pip

### Instalação

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/Renzo-Tognella/reconhecimento_alimentos.git
    cd reconhecimento_alimentos
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows, use `.venv\Scripts\activate`
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements_metrics.txt
    ```

4.  **Baixe o dataset:**

    O dataset não está incluído no repositório. Você precisa baixá-lo e colocá-lo no diretório apropriado. Crie um diretório chamado `Imagens_um_Alimento` e organize as imagens em subdiretórios para cada classe de alimento.

### Treinamento do Modelo

Para treinar o modelo do zero, execute o seguinte comando:

```bash
python cnn_alimentos_avancado.py
```

Este script irá:
-   Carregar as imagens e aplicar aumento de dados.
-   Treinar o modelo CNN.
-   Salvar o modelo treinado como `modelo_food_advanced_final.h5`.
-   Gerar uma análise de desempenho completa no diretório `metricas_diagnostico`.

### Executando a Aplicação

Para iniciar a aplicação web, execute:

```bash
python food_detector_web.py
```

Abra seu navegador e acesse `http://127.0.0.1:5000` para usar a aplicação.

## Estrutura do Projeto

```
.
├── .venv/                   # Ambiente virtual
├── Imagens_um_Alimento/     # Dataset para treinamento (Não está no repo)
├── metricas_diagnostico/    # Saída para métricas do modelo (Não está no repo)
├── templates/               # Templates HTML para a aplicação web
│   └── index.html
├── uploads/                 # Diretório para imagens enviadas pelo usuário (Não está no repo)
├── .gitignore               # Arquivos a serem ignorados pelo Git
├── cnn_alimentos_avancado.py  # Script para treinar a CNN
├── detector.py              # Lógica principal para detecção e segmentação de alimentos
├── food_detector_web.py     # Aplicação web com Flask
├── modelo_food_advanced_final.h5 # Modelo treinado (Não está no repo)
├── requirements_metrics.txt # Dependências Python
└── README.md                # Este arquivo
```

## Desempenho do Modelo

O modelo alcança alta acurácia na classificação de itens alimentares. Uma análise de desempenho detalhada é gerada automaticamente após o treinamento, incluindo:

-   **Matriz de Confusão**: Para visualizar o desempenho da classificação.
-   **Curvas ROC**: Para avaliar a capacidade do modelo de distinguir entre as classes.
-   **Relatório de Classificação**: Com precisão, recall e F1-score para cada classe.
-   **Dashboard de Desempenho**: Um resumo visual abrangente de todas as métricas.

Todos esses relatórios são salvos no diretório `metricas_diagnostico` após a execução do script de treinamento.

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para enviar um pull request.

1.  Faça um fork do repositório.
2.  Crie sua branch de funcionalidade (`git checkout -b feature/AmazingFeature`).
3.  Faça commit de suas alterações (`git commit -m 'Add some AmazingFeature'`).
4.  Faça push para a branch (`git push origin feature/AmazingFeature`).
5.  Abra um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes. 
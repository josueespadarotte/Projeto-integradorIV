# Projeto Integrador IV --- Assistente de IA para Segurança Rodoviária

O **Assistente de IA para Segurança Rodoviária** é um projeto acadêmico desenvolvido no curso de Sistemas de Informação da FAESA. A proposta é evoluir um painel de análise de acidentes rodoviários para uma aplicação que combine visualização de dados, análise estatística e inteligência artificial generativa, facilitando a consulta de informações relacionadas à segurança nas rodovias do ES.

A solução tem como foco apoiar gestores, analistas e profissionais envolvidos na segurança viária, permitindo explorar dados e, como evolução do projeto, consultar informações por meio de linguagem natural.

## Demonstração

**Acesse a demonstração:** [Visualizar funcionamento da aplicação](https://youtu.be/2ykeDIMSSpA?is=ZrXmsX4XUnEJHecK)

> O link será disponibilizado após a publicação da demonstração.
> Enquanto isso, a aplicação pode ser executada localmente seguindo as
> instruções deste documento.

## Objetivos

### Objetivo geral

Desenvolver um assistente inteligente para apoiar a análise de acidentes rodoviários, facilitando a exploração de dados e a obtenção de informações relevantes para a segurança viária.

### Objetivos específicos

-   Disponibilizar visualizações e análises exploratórias de dados de acidentes rodoviários.
-   Facilitar a identificação de padrões e fatores associados aos acidentes.
-   Explorar modelos de aprendizado de máquina para análise e predição de risco.
-   Desenvolver uma interface acessível para consulta e interpretação das informações.
-   Investigar a utilização de IA generativa para responder perguntas em linguagem natural.
-   Avaliar a qualidade, a relevância e a confiabilidade das respostas geradas pela IA.

## Funcionalidades

O projeto reúne funcionalidades de análise de dados e prevê a evolução para uma interface conversacional com inteligência artificial.

- **Dashboard de acidentes**: Visualização e exploração de informações sobre acidentes rodoviários.
- **Mapa de calor**: Identificação visual de padrões e concentrações de ocorrências.
- **Análise estatística**: Exploração de características dos acidentes e comparação entre grupos de dados.
- **Filtros de consulta**: Segmentação das análises por critérios disponíveis na aplicação, como município, rodovia, clima, tipo de pista e horário.
- **Predição de risco** : Área destinada à análise de risco por meio de modelos de aprendizado de máquina.
- **Assistente de IA**: Funcionalidade planejada para permitir consultas em linguagem natural sobre dados e documentos relacionados à segurança rodoviária.

## Tecnologias 
| Tecnologia | Papel no projeto |
|---|---|
| **Python** | Linguagem principal do backend, frontend e pipeline de IA. |
| **Streamlit** | Construção do dashboard e da interface de chat. |
| **FastAPI** | API RESTful que expõe os endpoints `/health` e `/chat`. |
| **Pydantic** | Validação dos dados de entrada e saída da API. |
| **Uvicorn** | Servidor ASGI para executar a aplicação FastAPI. |
| **pandas** | Leitura, filtragem e cálculo de estatísticas sobre o CSV de acidentes. |
| **scikit-learn / XGBoost / joblib** | Treinamento e carregamento do modelo de predição de risco (`modelo_risco_acidente.pkl`). |
| **FAISS** | Índice vetorial para busca semântica eficiente. |
| **Sentence Transformers** | Geração de embeddings com o modelo `tardellirs/brazembed-pt-br-retrieval`. |
| **LangChain Text Splitters** | Divisão dos documentos em chunks para indexação. |
| **GroqCloud / Groq SDK** | Acesso ao LLM para geração das respostas. |
| **Plotly** | Gráficos interativos no dashboard. |
| **pytest** | Execução dos testes automatizados do backend. |


# Como Executar

## Pré-requisitos

Para executar o projeto localmente, são necessários:

- **Python 3** instalado (recomenda-se 3.10 ou superior).
- Ambiente virtual Python.
- Dependências declaradas em `requirements.txt`.
- Arquivo `.env` na raiz do projeto com a chave da API do GroqCloud.
- Arquivos de dados (`data/prf_es_clean.csv`) e modelos (`models/*.pkl`) presentes no repositório.

A versão exata do Python deve ser confirmada no ambiente de desenvolvimento.

## 1. Clonar o repositório

```bash
git clone https://github.com/josueespadarotte/Projeto-integradorIV.git
cd Projeto-integradorIV
```

## 2. Criar e ativar o ambiente virtual

No Linux, macOS ou WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Instalar as dependências

Com o ambiente virtual ativado:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto com a chave da API do GroqCloud:

```env
GROQ_API_KEY=sua_chave_aqui
GROQ_MODEL=openai/gpt-oss-120b
```

> A chave pode ser obtida em <https://console.groq.com/keys>. Não versione o arquivo `.env`.

## 5. Construir o cache do RAG (obrigatório na primeira execução)

O assistente de IA utiliza um índice vetorial FAISS que precisa ser gerado a partir do CSV de acidentes antes de o backend iniciar. O cache fica em `data/rag_cache/` e não está versionado.

Se for necessário descartar um cache antigo:

```bash
rm -rf data/rag_cache
```

Em seguida, gere o índice (a primeira execução baixa o modelo de embeddings e pode demorar alguns minutos):

```bash
python -m backend.rag
```

## 6. Iniciar o backend FastAPI

Em um terminal, com o ambiente virtual ativado e a partir da raiz do projeto:

```bash
uvicorn backend.main:app --reload
```

O backend ficará disponível em:

```text
http://localhost:8000
```

## 7. Iniciar o frontend Streamlit

Em **outro terminal**, também na raiz do projeto e com o ambiente virtual ativado:

```bash
streamlit run frontend/app.py
```

O Streamlit disponibilizará a aplicação em:

```text
http://localhost:8501
```

> A aba **Assistente de IA** do frontend consome o backend em `http://localhost:8000`. Sem o backend ativo, apenas as abas de análise local (mapa, estatísticas e predição) funcionam.

## Executar os testes automatizados

```bash
pytest backend/test_main.py -v
```

## Ordem recomendada de execução

1. `python -m backend.rag` — constrói/atualiza o cache RAG.
2. `uvicorn backend.main:app --reload` — inicia o backend.
3. `streamlit run frontend/app.py` — inicia o frontend.


## Equipe

Projeto desenvolvido no curso de Sistemas de Informação da FAESA.

-   Josué Filipe Espadarotte
-   Laisa de Souza Camilo
-   Mikaelly do Nascimento Cardoso
-   Victória Sofia dos Santos Teixeira

**Instituição:** FAESA --- Curso de Sistemas de Informação\
**Professor:** M.Sc. Howard Cruz Roatti


------------------------------------------------------------------------

Projeto acadêmico desenvolvido com o objetivo de explorar o uso de análise de dados e inteligência artificial como ferramentas de apoio à segurança rodoviária do ES.

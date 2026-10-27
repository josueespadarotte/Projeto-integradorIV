# Projeto Integrador IV - Assistente de IA para Segurança Rodoviária (Evolução do Painel PRF)

**Alunos:** Josué Filipe Espadarotte, Laisa de Souza Camilo, Mikaelly do Nascimento Cardoso e Victoria Sofia dos Santos Teixeira

**Instituição:** FAESA - Curso de Sistemas de Informação

**Professor:** M.Sc. Howard Cruz Roatti

## 1. Escopo e Justificativa

A análise de grandes volumes de dados de acidentes rodoviários e a extração de *insights* táticos para a alocação de viaturas da Polícia Rodoviária Federal (PRF) exige tempo e conhecimento analítico avançado.

Este projeto visa evoluir o dashboard estatístico e preditivo construído anteriormente (baseado em modelos de *Ensemble* e XAI) para uma aplicação web interativa impulsionada por Inteligência Artificial Generativa. O escopo principal é a implementação de um Assistente de IA que utilizará a técnica de RAG (*Retrieval-Augmented Generation*) para permitir que gestores consultem a base de dados de acidentes e manuais de trânsito em linguagem natural, recebendo respostas precisas e ancoradas nos dados reais da comunidade.

## 2. Comunidade Impactada e Indicadores

A natureza extensionista deste projeto foca na segurança pública e na eficiência da gestão de tráfego.

* **Comunidade Impactada:** Gestores, analistas de dados e agentes da Polícia Rodoviária Federal (PRF) atuantes no Espírito Santo, além da sociedade civil impactada indiretamente pela melhoria na prevenção de acidentes nas rodovias capixabas.
* Indicadores de Impacto:


* **Quantitativo:** Redução percentual do tempo médio necessário para a extração de relatórios táticos e cruzamento de dados de acidentes.
* **Quantitativo (Eval da IA):** Taxa de precisão e relevância das respostas do modelo (mitigação de alucinações), medida através de um *harness de eval*.


* **Qualitativo:** Nível de satisfação e usabilidade (*Feedback Score*) dos agentes de segurança ao utilizar a interface conversacional em comparação com a análise manual de planilhas.



## 3. Metodologia e Plano de Trabalho

O desenvolvimento seguirá boas práticas de Engenharia de Software e IA Responsável, dividido nas seguintes camadas:

* **Frontend:** Desenvolvimento de uma interface web responsiva e intuitiva utilizando Streamlit.


* **Backend:** Construção de uma API RESTful em Python (FastAPI) para orquestrar a lógica de negócios e o roteamento dos dados.


* **Componente de IA:** Integração com LLMs de código aberto (família Llama) através da API do GroqCloud (garantindo alta velocidade de inferência e custo zero de infraestrutura). A IA será potencializada com RAG (engastando os dados estatísticos limpos e manuais de infração).


* **Engenharia de Qualidade:** Implementação de testes automatizados (unitários e de integração) nas rotas do backend e avaliação contínua das métricas de qualidade do LLM (robustez e taxa de erro).


* **Deploy:** Publicação do MVP (Minimum Viable Product) em nuvem (*free tier*), utilizando plataformas como Render ou Railway.



## 4. IA Responsável e Governança

A solução será desenvolvida respeitando os pilares de governança exigidos:

* **Privacidade e LGPD:** Os dados de acidentes utilizados como base de conhecimento (*context window*) passarão por processos rígidos de anonimização, garantindo que nenhuma informação sensível ou dado pessoal das vítimas seja exposto.


* **Segurança:** Implementação de proteções contra *prompt injection* na camada de backend, garantindo que o assistente responda exclusivamente sobre o domínio de segurança rodoviária.


* **Consciência de Custo:** Monitoramento do limite de requisições e *tokens* da API do GroqCloud para manter a viabilidade econômica do projeto no *free tier*.

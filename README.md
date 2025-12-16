# 🔍 Assistente de Auditoria de Compliance – Dunder Mifflin

Este projeto implementa um assistente de auditoria de compliance utilizando RAG (Retrieval-Augmented Generation) para analisar documentos internos (políticas, e-mails e transações) e responder perguntas relacionadas a compliance, conspirações e possíveis fraudes na empresa Dunder Mifflin.

A interface é construída com Streamlit, o armazenamento vetorial com ChromaDB, embeddings com Sentence Transformers, e a geração de respostas via OpenRouter API.

---

## 🧠 Arquitetura do Sistema

### Visão geral

```

Usuário
↓
Streamlit (Chat UI)
↓
Pipeline RAG
├─ SentenceTransformer (embeddings)
├─ ChromaDB (busca vetorial)
└─ OpenRouter (LLM)
↓
Resposta com evidências dos documentos

```

### Componentes

- **Interface (Streamlit)**
  - Chat interativo para envio de perguntas

- **Indexação de Documentos**
  - Leitura de arquivos `.txt` e `.csv` da pasta `dataset/`
  - Quebra de texto em chunks
  - Geração de embeddings semânticos

- **Base Vetorial (ChromaDB)**
  - Armazena embeddings e textos
  - Permite recuperação semântica dos documentos relevantes

- **Modelo de Linguagem (OpenRouter)**
  - Recebe a pergunta + documentos recuperados
  - Gera respostas baseadas nas regras de compliance

---

## 📁 Estrutura do Projeto

```

.
├── dataset/            # Documentos (txt / csv)
├── assets/             # Avatares
├── src/                # Código do projeto
    ├── front.py        # Código do streamlit + I.A.
    └── .env            # Variáveis de ambiente
├── requirements.txt    # Dependências
└── README.md

````

---

## ▶️ Como Executar o Projeto

### 1. Clonar o repositório
```bash
git clone <url-do-repositorio>
cd <nome-do-projeto>
````

### 2. Criar ambiente virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Crie um arquivo `.env` na pasta `src/`:

```env
API_KEY=coloque_sua_chave_aqui
```

---

### 5. Executar a aplicação

```bash
streamlit run src/front.py
```

---

## 🎥 Vídeos de Demonstração

[Código](https://youtu.be/AOsc3Y7wbPk)

[Demonstração](https://youtu.be/IC0JzM1PaBs)

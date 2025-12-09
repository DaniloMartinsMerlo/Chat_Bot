import streamlit as st
import time
import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import textwrap
import pandas as pd
from typing import Dict, Any, Tuple, List
from functools import lru_cache

# ================ CONFIGURAÇÕES INICIAIS ================

load_dotenv()
API_KEY = os.getenv("API_KEY")
DOCS_PATH = "dataset"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MODEL = "z-ai/glm-4.5-air:free"

# ================ INICIALIZAÇÃO OTIMIZADA ================

@st.cache_resource
def initialize_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def initialize_chroma():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="docs")
    return client, collection

embedder = initialize_embedder()
chroma_client, collection = initialize_chroma()

# ================ FUNÇÕES DE INDEXAÇÃO OTIMIZADAS ================

def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    return textwrap.wrap(text, max_chars, break_long_words=False, replace_whitespace=False)

def add_text_batch(doc_id_prefix: str, text: str):
    chunks = chunk_text(text)
    
    if not chunks:
        return
    
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    
    documents = chunks
    embedding_list = [emb.tolist() for emb in embeddings]
    ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=documents,
        embeddings=embedding_list,
        ids=ids
    )

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_csv(path: str, filename: str):
    df = pd.read_csv(path, low_memory=False)
    text = df.head(1000).to_string() 
    add_text_batch(filename, text)

@st.cache_data
def check_documents_loaded():
    try:
        count = collection.count()
        return count > 0
    except:
        return False

def load_all_documents(docs_path: str, force_reload: bool = False):
    if not os.path.exists(docs_path):
        return
    
    if not force_reload and check_documents_loaded():
        return

    global chroma_client, collection
    chroma_client.delete_collection(name="docs")
    collection = chroma_client.get_or_create_collection(name="docs")

    files = os.listdir(docs_path)
    
    for filename in enumerate(files):
        full_path = os.path.join(docs_path, filename)

        if filename.endswith(".csv"):
            load_csv(full_path, filename)
        elif filename.endswith(".txt"):
            text = load_txt(full_path)
            add_text_batch(filename, text)
        
load_all_documents(DOCS_PATH)

# ================ OPENROUTER API OTIMIZADA ================

def call_openrouter(payload: Dict[str, Any], timeout: int = 60) -> Tuple[Dict[str, Any], requests.Response]:
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=HEADERS,
            json=payload, 
            timeout=timeout
        )

        body = resp.json()

        if resp.status_code != 200:
            error_message = body.get("error", {}).get("message", "Erro desconhecido na API.")
            raise Exception(f"Erro na API (Status {resp.status_code}): {error_message}")

        return body, resp
    
    except requests.exceptions.Timeout:
        raise Exception("Timeout na requisição à API. Tente novamente.")
    except ValueError:
        raise Exception(f"Erro na API: Resposta não é JSON. Status: {resp.status_code}")

# ================ LÓGICA DE RESPOSTA OTIMIZADA ================

@lru_cache(maxsize=100)
def get_cached_embedding(question: str):
    return embedder.encode([question])[0]

def rag_query(question: str) -> str:
    embedding = get_cached_embedding(question)

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=10,
    )

    retrieved_text = "\n\n".join(results["documents"][0])

    prompt = f"""
        Você é o **Assistente de Auditoria de Compliance da Dunder Mifflin**, trabalhando para Toby Flenderson (RH).

        Sua missão é analisar documentos da empresa (políticas, e-mails, transações) e responder perguntas investigativas com PRECISÃO e EVIDÊNCIAS.


        **SUAS CAPACIDADES DE ANÁLISE: Faça apenas aquelas solicitadas pelo usuário**

        **CONSULTAS SOBRE POLÍTICAS DE COMPLIANCE**
        - Responda dúvidas dos colaboradores sobre regras, limites e procedimentos
        - Cite trechos específicos da política quando relevante
        - Seja claro, didático e completo

        **INVESTIGAÇÃO**
        - Vasculhe e-mails procurando evidências de conspiração
        - Para CADA e-mail suspeito, liste:
            * Remetente → Destinatário
            * Trecho específico do e-mail
            * Por que é evidência de conspiração
        - Conclusão final: "SIM, há evidências" ou "NÃO, não há evidências"

        **VIOLAÇÕES DIRETAS DE COMPLIANCE**
        - Identifique transações que SOZINHAS violam as políticas
        - Tipos de violação:
            * Valores acima dos limites permitidos
            * Categorias proibidas/restritas
            * Aprovações ausentes quando obrigatórias
            * Frequência/padrão suspeito
        - Para CADA violação, liste:
            * ID da transação
            * Funcionário, valor, categoria
            * Regra específica violada (cite a política)
            * Gravidade (baixa/média/alta)

        **FRAUDES COM CONTEXTO DE E-MAILS**
        - Correlacione e-mails com transações para detectar fraudes combinadas
        - Procure por:
            * E-mails combinando desvios + transação correspondente
            * Acordos para burlar políticas + evidência nas transações
            * Padrões de conspiração financeira entre funcionários
        - Para CADA fraude, forneça:
            * **E-mail:** [Remetente → Destinatário, trecho]
            * **Transação:** [ID, valor, categoria, funcionário]
            * **Conexão:** Como o e-mail comprova a fraude
            * **Gravidade:** baixa/média/alta


        **REGRAS IMPORTANTES:**

        Seja DETALHADO e forneça EVIDÊNCIAS CONCRETAS sempre
        Cite: trechos de políticas, IDs de transações, remetentes de e-mails
        Use formatação clara (tópicos, negrito) para organizar informações
        Se não houver dados suficientes, seja honesto: "Não encontrei evidências dessa violação nos documentos analisados."
        Analise TODOS os documentos recuperados, não apenas alguns

        Nunca invente dados ou transações que não estão nos documentos
        Não faça suposições sem evidências concretas


        **DOCUMENTOS RECUPERADOS:**
        {retrieved_text}


        **PERGUNTA DE INVESTIGAÇÃO:**
        {question}


        **Responda agora com base nos documentos, fornecendo evidências específicas e organizadas:**"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
    }

    body, _ = call_openrouter(payload, timeout=60)
    return body["choices"][0]["message"]["content"]

def general_query(question: str) -> str:

    prompt = f"""
        Você é o Assistente de Auditoria de Compliance da Dunder Mifflin,
        mas nesta resposta você deve **ignorar totalmente o modo de auditoria**.

        O usuário fez uma pergunta geral, NÃO relacionada a investigação, compliance ou documentos.

        Responda de forma:
        - curta
        - direta
        - clara
        - sem listar regras completas de auditoria
        - sem iniciar processos investigativos

        Explique APENAS o que foi perguntado de maneira simples e profissional.

        PERGUNTA: {question}
    """

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    }

    body, _ = call_openrouter(payload, timeout=30)
    return body["choices"][0]["message"]["content"]

@lru_cache(maxsize=50)
def classify_intent_cached(question: str) -> str:
    
    prompt = f"""
        Você deve classificar a pergunta do usuário em APENAS uma palavra:
        - "general"
        - "compliance"

        REGRAS:
        1. Se a pergunta for sobre você, suas habilidades, como você funciona, o que é capaz de fazer, limitações ou qualquer dúvida METALINGUÍSTICA → responda "general".
        2. Só classifique como "compliance" quando o usuário pedir para:
        - analisar documentos
        - investigar transações
        - investigar e-mails
        - detectar violações
        - explicar políticas da empresa
        - executar qualquer tarefa investigativa do sistema de auditoria
        3. Não classifique perguntas gerais como compliance.

        PERGUNTA: {question}
        Responda APENAS: general ou compliance.
    """

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
    }

    try:
        body, _ = call_openrouter(payload, timeout=15)
        intent = body["choices"][0]["message"]["content"].strip().lower()
        return "compliance" if intent not in ["general"] else intent
    except Exception:
        return "compliance"

def get_assistant_response(question: str) -> str:
    intent = classify_intent_cached(question)
    
    if intent == "general":
        return general_query(question)
    else:
        return rag_query(question)

# ================ FUNÇÕES DE EXIBIÇÃO DO STREAMLIT ================

def stream_text(text: str):
    for char in text:
        yield char
        time.sleep(0.005)  

def user_text(input_text: str) -> str:
    st.session_state["messages"].append({
        "role": "user", 
        "content": input_text, 
        "avatar": "assets/thunderatz.png"
    })
    
    with st.chat_message("user", avatar="assets/thunderatz.png"):
        st.write(input_text)
    
    return input_text

def ia_response(response: str):
    st.session_state["messages"].append({
        "role": "assistant", 
        "content": response, 
        "avatar": "🤖"
    })
    
    with st.chat_message("assistant", avatar="🤖"):
        st.write_stream(stream_text(response))

# ================ APLICAÇÃO STREAMLIT PRINCIPAL ================

st.set_page_config(
    page_title="Assistente de Auditoria - Dunder Mifflin",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Assistente de Auditoria de Compliance")
st.markdown("### Dunder Mifflin - Filial Scranton")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar=msg.get("avatar", "🤖")):
        st.write(msg["content"])

input_box = st.chat_input(
    placeholder="Digite sua pergunta sobre compliance...",
    key="chat_input"
)

if input_box:
    user_text(input_box)
    
    with st.spinner("Analisando..."):
        try:
            content = get_assistant_response(input_box)
        except Exception as e:
            content = f"Desculpe, ocorreu um erro: {str(e)}"
            st.error(content)
    
    ia_response(content)
    st.rerun()
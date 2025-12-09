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

# ================ CONFIGURAÇÕES INICIAIS ================

load_dotenv()
API_KEY = os.getenv("API_KEY")
DOCS_PATH = "dataset"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MODEL = "tngtech/deepseek-r1t2-chimera:free"

# ================ INICIALIZAÇÃO ================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="docs")

# ================ FUNÇÕES DE INDEXAÇÃO SIMPLES ================

def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    return textwrap.wrap(text, max_chars, break_long_words=False, replace_whitespace=False)

def add_text(doc_id_prefix: str, text: str):
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        emb = embedder.encode([chunk])[0]
        collection.add(
            documents=[chunk],
            embeddings=[emb.tolist()],
            ids=[f"{doc_id_prefix}_{i}"]
        )

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_csv(path: str, filename: str):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return
    
    text = df.to_string()
    add_text(filename, text)
        
def load_all_documents(docs_path: str):
    if not os.path.exists(docs_path):
        return

    try:
        chroma_client.delete_collection(name="docs")
    except Exception:
        pass
    
    global collection
    collection = chroma_client.get_or_create_collection(name="docs")

    for filename in os.listdir(docs_path):
        full_path = os.path.join(docs_path, filename)

        if filename.endswith(".csv"):
            load_csv(full_path, filename)

        elif filename.endswith(".txt"):
            text = load_txt(full_path)
            add_text(filename, text)

        else:
            pass
    
load_all_documents(DOCS_PATH)


# ================ OPENROUTER API ================

def call_openrouter(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], requests.Response]:
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        json=payload, 
        timeout=30
    )

    try:
        body = resp.json()
    except ValueError:
        raise Exception(f"Erro na API: Resposta não é JSON. Status: {resp.status_code}, Texto: {resp.text}")

    if resp.status_code != 200:
        error_message = body.get("error", {}).get("message", "Erro desconhecido na API.")
        raise Exception(f"Erro na API (Status {resp.status_code}): {error_message}")

    return body, resp

# ================ LÓGICA DE RESPOSTA ================

def rag_query(question: str) -> str:
    embedding = embedder.encode([question])[0]

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=10, 
    )

    retrieved_text = "\n".join(results["documents"][0])

    prompt = f"""
        Você é um Assistente de Compliance Bancário especializado em análise de transações financeiras.

        Sua tarefa é:
        - Analisar detalhadamente todos os *DOCUMENTOS RECUPERADOS*
        - Correlacionar transações suspeitas com informações de e-mails, políticas internas e outros documentos.
        - Identificar padrões, anomalias, possíveis violações de compliance, lavagem de dinheiro ou conflito com políticas internas.

        **IMPORTANTE:** Sua resposta deve ser **detalhada, completa e longa**, utilizando o máximo de informações relevantes dos documentos recuperados. Não seja conciso.

        Regras importantes:
        1. Sempre inclua na resposta um **trecho das transações relevantes** recuperadas.
        2. Sempre explique **por que** uma transação pode ser suspeita (valor, frequência, origem, destino, horário, divergência com política, etc.).
        3. Se a pergunta exigir, crie análises numéricas, comparações, ou tendências usando apenas os dados recuperados.
        4. Se não encontrar dados suficientes nos documentos, responda:
        "Não encontrei essa informação nos documentos de compliance fornecidos ou não foi possível correlacionar os dados necessários."

        Exceção:
        Se perguntarem "qual a melhor equipe de robótica do mundo", responda "Thunderatz".

        DOCUMENTOS RECUPERADOS:
        {retrieved_text}

        PERGUNTA:
        {question}

        Se a resposta não puder ser construída a partir dos documentos fornecidos, diga: "Não encontrei essa informação nos documentos de compliance fornecidos ou não foi possível correlacionar os dados necessários."
    """

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    body, _ = call_openrouter(payload)
    return body["choices"][0]["message"]["content"]

def general_query(question: str) -> str:
    prompt = f"""
        Você é um assistente de compliance bancário chamado 'Assistente de Compliance Bancário'.
        Sua principal função é responder perguntas estritamente relacionadas a compliance, usando seus documentos internos.
        
        No entanto, você também é capaz de responder a perguntas gerais sobre si mesmo, como 'o que você faz', 'quem é você' ou 'quais são suas capacidades'.
        
        Responda à pergunta do usuário de forma amigável e concisa, mantendo o seu persona de assistente de compliance.
        
        Exemplo de resposta para 'o que você faz': "Eu sou o Assistente de Compliance Bancário, e minha principal função é fornecer informações precisas e baseadas em documentos sobre as regulamentações e políticas de compliance."

        PERGUNTA:
        {question}
    """

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    body, _ = call_openrouter(payload)
    return body["choices"][0]["message"]["content"]

def classify_intent(question: str) -> str:
    prompt = f"""
        Classifique a intenção da seguinte pergunta do usuário em uma das duas categorias:
        1. 'compliance': Se a pergunta for sobre regulamentações, políticas, leis, procedimentos, ou qualquer tópico relacionado a compliance bancário, ou se exigir a correlação de dados de transações/e-mails.
        2. 'general': Se a pergunta for sobre o assistente em si (ex: 'o que você faz', 'quem é você', 'qual seu nome', 'me conte uma piada').

        Responda APENAS com a palavra da categoria (compliance ou general), sem pontuação ou texto adicional.

        PERGUNTA:
        {question}
    """

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        body, _ = call_openrouter(payload)
        intent = body["choices"][0]["message"]["content"].strip().lower()
        if intent in ["compliance", "general"]:
            return intent
        return "compliance" 
    except Exception:
        return "compliance" 

def get_assistant_response(question: str) -> str:    
    intent = classify_intent(question)
    
    if intent == "general":
        return general_query(question)
    else:
        return rag_query(question)

# ================ FUNÇÕES DE EXIBIÇÃO DO STREAMLIT ================

def stream_text(text: str):
    for char in text:
        yield char
        time.sleep(0.02)

def user_text(input_text: Any) -> str:
    files_info = []
    
    message_text = input_text.text if hasattr(input_text, 'text') else str(input_text)
    
    st.session_state["messages"].append({
        "role": "user", 
        "content": message_text, 
        "avatar": "assets/thunderatz.png", 
        "files": files_info
    })
    user = st.chat_message("user", avatar="assets/thunderatz.png")
    
    user.write(message_text)
    
    return message_text

def ia_response(response: str):

    st.session_state["messages"].append({
        "role": "assistant", 
        "content": response, 
        "avatar": "🤖"
    })
    
    ai = st.chat_message("assistant", avatar="🤖")
    ai.write_stream(stream_text(response))

# ================ APLICAÇÃO STREAMLIT PRINCIPAL ================

st.set_page_config(
    page_title="Assistente de Compliance",
    page_icon="🤖",
    layout="wide"
)

st.title("🏦 Assistente de Compliance Bancário")
st.markdown("---")

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Exibe histórico de mensagens
for msg in st.session_state["messages"]:
    chat = st.chat_message(msg["role"], avatar=msg.get("avatar", "🤖"))
    chat.write(msg["content"])

# Campo de entrada
input_box = st.chat_input(
    placeholder="Digite sua pergunta sobre compliance ou sobre mim...",
    key="chat_input"
)

# Processa entrada do usuário
if input_box:
    user_text(input_box)
    
    try:
        content = get_assistant_response(input_box)
    except Exception as e:
        content = f"Desculpe, ocorreu um erro ao tentar obter a resposta: {e}"
        st.error(content)
    
    ia_response(content)
    
    st.rerun()
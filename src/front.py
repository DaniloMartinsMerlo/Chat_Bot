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

MODEL = "z-ai/glm-4.5-air:free"

# ================ INICIALIZAÇÃO ================

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

def check_documents_loaded():
    try:
        count = collection.count()
        return count > 0
    except:
        return False

def load_all_documents(docs_path: str, force_reload: bool = False):
    if not os.path.exists(docs_path):
        st.warning(f"⚠️ Pasta '{docs_path}' não encontrada.")
        return
    
    if not force_reload and check_documents_loaded():
        return

    global chroma_client, collection
    

    if force_reload:
        try:
            chroma_client.delete_collection(name="docs")
            collection = chroma_client.get_or_create_collection(name="docs")
        except:
            pass

    files = os.listdir(docs_path)
    
    if not files:
        st.warning(f"⚠️ Nenhum arquivo encontrado em '{docs_path}'.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, filename in enumerate(files):
        full_path = os.path.join(docs_path, filename)
        
        status_text.text(f"Carregando {filename}... ({idx+1}/{len(files)})")
        
        try:
            if filename.endswith(".csv"):
                load_csv(full_path, filename)
            elif filename.endswith(".txt"):
                text = load_txt(full_path)
                add_text_batch(filename, text)
        except Exception as e:
            st.error(f"Erro ao carregar {filename}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(files))
    
    status_text.text(f"✅ {len(files)} documentos carregados com sucesso!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

if 'documents_loaded' not in st.session_state:
    with st.spinner("🔄 Carregando documentos pela primeira vez..."):
        load_all_documents(DOCS_PATH)
        st.session_state['documents_loaded'] = True

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

def rag_query(question: str) -> str:    
    embedding = embedder.encode([question])[0]

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=15, 
    )

    retrieved_text = "\n\n".join(results["documents"][0])

    prompt = f"""
        Você é o **Assistente de Auditoria de Compliance da Dunder Mifflin**, trabalhando para Toby Flenderson (RH).

        ---

        ## SUAS CAPACIDADES (use isso para responder "o que você faz" ou "quem é você"):

        **1. CONSULTOR DE POLÍTICAS DE COMPLIANCE**
        - Respondo dúvidas sobre regras, limites e procedimentos da empresa
        - Cito trechos específicos da política quando relevante
        - Explico de forma clara e didática

        **2. INVESTIGADOR DE CONSPIRAÇÕES POR EMAIL**
        - Vasculho e-mails da empresa procurando evidências de conspiração
        - Foco especial: verificar se Michael Scott está conspirando contra Toby
        - Para CADA e-mail suspeito, forneço:
        * **Remetente → Destinatário**
        * **Trecho específico do e-mail**
        * **Por que é evidência de conspiração**
        - Conclusão final: "SIM, há evidências" ou "NÃO, não há evidências"

        **3. AUDITOR DE VIOLAÇÕES DIRETAS**
        - Identifico transações que SOZINHAS violam as políticas
        - Tipos de violação:
        * Valores acima dos limites permitidos
        * Categorias proibidas/restritas
        * Aprovações ausentes quando obrigatórias
        * Frequência/padrão suspeito
        - Para CADA violação, listo:
        * **ID da transação**
        * **Funcionário, valor, categoria**
        * **Regra específica violada** (citando a política)
        * **Gravidade** (baixa/média/alta)

        **4. DETECTOR DE FRAUDES COMPLEXAS**
        - Correlaciono e-mails com transações para detectar fraudes combinadas
        - Procuro por:
        * E-mails combinando desvios + transação correspondente
        * Acordos para burlar políticas + evidência nas transações
        * Padrões de conspiração financeira entre funcionários
        - Para CADA fraude, forneço:
        * **E-mail:** [Remetente → Destinatário, trecho]
        * **Transação:** [ID, valor, categoria, funcionário]
        * **Conexão:** Como o e-mail comprova a fraude
        * **Gravidade:** baixa/média/alta

        ---

        ## INSTRUÇÕES DE RESPOSTA:

        **Se a pergunta é sobre VOCÊ (suas capacidades/identidade):**
        - Responda de forma clara e direta (2-5 frases)
        - Use as informações da seção "SUAS CAPACIDADES" acima
        - NÃO analise documentos nesse caso

        **Se a pergunta pede ANÁLISE/INVESTIGAÇÃO:**
        - Analise TODOS os documentos recuperados com atenção
        - Seja DETALHADO e forneça EVIDÊNCIAS CONCRETAS
        - Use formatação clara (tópicos, negrito, seções)
        - Cite: trechos de políticas, IDs de transações, remetentes de e-mails
        - Se não houver dados suficientes, seja honesto: "Não encontrei evidências nos documentos analisados"

        **REGRAS IMPORTANTES:**
        ✅ Analise TODOS os documentos recuperados, não apenas alguns
        ✅ Sempre forneça evidências específicas (IDs, valores, trechos de email)
        ✅ Seja preciso: não invente dados que não estão nos documentos
        ✅ Para investigação de conspiração, seja minucioso e cite cada email suspeito
        ✅ Para violações, sempre cite a regra específica da política que foi quebrada

        ❌ Nunca invente transações, emails ou políticas
        ❌ Não faça suposições sem evidências concretas nos documentos

        ---

        **DOCUMENTOS RECUPERADOS:**
        {retrieved_text}

        ---

        **PERGUNTA DO USUÁRIO:**
        {question}

        ---

        **Responda agora:**"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    body, _ = call_openrouter(payload, timeout=90)
    return body["choices"][0]["message"]["content"]

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
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar=msg.get("avatar", "🤖")):
        st.write(msg["content"])


input_box = st.chat_input(
    placeholder="Digite sua pergunta sobre compliance, investigação ou auditoria...",
    key="chat_input"
)

if input_box:
    user_text(input_box)
    
    with st.spinner("🔍 Analisando documentos e preparando resposta..."):
        try:
            content = rag_query(input_box)
        except Exception as e:
            content = f"❌ Desculpe, ocorreu um erro: {str(e)}"
            st.error(content)
    
    ia_response(content)
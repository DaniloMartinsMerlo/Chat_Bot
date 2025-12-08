from google.adk.agents.llm_agent import Agent

def compliance():
    try:
        with open("dataset/politica_compliance.txt", "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "Erro: Arquivo de política de compliance não encontrado"

def transacoes():
    try:
        with open("dataset/transacoes_bancarias.csv", "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "Erro: Arquivo de transações bancárias não encontrado"

def emails():
    try:
        with open("dataset/emails.txt", "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "Erro: Arquivo de emails não encontrado"

# Configuração do agente
root_agent = Agent(
    model='gemini-2.5-pro',
    name='compliance_agent',
    description='Assistente especializado em questões de compliance bancário.',
    instruction='''Você é um assistente especializado em compliance bancário. 
    Use as ferramentas disponíveis para acessar:
    - Política de compliance da empresa
    - Transações bancárias
    - Emails relacionados a compliance
    
    Responda de forma clara, objetiva e sempre baseada nas informações dos documentos.
    Se necessário, cite especificamente de onde veio a informação.''',
    tools=[compliance, transacoes, emails],
)
import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # volta uma pagina para importar alguma coisa dentro de outro arquivo

from models import LLMFactory
from Database.utils import VectorDatabaseHelper

def carregar_prompt(caminho_arquivo: str, nome_agente: str) -> str: # busca o caminho do prompt e qual o agente que vai trabalhar
    with open(caminho_arquivo, "r", encoding="utf-8") as file: # espera receber o enderenço do arquivo, faz a leitura e consideraça ç e outros
        schemas = yaml.safe_load(file) # salva na variavel schemas a consulta do arquivo yaml

    return schemas[nome_agente]["content"]

def executar_agente_ia(mensagem_usuario: str, provedor_ia: str = "gemini"): # executa o agente de IA buscando a mensagem do usuario e definindo default para gemini o agente
    caminho_prompt = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../Schemas/prompt_agente.yaml") # pega o diretorio atual junta com ".." para voltar a pasta e depois transforma em abosulto o caminho
    )

    prompt_sistema = carregar_prompt(caminho_prompt, "agente_suporte")

    db = VectorDatabaseHelper("banco_teste")
    documentos = db.buscar_contexto_relevante(mensagem_usuario) # Busca na lista fake os 3 documentos mais parecidos com a pergunta do usuário

    contexto_formatado = "\n".join(documentos)

    # gera um prompt que unifica o contexto com a mensagem atual do usuario
    prompt_enriquecido = f"""
    Contexto da Base de Conhecimento:
    {contexto_formatado}

    Pergunta do Usuário:
    {mensagem_usuario}
    """
    
    modelo_ia = LLMFactory.criar_modelo(provedor_ia) # cria um modelo de ia seguindo os parametros da herança

    resposta = modelo_ia.gerar_resposta(
        prompt_sistema=prompt_sistema,
        prompt_usuario=prompt_enriquecido
    )

    return resposta

    
if __name__ == "__main__":
    pergunta = "Como organizo minha infraestrutura de dados?"

    print("\n--- Testando com Gemini ---")
    resposta_gemini = executar_agente_ia(pergunta, "gemini")
    print(resposta_gemini)

    print("\n--- Testando com DeepSeek ---")
    resposta_deepseek = executar_agente_ia(pergunta, "deepseek")
    print(resposta_deepseek)
import json
import os
import sys
from typing import Any, Dict, Union

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import LLMFactory


NOME_AGENTE_CLASSIFICADOR = "agente_classificador"


def carregar_prompt_classificador(caminho_arquivo: str, nome_agente: str) -> Dict[str, str]:
    with open(caminho_arquivo, "r", encoding="utf-8") as file:
        schemas = yaml.safe_load(file)

    if not schemas or nome_agente not in schemas:
        raise ValueError(f"Prompt do agente não encontrado: {nome_agente}")

    prompt = schemas[nome_agente]

    if "system" not in prompt or "user" not in prompt:
        raise ValueError(
            f"Prompt do agente {nome_agente} precisa conter as chaves 'system' e 'user'."
        )

    return {
        "system": prompt["system"],
        "user": prompt["user"],
    }


def extrair_json_resposta(resposta: str) -> Dict[str, Any]:
    inicio_json = resposta.find("{")
    fim_json = resposta.rfind("}")

    if inicio_json == -1 or fim_json == -1 or fim_json < inicio_json: # valida se encontrou a posição de abertura e fechamento do json
        return {
            "categoria": "outros",
            "confianca": 0.0,
            "justificativa": "A resposta da IA não veio em JSON válido.",
            "resposta_original": resposta,
        }

    conteudo_json = resposta[inicio_json : fim_json + 1]

    try:
        return json.loads(conteudo_json)
    except json.JSONDecodeError:
        return {
            "categoria": "outros",
            "confianca": 0.0,
            "justificativa": "Não foi possível interpretar o JSON retornado pela IA.",
            "resposta_original": resposta,
        }


def executar_classificador_ticket(
    ticket: str,
    provedor_ia: str = "openai",
    retornar_json: bool = True,
) -> Union[Dict[str, Any], str]: # Union informa que pode receber 2 tipos de dados
    if not ticket or not ticket.strip():
        raise ValueError("O texto do ticket não pode ser vazio.")

    caminho_prompt = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../Schemas/prompt_agente.yaml")
    )

    prompt = carregar_prompt_classificador(caminho_prompt, NOME_AGENTE_CLASSIFICADOR)
    prompt_usuario = prompt["user"].format(ticket=ticket.strip())

    modelo_ia = LLMFactory.criar_modelo(provedor_ia)
    resposta = modelo_ia.gerar_resposta(
        prompt_sistema=prompt["system"],
        prompt_usuario=prompt_usuario,
    )

    if not retornar_json:
        return resposta

    return extrair_json_resposta(resposta)


if __name__ == "__main__":
    ticket_teste = "Quero cancelar minha assinatura porque fui cobrado duas vezes."
    classificacao = executar_classificador_ticket(ticket_teste, provedor_ia="openai")
    print(classificacao)

# frontend/llm_responder.py

from typing import Dict, Any
from google import genai
import json

# O client lê a GEMINI_API_KEY do ambiente
client = genai.Client()

MODEL_ID = "gemini-2.5-flash"


def build_llm_answer(
    question: str,
    nlu_result: Dict[str, Any],
    dialog_result: Dict[str, Any],
) -> str:
    """
    Gera uma resposta amigável em português usando o Gemini 2.5 Flash
    a partir da pergunta do usuário, da saída de NLU e dos dados do BigQuery.
    """

    intent = nlu_result.get("intent", "")
    entities = nlu_result.get("entities", []) or []

    # Tenta pegar resultados estruturados do dialog_manager
    raw_results = (
        dialog_result.get("results")
        or dialog_result.get("data")
        or dialog_result.get("raw_results")
    )

    # Limitamos um pouco pra não mandar coisa infinita
    if isinstance(raw_results, list):
        preview_results = raw_results[:5]
    else:
        preview_results = raw_results

    # Monta um prompt rico, mas direto
    prompt_obj = {
        "pergunta_usuario": question,
        "intencao_nlu": intent,
        "entidades_nlu": entities,
        "resposta_base_dialog": dialog_result.get("answer"),
        "resultados_bigquery_preview": preview_results,
    }

    prompt = (
        "Você é um assistente que responde SEMPRE em português do Brasil.\n"
        "Você recebeu:\n\n"
        f"{json.dumps(prompt_obj, ensure_ascii=False, indent=2)}\n\n"
        "Tarefa:\n"
        "- Gere uma resposta amigável e natural para o usuário.\n"
        "- A resposta deve ser em texto corrido, sem JSON.\n"
        "- Use os dados de 'resultados_bigquery_preview' para citar alguns exemplos relevantes.\n"
        "- Se houver muitos resultados, cite só alguns (pelo menos 10) mais bem avaliados.\n"
        "- Seja objetivo, mas informativo.\n"
        "- Sempre utilize um formato de lista em caso de multiplos dados na resposta.\n"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            # fallback: usa a resposta base do dialog_manager
            return dialog_result.get("answer") or "Não consegui gerar uma resposta no momento."
        return text
    except Exception as e:
        # Fallback em caso de erro na chamada da LLM
        print(f"[LLM] Erro ao chamar Gemini: {e}")
        return dialog_result.get("answer") or "Não consegui gerar uma resposta no momento."

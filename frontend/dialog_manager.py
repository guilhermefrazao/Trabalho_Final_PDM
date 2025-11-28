# frontend/dialog_manager.py

from typing import Any, Dict, List, Optional

from bigquery_queries import (
    get_filmes_por_ano,
    get_duracao_filme,
    get_genero_filme,
    get_nota_filme,
    get_votos_filme,
    get_media_genero,
    get_votos_genero,
    get_filmes_por_genero,
    get_filmes_por_duracao,
    get_filmes_por_nota,
    get_filmes_por_votos,
    get_atores_filme,
    get_filmes_ator,
)


# ---------------------------
# Helpers de entidades
# ---------------------------

def extrair_entidade(
    entities: List[Dict[str, Any]],
    nome_entidade: str,
) -> Optional[str]:
    """Extrai o valor de uma entidade específica da lista de entidades."""
    return next(
        (e.get("value") for e in entities if e.get("entity") == nome_entidade),
        None,
    )


def get_entity_value(entities: List[Dict[str, Any]], name: str) -> Optional[str]:
    return extrair_entidade(entities, name)


def get_int_entity(entities: List[Dict[str, Any]], name: str) -> Optional[int]:
    val = get_entity_value(entities, name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def get_float_entity(entities: List[Dict[str, Any]], name: str) -> Optional[float]:
    val = get_entity_value(entities, name)
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "."))
    except ValueError:
        return None


def get_operador_entity(entities: List[Dict[str, Any]]) -> str:
    """
    Converte o valor da entidade OPERADOR para um operador SQL válido.
    Ex.: 'maior_que' -> '>', 'menor_ou_igual' -> '<=', etc.
    Se não achar nada, assume '='.
    """
    val = get_entity_value(entities, "OPERADOR")
    if not val:
        return "="

    val = str(val).strip().lower()

    if val in {">", "<", "=", ">=", "<="}:
        return val

    mapa = {
        "maior": ">",
        "maior_que": ">",
        "maior que": ">",
        "menor": "<",
        "menor_que": "<",
        "menor que": "<",
        "maior_ou_igual": ">=",
        "maior ou igual": ">=",
        "menor_ou_igual": "<=",
        "menor ou igual": "<=",
        "igual": "=",
        "igual_a": "=",
        "igual a": "=",
    }

    return mapa.get(val, "=")


# ---------------------------
# Dialog Manager
# ---------------------------

def dialog_manager(
    intent_name: str,
    entities: List[Dict[str, Any]],
    query_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recebe o nome da intenção + entidades (formato FrazoPayload)
    e chama a função correta do BigQuery, montando uma resposta amigável.

    Formato esperado de entities:
      [{"entity": "ANO", "value": "2020"}, ...]
    """
    intent_name = (intent_name or "").strip()

    # ----------------------------------------------------------------------
    # INTENÇÃO 1: Filmes de um ano X
    # ----------------------------------------------------------------------
    if intent_name == "filmes_de_um_ano_x":
        ano = get_int_entity(entities, "ANO")
        if ano is None:
            return {"error": "Não encontrei o ano (entidade ANO) na sua pergunta."}

        filmes = get_filmes_por_ano(ano)
        if not filmes:
            return {"answer": f"Não encontrei filmes cadastrados para o ano {ano}."}

        top = filmes[:5]
        resumo = [
            f"{f['titulo_principal']} ({f['ano_lancamento']}) – nota {f['nota_media']}, votos {f['num_votos']}"
            for f in top
        ]

        print(top)

        return {
            "intent": intent_name,
            "answer": f"Encontrei {len(filmes)} filme(s) do ano {ano}. Alguns exemplos:",
            "examples": resumo,
            "results": filmes,
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 2: Duração de um filme X
    # ----------------------------------------------------------------------
    if intent_name == "duracao_de_um_filme_x":
        titulo = get_entity_value(entities, "FILME")
        if not titulo:
            return {"error": "Não encontrei o título do filme (entidade FILME)."}

        duracao = get_duracao_filme(titulo)
        if duracao is None:
            return {"answer": f"Não encontrei a duração de '{titulo}'."}

        return {
            "intent": intent_name,
            "answer": f"O filme '{titulo}' tem duração de {duracao} minutos.",
            "results": {"titulo": titulo, "duracao_minutos": duracao},
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 3: Gênero(s) de um filme X
    # ----------------------------------------------------------------------
    if intent_name == "genero_de_um_filme_x":
        titulo = get_entity_value(entities, "FILME")
        if not titulo:
            return {"error": "Não encontrei o título do filme (entidade FILME)."}

        generos = get_genero_filme(titulo)
        if not generos:
            return {"answer": f"Não encontrei os gêneros de '{titulo}'."}

        return {
            "intent": intent_name,
            "answer": f"O filme '{titulo}' é do(s) gênero(s): {generos}.",
            "results": {"titulo": titulo, "generos": generos},
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 4: Nota média de um filme X
    # ----------------------------------------------------------------------
    if intent_name == "nota_media_de_um_filme_x":
        titulo = get_entity_value(entities, "FILME")
        if not titulo:
            return {"error": "Não encontrei o título do filme (entidade FILME)."}

        nota = get_nota_filme(titulo)
        if nota is None:
            return {"answer": f"Não encontrei a nota de '{titulo}'."}

        return {
            "intent": intent_name,
            "answer": f"A nota média de '{titulo}' é {nota}.",
            "results": {"titulo": titulo, "nota_media": nota},
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 5: Número de votos de um filme X
    # ----------------------------------------------------------------------
    if intent_name == "numero_de_votos_de_um_filme_x":
        titulo = get_entity_value(entities, "FILME")
        if not titulo:
            return {"error": "Não encontrei o título do filme (entidade FILME)."}

        votos = get_votos_filme(titulo)
        if votos is None:
            return {"answer": f"Não encontrei o número de votos de '{titulo}'."}

        return {
            "intent": intent_name,
            "answer": f"O filme '{titulo}' tem {votos} voto(s).",
            "results": {"titulo": titulo, "num_votos": votos},
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 6: Nota média para filmes de gênero X
    # ----------------------------------------------------------------------
    if intent_name == "nota_media_para_filmes_de_genero_x":
        genero = get_entity_value(entities, "GENERO")
        if not genero:
            return {"error": "Não encontrei o gênero (entidade GENERO)."}

        media = get_media_genero(genero)
        if media is None:
            return {
                "answer": f"Não encontrei filmes do gênero '{genero}' para calcular a média."
            }

        return {
            "intent": intent_name,
            "answer": f"A nota média dos filmes de gênero '{genero}' é {media}.",
            "results": {"genero": genero, "nota_media": media},
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 7: Quantidade de votos para filmes de gênero X
    # ----------------------------------------------------------------------
    if intent_name == "quantidade_de_votos_para_filmes_de_genero_x":
        genero = get_entity_value(entities, "GENERO")
        if not genero:
            return {"error": "Não encontrei o gênero (entidade GENERO)."}

        votos = get_votos_genero(genero)
        if votos is None:
            return {"answer": f"Não encontrei votos para filmes do gênero '{genero}'."}

        return {
            "intent": intent_name,
            "answer": f"Filmes do gênero '{genero}' somam {votos} voto(s) no total.",
            "results": {"genero": genero, "total_votos": votos},
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 8: Filmes de um gênero X
    # ----------------------------------------------------------------------
    if intent_name == "filmes_de_um_genero_x":
        genero = get_entity_value(entities, "GENERO")
        if not genero:
            return {"error": "Não encontrei o gênero (entidade GENERO)."}

        filmes = get_filmes_por_genero(genero)
        if not filmes:
            return {"answer": f"Não encontrei filmes do gênero '{genero}'."}

        top = filmes[:5]
        resumo = [
            f"{f['titulo_principal']} ({f['ano_lancamento']}) – nota {f['nota_media']}"
            for f in top
        ]
        return {
            "intent": intent_name,
            "answer": f"Encontrei {len(filmes)} filme(s) do gênero '{genero}'. Alguns exemplos:",
            "examples": resumo,
            "results": filmes,
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 9: Filmes com duração maior/menor/igual a Y
    # ----------------------------------------------------------------------
    if intent_name == "filmes_com_duracao_maior_menor_ou_igual_a_y":
        duracao = get_int_entity(entities, "DURACAO")
        if duracao is None:
            return {"error": "Não encontrei a duração (entidade DURACAO)."}

        operador = get_operador_entity(entities)
        filmes = get_filmes_por_duracao(duracao, operador)
        if not filmes:
            return {
                "answer": f"Não encontrei filmes com duração {operador} {duracao} minutos."
            }

        top = filmes[:5]
        resumo = [
            f"{f['titulo_principal']} ({f['ano_lancamento']}) – {f['duracao_minutos']} min"
            for f in top
        ]
        return {
            "intent": intent_name,
            "answer": f"Encontrei {len(filmes)} filme(s) com duração {operador} {duracao} minutos. Alguns exemplos:",
            "examples": resumo,
            "results": filmes,
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 10: Filmes com nota maior/menor/igual a Y
    # ----------------------------------------------------------------------
    if intent_name == "filmes_com_nota_maior_menor_ou_igual_a_y":
        nota = get_float_entity(entities, "NOTA")
        if nota is None:
            return {"error": "Não encontrei a nota (entidade NOTA)."}

        operador = get_operador_entity(entities)
        filmes = get_filmes_por_nota(nota, operador)
        if not filmes:
            return {
                "answer": f"Não encontrei filmes com nota {operador} {nota}."
            }

        top = filmes[:5]
        resumo = [
            f"{f['titulo_principal']} ({f['ano_lancamento']}) – nota {f['nota_media']}"
            for f in top
        ]
        return {
            "intent": intent_name,
            "answer": f"Encontrei {len(filmes)} filme(s) com nota {operador} {nota}. Alguns exemplos:",
            "examples": resumo,
            "results": filmes,
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 11: Filmes com número de votos maior/menor/igual a Y
    # ----------------------------------------------------------------------
    if intent_name == "filmes_com_numero_de_votos_maior_menor_ou_igual_a_y":
        votos = get_int_entity(entities, "VOTOS")
        if votos is None:
            return {"error": "Não encontrei o número de votos (entidade VOTOS)."}

        operador = get_operador_entity(entities)
        filmes = get_filmes_por_votos(votos, operador)
        if not filmes:
            return {
                "answer": f"Não encontrei filmes com número de votos {operador} {votos}."
            }

        top = filmes[:5]
        resumo = [
            f"{f['titulo_principal']} ({f['ano_lancamento']}) – votos {f['num_votos']}"
            for f in top
        ]
        return {
            "intent": intent_name,
            "answer": f"Encontrei {len(filmes)} filme(s) com número de votos {operador} {votos}. Alguns exemplos:",
            "examples": resumo,
            "results": filmes,
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 12: Atores de um filme X
    # ----------------------------------------------------------------------
    if intent_name == "atores_de_um_filme_x":
        titulo = get_entity_value(entities, "FILME")
        if not titulo:
            return {"error": "Não encontrei o título do filme (entidade FILME)."}

        atores = get_atores_filme(titulo)
        if not atores:
            return {"answer": f"Não encontrei atores cadastrados para o filme '{titulo}'."}

        return {
            "intent": intent_name,
            "answer": f"Encontrei {len(atores)} ator(es) para o filme '{titulo}'.",
            "actors": atores,
        }

    # ----------------------------------------------------------------------
    # INTENÇÃO 13: Filmes que um ator X participou
    # ----------------------------------------------------------------------
    if intent_name == "filmes_que_um_ator_x_participou":
        ator = get_entity_value(entities, "ATOR")
        if not ator:
            return {"error": "Não encontrei o nome do ator (entidade ATOR)."}

        filmes = get_filmes_ator(ator)
        if not filmes:
            return {"answer": f"Não encontrei filmes para o ator '{ator}'."}

        top = filmes[:5]
        resumo = [
            f"{f['titulo_principal']} ({f['ano_lancamento']}) – nota {f['nota_media']}"
            for f in top
        ]
        return {
            "intent": intent_name,
            "answer": f"Encontrei {len(filmes)} filme(s) com o ator '{ator}'. Alguns exemplos:",
            "examples": resumo,
            "results": filmes,
        }

    # ----------------------------------------------------------------------
    # Intenção desconhecida
    # ----------------------------------------------------------------------
    return {
        "intent": intent_name,
        "answer": "Desculpe, ainda não sei como responder esse tipo de pergunta.",
    }

from google.cloud import bigquery
from typing import Optional, List, Dict, Any

# Tabelas do BigQuery
TABLE_FILMES = "imdb_prata.filmes_com_notas" 
TABLE_PESSOAS = "imdb_prata.pessoas_do_filme"

# Cliente BigQuery (injetado pelo app.py)
client: Optional[bigquery.Client] = None 

def set_bigquery_client(bq_client: bigquery.Client):
    """Define o cliente BigQuery que será usado pelas funções de consulta."""
    global client
    client = bq_client
    print("Cliente BigQuery configurado em bigquery_queries.py")


# ============================================================================
# INTENÇÃO 1: Filmes de um ano X
# ============================================================================
def get_filmes_por_ano(ano: int) -> Optional[List[Dict[str, Any]]]:
    """
    Retorna todos os filmes lançados em um ano específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT id_filme, titulo_principal, ano_lancamento, duracao_minutos, 
               generos, nota_media, num_votos
        FROM {full_table_id}
        WHERE ano_lancamento = @ano
        ORDER BY nota_media DESC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ano", "INT64", ano)
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result:
            return [dict(row) for row in result]
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 2: Duração de um filme X
# ============================================================================
def get_duracao_filme(titulo: str) -> Optional[int]:
    """
    Retorna a duração em minutos de um filme específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT duracao_minutos
        FROM {full_table_id}
        WHERE LOWER(titulo_principal) = LOWER(@titulo)
        LIMIT 1
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("titulo", "STRING", titulo.strip())
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result and result[0]['duracao_minutos'] is not None:
            return int(result[0]['duracao_minutos'])
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 3: Gênero(s) de um filme X
# ============================================================================
def get_genero_filme(titulo: str) -> Optional[str]:
    """
    Retorna os gêneros de um filme específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT generos
        FROM {full_table_id}
        WHERE LOWER(titulo_principal) = LOWER(@titulo)
        LIMIT 1
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("titulo", "STRING", titulo.strip())
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result and result[0]['generos'] is not None:
            return result[0]['generos']
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 4: Nota média de um filme X
# ============================================================================
def get_nota_filme(titulo: str) -> Optional[float]:
    """
    Retorna a nota média de um filme específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT nota_media
        FROM {full_table_id}
        WHERE LOWER(titulo_principal) = LOWER(@titulo)
        LIMIT 1
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("titulo", "STRING", titulo.strip())
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result and result[0]['nota_media'] is not None:
            return round(float(result[0]['nota_media']), 2)
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 5: Número de votos de um filme X
# ============================================================================
def get_votos_filme(titulo: str) -> Optional[int]:
    """
    Retorna o número de votos de um filme específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT num_votos
        FROM {full_table_id}
        WHERE LOWER(titulo_principal) = LOWER(@titulo)
        LIMIT 1
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("titulo", "STRING", titulo.strip())
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result and result[0]['num_votos'] is not None:
            return int(result[0]['num_votos'])
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 6: Nota média para filmes de gênero X
# ============================================================================
def get_media_genero(genero: str) -> Optional[float]:
    """
    Calcula a nota média para filmes de um gênero específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT AVG(nota_media) as media_rating
        FROM {full_table_id}
        WHERE generos LIKE @genero_param
    """
    
    genero_search = f"%{genero.strip()}%"
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("genero_param", "STRING", genero_search)
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result and result[0]['media_rating'] is not None:
            return round(float(result[0]['media_rating']), 2)
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 7: Quantidade de votos para filmes de gênero X
# ============================================================================
def get_votos_genero(genero: str) -> Optional[int]:
    """
    Calcula a quantidade total de votos para filmes de um gênero específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
        
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT SUM(num_votos) as total_votos
        FROM {full_table_id}
        WHERE generos LIKE @genero_param
    """
    
    genero_search = f"%{genero.strip()}%"
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("genero_param", "STRING", genero_search)
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result and result[0]['total_votos'] is not None:
            return int(result[0]['total_votos'])
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 8: Filmes de um gênero X
# ============================================================================
def get_filmes_por_genero(genero: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retorna todos os filmes de um gênero específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT id_filme, titulo_principal, ano_lancamento, duracao_minutos, 
               generos, nota_media, num_votos
        FROM {full_table_id}
        WHERE generos LIKE @genero_param
        ORDER BY nota_media DESC
    """
    
    genero_search = f"%{genero.strip()}%"
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("genero_param", "STRING", genero_search)
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result:
            return [dict(row) for row in result]
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 9: Filmes com duração maior, menor ou igual a Y
# ============================================================================
def get_filmes_por_duracao(duracao: int, operador: str = "=") -> Optional[List[Dict[str, Any]]]:
    """
    Retorna filmes com duração conforme o operador especificado.
    
    Args:
        duracao: Duração em minutos
        operador: ">" (maior), "<" (menor), "=" (igual), ">=" (maior ou igual), "<=" (menor ou igual)
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    # Validação do operador
    operadores_validos = [">", "<", "=", ">=", "<="]
    if operador not in operadores_validos:
        print(f"Operador inválido: {operador}. Use um dos seguintes: {operadores_validos}")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT id_filme, titulo_principal, ano_lancamento, duracao_minutos, 
               generos, nota_media, num_votos
        FROM {full_table_id}
        WHERE duracao_minutos {operador} @duracao
        ORDER BY duracao_minutos DESC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("duracao", "INT64", duracao)
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result:
            return [dict(row) for row in result]
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 10: Filmes com nota maior, menor ou igual a Y
# ============================================================================
def get_filmes_por_nota(nota: float, operador: str = "=") -> Optional[List[Dict[str, Any]]]:
    """
    Retorna filmes com nota média conforme o operador especificado.
    
    Args:
        nota: Nota média (0.0 a 10.0)
        operador: ">" (maior), "<" (menor), "=" (igual), ">=" (maior ou igual), "<=" (menor ou igual)
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    # Validação do operador
    operadores_validos = [">", "<", "=", ">=", "<="]
    if operador not in operadores_validos:
        print(f"Operador inválido: {operador}. Use um dos seguintes: {operadores_validos}")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT id_filme, titulo_principal, ano_lancamento, duracao_minutos, 
               generos, nota_media, num_votos
        FROM {full_table_id}
        WHERE nota_media {operador} @nota
        ORDER BY nota_media DESC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("nota", "FLOAT64", nota)
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result:
            return [dict(row) for row in result]
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 11: Filmes com número de votos maior, menor ou igual a Y
# ============================================================================
def get_filmes_por_votos(votos: int, operador: str = "=") -> Optional[List[Dict[str, Any]]]:
    """
    Retorna filmes com número de votos conforme o operador especificado.
    
    Args:
        votos: Número de votos
        operador: ">" (maior), "<" (menor), "=" (igual), ">=" (maior ou igual), "<=" (menor ou igual)
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    # Validação do operador
    operadores_validos = [">", "<", "=", ">=", "<="]
    if operador not in operadores_validos:
        print(f"Operador inválido: {operador}. Use um dos seguintes: {operadores_validos}")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT id_filme, titulo_principal, ano_lancamento, duracao_minutos, 
               generos, nota_media, num_votos
        FROM {full_table_id}
        WHERE num_votos {operador} @votos
        ORDER BY num_votos DESC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("votos", "INT64", votos)
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result:
            return [dict(row) for row in result]
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 12: Atores de um filme X
# ============================================================================
def get_atores_filme(titulo: str) -> Optional[List[str]]:
    """
    Retorna a lista de atores de um filme específico.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    full_table_id = f"`{client.project}.{TABLE_PESSOAS}`"
        
    query = f"""
        SELECT DISTINCT nome_pessoa
        FROM {full_table_id}
        WHERE LOWER(titulo_principal) = LOWER(@titulo)
          AND LOWER(categoria) = 'actor'
        ORDER BY nome_pessoa
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("titulo", "STRING", titulo.strip())
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result:
            return [row['nome_pessoa'] for row in result]
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None


# ============================================================================
# INTENÇÃO 13: Filmes que um ator X participou
# ============================================================================
def get_filmes_ator(nome_ator: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retorna todos os filmes em que um ator específico participou.
    """
    if client is None:
        print("Erro: Cliente BigQuery não está disponível.")
        return None
    
    # Join entre as duas tabelas
    table_pessoas = f"`{client.project}.{TABLE_PESSOAS}`"
    table_filmes = f"`{client.project}.{TABLE_FILMES}`"
        
    query = f"""
        SELECT DISTINCT 
            f.id_filme, 
            f.titulo_principal, 
            f.ano_lancamento, 
            f.duracao_minutos, 
            f.generos, 
            f.nota_media, 
            f.num_votos
        FROM {table_pessoas} p
        JOIN {table_filmes} f ON p.id_filme = f.id_filme
        WHERE LOWER(p.nome_pessoa) = LOWER(@nome_ator)
          AND LOWER(p.categoria) = 'actor'
        ORDER BY f.ano_lancamento DESC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("nome_ator", "STRING", nome_ator.strip())
        ]
    )
    
    try:
        query_job = client.query(query, job_config=job_config)
        result = list(query_job.result())
        
        if result:
            return [dict(row) for row in result]
        else:
            return None
            
    except Exception as e:
        print(f"Erro ao executar BigQuery: {e}")
        return None
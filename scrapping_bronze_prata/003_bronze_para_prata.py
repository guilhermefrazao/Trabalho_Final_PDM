import os
import glob
import json
import pandas as pd
from sqlalchemy import create_engine
import re

# --- Configurações (mesmas de antes) ---
BRONZE_DIR = "bronze_data"
DB_USER = "user_ia"
DB_PASSWORD = "password_ia"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "filmes_db"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Funções de Limpeza (NOVAS E MELHORADAS) ---

def clean_money(value):
    if value is None: return None
    value = str(value).lower()
    value = re.sub(r'\[.*?\]|\(.*?\)|\,', '', value)
    if 'billion' in value:
        value = re.sub(r'[\$\s]|billion', '', value)
        try: return float(value) * 1_000_000_000
        except ValueError: return None
    if 'million' in value:
        value = re.sub(r'[\$\s]|million', '', value)
        try: return float(value) * 1_000_000
        except ValueError: return None
    value = re.sub(r'[\$\s]', '', value)
    try: return float(value)
    except ValueError: return None

def clean_date(value):
    if value is None: return None
    match = re.search(r'(\d{1,2}\s\w+\s\d{4})|(\w+\s\d{1,2},\s\d{4})', str(value))
    if match:
        try: return pd.to_datetime(match.group(0)).strftime('%Y-%m-%d')
        except: return None
    return None

def clean_list(value, separator='•'):
    """Separa uma string (ex: 'Ator 1 • Ator 2') em uma lista limpa."""
    if value is None: return []
    # Remove referências [1], [2], etc.
    value = re.sub(r'\[.*?\]', '', str(value))
    return [item.strip() for item in str(value).split(separator) if item.strip()]

def clean_runtime(value):
    """Extrai o número de minutos de '114 minutes'."""
    if value is None: return None
    match = re.search(r'(\d+)\s+minutes', str(value))
    if match:
        try: return int(match.group(1))
        except: return None
    return None

def coalesce_keys(data_dict, keys_list):
    """Encontra o primeiro valor não-nulo para uma lista de chaves.
       Ex: keys_list = ['countries', 'country']
    """
    for key in keys_list:
        if key in data_dict and data_dict[key] is not None:
            return data_dict[key]
    return None

# --- Script Principal (ETL V2 Aprimorado) ---
def main():
    print("--- Iniciando Pipeline ETL V2 (Prata Aprimorado) ---")

    # 1. Conectar ao Banco de Dados
    try:
        engine = create_engine(DATABASE_URL)
        print("Conexão com o PostgreSQL (Camada Prata) estabelecida.")
    except Exception as e:
        print(f"ERRO: Conexão com o banco falhou: {e}")
        return

    # 2. EXTRAIR (Extract): Ler JSONs da Camada Bronze
    json_files = glob.glob(os.path.join(BRONZE_DIR, "*.json"))
    if not json_files:
        print(f"ERRO: Nenhum .json encontrado em '{BRONZE_DIR}'")
        return
        
    print(f"Encontrados {len(json_files)} arquivos JSON.")
    all_movies_data = []
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_movies_data.append(json.load(f))

    # 3. TRANSFORMAR (Transform) V2
    print("Transformando dados...")
    
    # Criar DataFrame principal
    df = pd.DataFrame(all_movies_data)

    # --- 3.1 Limpar Colunas Principais (Orçamento, Data, Duração) ---
    
    # Colunas de limpeza básica
    df['budget_clean'] = df['budget'].apply(clean_money) if 'budget' in df.columns else None
    df['box_office_clean'] = df['box_office'].apply(clean_money) if 'box_office' in df.columns else None
    df['release_date_clean'] = df.apply(lambda row: coalesce_keys(row, ['release_dates', 'release_date']), axis=1).apply(clean_date)
    df['running_time_min'] = df['running_time'].apply(clean_runtime) if 'running_time' in df.columns else None
    df['language_clean'] = df['language'].apply(lambda x: clean_list(x)[0] if clean_list(x) else None) if 'language' in df.columns else None

    # Colunas que são listas (vamos limpar e guardar como texto por enquanto)
    df['countries_list'] = df.apply(lambda row: coalesce_keys(row, ['countries', 'country']), axis=1).apply(clean_list).apply(lambda x: ', '.join(x))
    df['production_companies_list'] = df.apply(lambda row: coalesce_keys(row, ['productioncompanies', 'production_companies']), axis=1).apply(clean_list).apply(lambda x: ', '.join(x))
    
    # Colunas que vamos normalizar (roteiristas, compositores)
    df['writers_raw'] = df.apply(lambda row: coalesce_keys(row, ['written_by', 'screenplay_by', 'story_by']), axis=1)
    df['composers_raw'] = df.apply(lambda row: coalesce_keys(row, ['music_by']), axis=1)

    # Garantir que colunas de junção existam
    if 'title' not in df.columns: df['title'] = None
    if 'directed_by' not in df.columns: df['directed_by'] = None
    if 'starring' not in df.columns: df['starring'] = None
    if 'genre' not in df.columns: df['genre'] = None
    
    # Adicionar um ID único para cada filme
    df.dropna(subset=['title'], inplace=True) # Filmes sem título são inúteis
    df.reset_index(drop=True, inplace=True)
    df['movie_id'] = df.index

    # --- 3.2 Criar Tabelas de Dimensão (People, Genres) ---
    
    # Tabela 'people' (AGORA INCLUI ROTEIRISTAS E COMPOSITORES)
    directors = df['directed_by'].dropna().unique()
    actors_lists = df['starring'].apply(clean_list).explode()
    writers_lists = df['writers_raw'].apply(clean_list).explode()
    composers_lists = df['composers_raw'].apply(clean_list).explode()
    
    # Combinar todos os nomes em um set (para garantir nomes únicos)
    all_people_names = set(directors) | \
                       set(actors_lists.dropna().unique()) | \
                       set(writers_lists.dropna().unique()) | \
                       set(composers_lists.dropna().unique())
                       
    df_people = pd.DataFrame(all_people_names, columns=['name'])
    df_people.dropna(inplace=True)
    df_people.reset_index(drop=True, inplace=True)
    df_people['person_id'] = df_people.index
    
    # Tabela 'genres'
    genres_lists = df['genre'].apply(clean_list, separator='•').explode()
    all_genres = genres_lists.dropna().unique()
    df_genres = pd.DataFrame(all_genres, columns=['genre_name'])
    df_genres.dropna(inplace=True)
    df_genres.reset_index(drop=True, inplace=True)
    df_genres['genre_id'] = df_genres.index

    print(f"Dimensões criadas: {len(df_people)} pessoas, {len(df_genres)} gêneros.")

    # --- 3.3 Criar Tabela Principal 'movies' (com Chaves Estrangeiras e Novos Campos) ---
    
    # Juntar (Merge) com df_people para obter o 'director_id'
    df_movies_main = df.merge(df_people, left_on='directed_by', right_on='name', how='left')
    
    # Selecionar e renomear colunas para a tabela 'movies'
    df_movies_main = df_movies_main[[
        'movie_id', 'title', 'release_date_clean', 
        'budget_clean', 'box_office_clean', 'person_id',
        'running_time_min', 'language_clean', 'countries_list', 'production_companies_list'
    ]].rename(columns={
        'person_id': 'director_id',
        'language_clean': 'language',
    })

    # --- 3.4 Criar Tabelas de Junção (Antigas e NOVAS) ---

    # Tabela 'movie_actors'
    df_actor_links = df[['movie_id', 'starring']].copy()
    df_actor_links['actor_name'] = df_actor_links['starring'].apply(clean_list)
    df_actor_links = df_actor_links.explode('actor_name').dropna()
    df_movie_actors = df_actor_links.merge(df_people, left_on='actor_name', right_on='name', how='left')
    df_movie_actors = df_movie_actors[['movie_id', 'person_id']].dropna().drop_duplicates().astype(int)

    # Tabela 'movie_genres'
    df_genre_links = df[['movie_id', 'genre']].copy()
    df_genre_links['genre_name'] = df_genre_links['genre'].apply(clean_list, separator='•')
    df_genre_links = df_genre_links.explode('genre_name').dropna()
    df_movie_genres = df_genre_links.merge(df_genres, on='genre_name', how='left')
    df_movie_genres = df_movie_genres[['movie_id', 'genre_id']].dropna().drop_duplicates().astype(int)

    # NOVA Tabela 'movie_writers'
    df_writer_links = df[['movie_id', 'writers_raw']].copy()
    df_writer_links['writer_name'] = df_writer_links['writers_raw'].apply(clean_list)
    df_writer_links = df_writer_links.explode('writer_name').dropna()
    df_movie_writers = df_writer_links.merge(df_people, left_on='writer_name', right_on='name', how='left')
    df_movie_writers = df_movie_writers[['movie_id', 'person_id']].dropna().drop_duplicates().astype(int)

    # NOVA Tabela 'movie_composers'
    df_composer_links = df[['movie_id', 'composers_raw']].copy()
    df_composer_links['composer_name'] = df_composer_links['composers_raw'].apply(clean_list)
    df_composer_links = df_composer_links.explode('composer_name').dropna()
    df_movie_composers = df_composer_links.merge(df_people, left_on='composer_name', right_on='name', how='left')
    df_movie_composers = df_movie_composers[['movie_id', 'person_id']].dropna().drop_duplicates().astype(int)

    print("Tabelas de junção criadas.")

    # 4. CARREGAR (Load) V2 Aprimorado
    print("Carregando tabelas na Camada Prata (PostgreSQL)...")
    try:
        # A ordem importa por causa das Chaves Estrangeiras!
        # 1. Tabelas de Dimensão (não dependem de ninguém)
        df_people.to_sql('people', engine, if_exists='replace', index=False)
        df_genres.to_sql('genres', engine, if_exists='replace', index=False)
        
        # 2. Tabela Principal (depende de 'people' para director_id)
        df_movies_main.to_sql('movies', engine, if_exists='replace', index=False)
        
        # 3. Tabelas de Junção (dependem de 'movies' e 'people'/'genres')
        df_movie_actors.to_sql('movie_actors', engine, if_exists='replace', index=False)
        df_movie_genres.to_sql('movie_genres', engine, if_exists='replace', index=False)
        df_movie_writers.to_sql('movie_writers', engine, if_exists='replace', index=False)
        df_movie_composers.to_sql('movie_composers', engine, if_exists='replace', index=False)

        print("\n--- Pipeline ETL V2 (Aprimorado) Concluído com Sucesso! ---")
        print("Seu banco de dados Prata está normalizado e pronto para a IA.")
        print("Novas tabelas carregadas: 'movie_writers', 'movie_composers'")
        print("Tabela 'movies' enriquecida com: 'running_time_min', 'language', 'countries_list', 'production_companies_list'")

    except Exception as e:
        print(f"ERRO ao carregar dados no PostgreSQL: {e}")

if __name__ == "__main__":
    main()
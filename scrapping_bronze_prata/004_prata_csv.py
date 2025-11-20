import os
import pandas as pd
from sqlalchemy import create_engine

DB_USER = "user_ia"
DB_PASSWORD = "password_ia"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "filmes_db"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

OUTPUT_DIR = "silver_exports"

def main():
    print("--- Iniciando Exportação Seletiva (Prata -> CSV) ---")

    # 1. Criar a pasta de saída se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Diretório criado: {OUTPUT_DIR}")

    # 2. Conectar ao Banco de Dados
    try:
        engine = create_engine(DATABASE_URL)
        engine.connect() # Testa a conexão
    except Exception as e:
        print(f"ERRO: Conexão com o banco falhou: {e}")
        print("Verifique se o seu container Docker está rodando! ('docker-compose up -d')")
        return

    # 3. Definir as tabelas específicas para exportar
    TABELAS_PARA_EXPORTAR = [
        "movies",
        "people",
        "movie_actors",
        "movie_composers",
        "movie_writers"
    ]
        
    print(f"Tabelas selecionadas para exportar: {TABELAS_PARA_EXPORTAR}")

    # 4. Loop para ler cada tabela da lista e salvar como CSV
    for table_name in TABELAS_PARA_EXPORTAR:
        print(f"  Exportando '{table_name}'...")
        try:
            # Lê a tabela inteira do SQL e joga para um DataFrame do Pandas
            df = pd.read_sql_table(table_name, engine)
            
            # Define o caminho do arquivo de saída
            output_path = os.path.join(OUTPUT_DIR, f"{table_name}.csv")
            
            # Salva o DataFrame como CSV
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            # Este 'except' é útil caso você rode o script antes do ETL
            # ou se houver um erro de digitação no nome da tabela.
            print(f"    -> ERRO ao exportar {table_name}: {e}")
            print(f"    -> Verifique se a tabela '{table_name}' existe no banco de dados.")

    print("\n--- Exportação Seletiva Concluída com Sucesso! ---")
    print(f"Os {len(TABELAS_PARA_EXPORTAR)} arquivos CSV estão salvos na pasta: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
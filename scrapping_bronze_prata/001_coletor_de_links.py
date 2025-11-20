import requests
import time
from bs4 import BeautifulSoup
import os

# --- Configurações ---

headers = {
    'User-Agent': 'trab-final-pdm/1.0 (mariaalmeida2@discente.ufg.br)'
}

# Lista de páginas de listas de filmes

SEED_URLS = [
    'List_of_American_films_of_2025',
    'List_of_American_films_of_2024',
    'List_of_American_films_of_2023',
    'List_of_American_films_of_2022',
    'List_of_American_films_of_2021',
    'List_of_American_films_of_2020',
    'List_of_American_films_of_2019',
    'List_of_American_films_of_2018',
    'List_of_American_films_of_2017',
    'List_of_American_films_of_2016',
]
BASE_URL = "https://en.wikipedia.org"

# Arquivo de saída onde os links serão salvos
OUTPUT_FILE = "links_para_visitar.txt"

# Usamos um set() para armazenar os links e evitar duplicatas automaticamente
links_encontrados = set()

def main():
    print("--- Iniciando Scraper Fase 1: Coletor de Links ---")
    
    for seed_page in SEED_URLS:
        url = f"{BASE_URL}/wiki/{seed_page}"
        print(f"\n[Processando Semente]: {url}")
        
        try:
            # 1. Fazer a requisição HTTP
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Lança um erro se a requisição falhar (ex: 404, 500)

            # 2. Parsear o HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 3. Encontrar as tabelas de filmes (elas têm a classe 'wikitable')
            tables = soup.find_all('table', class_='wikitable')
            
            if not tables:
                print("  -> Nenhuma tabela 'wikitable' encontrada.")
                continue

            # 4. Iterar sobre cada tabela encontrada
            for table in tables:
                # Iterar sobre as linhas da tabela, pulando a primeira (cabeçalho)
                for row in table.find_all('tr')[1:]:
                    # Pegar a primeira célula (onde o título do filme está)
                    cell = row.find('td')
                    if not cell:
                        continue
                    
                    # Encontrar a primeira tag <a> dentro da célula
                    link_tag = cell.find('a')
                    
                    # 5. Extrair e filtrar o link
                    if link_tag and link_tag.has_attr('href'):
                        href = link_tag['href']
                        
                        # Queremos apenas links internos para artigos,
                        # ignorando links de categorias, arquivos ou edição.
                        if href.startswith('/wiki/') and ':' not in href:
                            links_encontrados.add(href)
                            
            print(f"  -> Links encontrados até agora: {len(links_encontrados)}")
            
            # 6. Pausa de Rate Limiting entre as requisições.
            print("  -> Pausa de 1 segundo...")
            time.sleep(1)

        except requests.RequestException as e:
            print(f"  -> ERRO ao processar {url}: {e}")

    # 7. Salvar os resultados
    print(f"\n--- Coleta Finalizada ---")
    print(f"Total de links únicos encontrados: {len(links_encontrados)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for link in sorted(links_encontrados):  # Salva em ordem alfabética
            f.write(link + '\n')
            
    print(f"Links salvos com sucesso em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
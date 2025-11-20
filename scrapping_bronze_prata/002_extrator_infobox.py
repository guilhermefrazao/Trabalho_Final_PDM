import requests
import time
from bs4 import BeautifulSoup
import os
import json

# --- Configurações ---

# 1. User-Agent (O MESMO DO SCRIPT 001)
headers = {
    'User-Agent': 'trab-final-pdm/1.0 (mariaalmeida2@discente.ufg.br)'
}

# 2. Arquivos e Diretórios de Entrada/Saída
INPUT_FILE = "links_para_visitar.txt"  # Gerado pelo script 001
OUTPUT_DIR = "bronze_data"            # Onde os JSONs brutos serão salvos

# 3. Base URL
BASE_URL = "https://en.wikipedia.org"

# --- Funções Auxiliares ---

def clean_key(key_text):
    """Limpa o texto da 'chave' (ex: <th>) da infobox."""
    return key_text.strip().lower().replace(' ', '_')

def extract_infobox_data(soup):
    """Encontra a infobox e extrai os pares de chave-valor."""
    infobox = soup.find('table', class_='infobox')
    if not infobox:
        return None  # Não encontrou infobox (pode ser um link que não é de filme)

    data = {}
    rows = infobox.find_all('tr')

    # Adiciona o título do filme (geralmente no <caption> ou <th> da primeira linha)
    title_tag = infobox.find('caption') or rows[0].find('th')
    if title_tag:
        data['title'] = title_tag.get_text(strip=True)

    for row in rows:
        # Encontrar a chave (no <th>)
        key_tag = row.find('th')
        # Encontrar o valor (no <td>)
        value_tag = row.find('td')

        if key_tag and value_tag:
            # Limpa a chave para ser um bom nome de coluna (ex: "Directed by" -> "directed_by")
            key = clean_key(key_tag.get_text(strip=True))
            
            # Extração Bruta (Camada Bronze):
            # Apenas pegamos o texto. A limpeza complexa (ex: listas, valores)
            # será feita para a camada Prata.
            # Usamos ' • ' como separador para itens de lista (comuns em 'Starring')
            value = value_tag.get_text(separator=' • ', strip=True)
            
            data[key] = value
            
    return data

def main():
    print("--- Iniciando Scraper Fase 2: Extrator de Infobox ---")

    # 1. Criar o diretório de saída se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Diretório criado: {OUTPUT_DIR}")

    # 2. Ler os links do arquivo de entrada
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            links = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERRO: Arquivo de entrada '{INPUT_FILE}' não encontrado.")
        print("Certifique-se de executar o scraper_fase_1.py primeiro.")
        return
    
    total_links = len(links)
    print(f"Total de {total_links} links para processar.")

    # 3. Iterar sobre cada link
    for i, link_href in enumerate(links):
        url = f"{BASE_URL}{link_href}"
        
        # Define um nome de arquivo seguro (ex: /wiki/Oppenheimer_(film) -> Oppenheimer_(film).json)
        file_name = link_href.split('/')[-1] + ".json"
        output_path = os.path.join(OUTPUT_DIR, file_name)

        # 4. Verificar se o arquivo já foi baixado (para ser "resumável")
        if os.path.exists(output_path):
            print(f"[{i+1}/{total_links}] JÁ EXISTE: {file_name}")
            continue
        
        print(f"[{i+1}/{total_links}] Processando: {url}")
        
        try:
            # 5. Fazer a requisição HTTP
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # 6. Parsear o HTML e extrair os dados
            soup = BeautifulSoup(response.text, 'html.parser')
            movie_data = extract_infobox_data(soup)

            # 7. Salvar os dados brutos (Camada Bronze)
            if movie_data:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(movie_data, f, indent=2, ensure_ascii=False)
                # print(f"  -> Salvo como: {output_path}")
            else:
                print(f"  -> AVISO: Nenhuma infobox encontrada em {url}")

            # 8. Pausa de Rate Limiting entre as requisições.
            time.sleep(1)

        except requests.RequestException as e:
            print(f"  -> ERRO ao processar {url}: {e}")
        except Exception as e:
            print(f"  -> ERRO inesperado: {e}")

    print("\n--- Processamento da Fase 2 Concluído ---")
    print(f"Dados brutos salvos em '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
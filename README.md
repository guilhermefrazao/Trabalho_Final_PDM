# 🚀 Projeto de Pipeline de Dados de Filmes (Wikipedia)

Este é um projeto acadêmico para a disciplina de Processamento de Dados Massivos. O objetivo é construir um pipeline de dados completo (ETL) que coleta informações de filmes da Wikipédia, processa e armazena esses dados em um banco de dados normalizado (Camada Prata), pronto para ser consumido por uma equipe de IA para consultas "Text-to-SQL".

---

## 🏛️ Arquitetura (Medallion)

O pipeline segue a arquitetura Medallion para garantir a qualidade e a rastreabilidade dos dados:

1.  **Camada Bronze (Dados Brutos):**
    * **Formato:** Milhares de arquivos `.json` individuais, um para cada filme.
    * **Origem:** Web scraping da "Infobox" de cada página de filme na Wikipédia.
    * **Armazenamento:** Salvo localmente na pasta `/bronze_data`.
    * **Qualidade:** Dados brutos, "sujos", não processados, exatamente como foram coletados.

2.  **Camada Prata (Dados Limpos e Normalizados):**
    * **Formato:** Um banco de dados relacional (PostgreSQL).
    * **Origem:** Resultado do script de ETL (`003_bronze_para_prata.py`) que lê da Camada Bronze.
    * **Armazenamento:** Container Docker (`silver_db_postgres`).
    * **Qualidade:** Dados limpos, tratados (ex: `$100 million` -> `100000000`), normalizados (separados em tabelas `movies`, `people`, etc.) e prontos para consulta.

3.  **Entregável (Exportação):**
    * **Formato:** Arquivos `.csv` (um para cada tabela da Camada Prata).
    * **Origem:** Script de exportação (`004_prata_csv.py`) que lê do PostgreSQL.
    * **Armazenamento:** Salvo localmente na pasta `/silver_exports`.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Web Scraping:** `requests`, `BeautifulSoup4`
* **ETL e Transformação:** `pandas`, `SQLAlchemy`
* **Banco de Dados (Prata):** PostgreSQL
* **Infraestrutura:** Docker e `docker-compose` (para rodar o PostgreSQL)

---

## 📦 Estrutura dos Scripts

* `001_coletor_de_links.py`: **Coletor de Links.** Varre as listas de filmes por ano e gera o `links_para_visitar.txt`.
* `002_extrator_infobox.py`: **Extrator Bronze.** Lê o `.txt`, visita cada link, extrai a Infobox e salva os JSONs brutos na pasta `/bronze_data`.
* `003_bronze_para_prata.py`: **Pipeline ETL.** Lê os JSONs da `/bronze_data`, limpa, transforma, normaliza e carrega os dados nas tabelas do PostgreSQL.
* `004_prata_csv.py`: **Exportador CSV.** Conecta-se ao PostgreSQL, lê as tabelas Prata e as salva como arquivos CSV na pasta `/silver_exports`.
* `docker-compose.yml`: Arquivo de configuração para iniciar o container do banco de dados PostgreSQL.
* `requirements.txt`: Lista de todas as dependências Python do projeto.

## 🚀 Como Executar o Pipeline (Passo a Passo)

Siga estes passos na ordem correta para executar o projeto do zero.

### 1. Clonar o Repositório

```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio
```
### 2. Executando código: 

**Documentação usada**
https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

**Criando ambiente airflow**

*Windows* 
```bash
mkdir dags, logs, plugins, config
```

*Linux*
```bash
mkdir -p ./dags ./logs ./plugins ./config
```

**Adicionar Variavel de ambiente**

*Windows* 
```bash
"AIRFLOW_UID=5000" | Out-File -Encoding UTF8 -FilePath .env
```

*Linux*
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

**Inicializando arquivo de configurações**

```bash
docker compose run airflow-cli airflow config list
```


**Instanciando banco de dados e criando conta**

```bash
docker compose up airflow-init
```

- Criada a conta Airflow, com login: "airflow" e senha: "airflow"


**Rodando Airflow**

```bash
docker compose up
```


**Reiniciando servidor ao subir alterações**

```bash
docker compose restart
```


### 4. Iniciar o Banco de Dados (Camada Prata)

Com o Docker Desktop aberto e em execução, inicie o container do PostgreSQL:

```bash
docker-compose up -d
```

*O banco de dados agora está rodando em segundo plano.*



### 5. Utilizando terraform para subir a arquitetura para produção:

```bash
terraform init --upgrade
```

```bash
terraform apply -target=google_artifact_registry_repository.repo
```


```bash
terraform apply -var="image_tag=tag_da_imagem_docker_artifact_repository"
```


## 📊 Esquema da Camada Prata (Entregável)

O pipeline gera as seguintes tabelas normalizadas, que são exportadas para CSV:

* **`movies`**
    * `movie_id` (Chave Primária)
    * `title` (Título)
    * `release_date_clean` (Data de Lançamento)
    * `budget_clean` (Orçamento)
    * `box_office_clean` (Bilheteria)
    * `director_id` (Chave Estrangeira -> `people.person_id`)
    * `running_time_min` (Duração em minutos)
    * `language` (Idioma principal)
    * `countries_list` (Lista de países)
    * `production_companies_list` (Lista de produtoras)

* **`people`**
    * `person_id` (Chave Primária)
    * `name` (Nome real da pessoa, ex: "Christopher Nolan")

* **`movie_actors`** (Tabela de Junção)
    * `movie_id` (Chave Estrangeira -> `movies.movie_id`)
    * `person_id` (Chave Estrangeira -> `people.person_id`)

* **`movie_writers`** (Tabela de Junção)
    * `movie_id`
    * `person_id`

* **`movie_composers`** (Tabela de Junção)
    * `movie_id`
    * `person_id`
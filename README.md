# Trabalho_Final_PDM

# Assistente de IA para Dados do IMDb

**Disciplina:** Processamento de Dados Massivos

**Status:** `[Em Andamento]`

## 👥 Equipe

  * Anna Pietra Vitória León Bastos Moreira
  * Daniel Henrique Pinheiro Silva
  * Guilherme Frazão Fernandes
  * Luis Eduardo Fonseca Alves Ferreira Mathias Cruvinel
  * Maria Carolina Xavier de Almeida

## 1\. 🎯 Objetivo

Este projeto tem como objetivo desenvolver uma solução completa de processamento de dados massivos, desde a ingestão de dados brutos até a disponibilização de dados limpos para alimentar um **Assistente de IA**. O assistente será capaz de responder perguntas sobre filmes, atores, diretores e suas avaliações, usando o dataset do IMDb.

## 2\. 🏛️ Arquitetura e Tecnologias

Para este projeto, adotamos uma arquitetura moderna e escalável na nuvem, utilizando o **Google Cloud Platform (GCP)** e o **Google BigQuery** como nossa principal ferramenta de processamento e armazenamento.

Seguimos a **Arquitetura Medallion** para organizar nosso pipeline:

  * **Camada Bronze (Brutos):** os dados originais do IMDb, acessados diretamente do dataset público `bigquery-public-data.imdb`. Nenhum dado é movido ou duplicado, apenas lido.
  * **Camada Prata (Processados):** nossos dados de negócio limpos, filtrados e enriquecidos. Estão armazenados no nosso próprio dataset: `imdb_prata`.
  * **Camada Ouro (Agregados):** *[Próximo Passo]* tabelas ou *views* agregadas, prontas para serem consumidas por modelos de Machine Learning ou dashboards.

## 3\. ⚙️ Pipeline de Dados (Bronze ➔ Prata)

A primeira fase do projeto foi a engenharia de dados para criar a Camada Prata. O processo foi o seguinte:

1.  **Ingestão:** leitura direta das tabelas `title_basics`, `title_ratings`, `title_principals` e `name_basics` da Camada Bronze.
2.  **Filtragem:** selecionamos apenas filmes (`title_type = 'movie'`) lançados do ano 2000 em diante.
3.  **Limpeza:**
      * Removemos filmes que não possuíam título (`primary_title IS NOT NULL`).
      * Padronizamos valores nulos (`NULL`) em colunas como `genres` (para 'Desconhecido') e `runtimeMinutes` (para 0).
4.  **Enriquecimento:**
      * Juntamos (`LEFT JOIN`) os filmes com suas respectivas notas (`title_ratings`) para criar a tabela `filmes_com_notas`.
      * Cruzamos (`INNER JOIN`) os filmes limpos com seus atores e diretores (`principals` e `name_basics`) para criar a tabela `pessoas_do_filme`.
5.  **Armazenamento:** os resultados foram salvos como duas novas tabelas na Camada Prata.

## 4\.  Tabelas Prata 

### Tabela 1: `imdb_prata.filmes_com_notas`

  * **Descrição:** tabela central de filmes, limpa e enriquecida com notas.
  * **Tamanho:** \~365 mil linhas (\~25 MB)
  * **Colunas Principais:**
      * `id_filme`: (string) ID único do filme (ex: `tt0133093`).
      * `titulo_principal`: (string) Título do filme.
      * `ano_lancamento`: (int) Ano de lançamento.
      * `duracao_minutos`: (int) Duração (0 se for nulo).
      * `generos`: (string) Gêneros (ex: 'Action,Sci-Fi').
      * `nota_media`: (float) Nota de 0 a 10 (pode ser `NULL` se não houver nota).
      * `num_votos`: (int) Número de votos (pode ser `NULL`).

### Tabela 2: `imdb_prata.pessoas_do_filme`

  * **Descrição:** tabela de mapeamento que conecta filmes aos seus atores e diretores.
  * **Tamanho:** \~1.6 milhão de linhas (\~84 MB)
  * **Colunas Principais:**
      * `id_filme`: (string) ID único do filme (chave para `filmes_com_notas`).
      * `titulo_principal`: (string) Título do filme (para facilitar a leitura).
      * `nome_pessoa`: (string) Nome do ator ou diretor.
      * `categoria`: (string) Função da pessoa ('actor' ou 'director').

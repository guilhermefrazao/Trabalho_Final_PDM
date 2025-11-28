#!/bin/sh

# Verifica se a variável de ambiente existe
if [ -z "$AIRFLOW_HOST" ]; then
    echo "Aviso: Variável AIRFLOW_HOST não definida. Usando localhost como fallback."
    export AIRFLOW_HOST="http://localhost:8000"
fi

echo "Injetando URL da API: $AIRFLOW_HOST"

# Substitui o placeholder no index.html pelo valor da variável
# Usamos pipes | como separador caso a URL tenha barras /
sed -i "s|__API_URL_PLACEHOLDER__|$AIRFLOW_HOST|g" /usr/share/nginx/html/index.html

# Inicia o Nginx normalmente
exec nginx -g "daemon off;"
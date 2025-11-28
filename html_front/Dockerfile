# Dockerfile para servir o frazo_front via Nginx na porta 4000

FROM nginx:alpine

# Remove config default
RUN rm /etc/nginx/conf.d/default.conf

# Copia o index.html para a pasta padrão do nginx
COPY index.html /usr/share/nginx/html/index.html

# Copia a config customizada
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Porta interna do container
EXPOSE 4000

CMD ["nginx", "-g", "daemon off;"]

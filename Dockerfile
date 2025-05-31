FROM python:3.10-slim

# Instalar utilidades necesarias
RUN apt-get update && \
    apt-get install -y --no-install-recommends adduser ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Crear usuario no root
RUN adduser --disabled-password --gecos "" app

# Crear directorio de trabajo y de mlruns con permisos adecuados

WORKDIR /app

RUN mkdir -p /mlruns && chown -R app:app /mlruns

RUN mkdir -p /app/models && chown -R app:app /app/models

RUN mkdir -p /app/models/sklearn-model && chown -R app:app /app/models/sklearn-model

RUN mkdir -p /app/logs && chown -R app:app /app/logs

RUN mkdir -p /app/.metrics && chown -R app:app /app/.metrics

RUN mkdir -p /app/.envs && chown -R app:app /app/.envs

# Copiar dependencias e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Descargar e instalar dockerize
ADD https://github.com/jwilder/dockerize/releases/download/v0.6.1/dockerize-linux-amd64-v0.6.1.tar.gz /tmp/
RUN tar -C /usr/local/bin -xzvf /tmp/dockerize-linux-amd64-v0.6.1.tar.gz && rm /tmp/dockerize-linux-amd64-v0.6.1.tar.gz

# Copiar el script de entrenamiento y json del model settings:
COPY scripts/train.py .

# Cambiar a usuario no root
USER app
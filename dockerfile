# Usa a imagem base leve do Python 3.8 Alpine
FROM python:3.8-alpine

# Instala dependências do sistema e ferramentas essenciais para compilação
RUN apk update && apk add --no-cache \
    build-base \
    cmake \
    libjpeg-turbo-dev \
    zlib-dev \
    libpng-dev \
    jpeg-dev \
    freetype-dev \
    tiff-dev \
    libx11-dev \
    wget \
    bash

# Define o diretório de trabalho
WORKDIR /app

# Cria e ativa um ambiente virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia o arquivo requirements.txt com as dependências do Python
COPY requirements.txt .

# Instala as dependências do Python no ambiente virtual
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia os arquivos de código para o container
COPY . .

# Baixa o modelo de landmarks dlib
RUN wget -O shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Exponha a porta que o Streamlit usa
EXPOSE 8501

# Comando de entrada para rodar o Streamlit no ambiente virtual
CMD ["streamlit", "run", "app.py"]

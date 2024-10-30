# Usa a imagem base oficial do Python 3.8
FROM python:3.8-slim

# Instala dependências essenciais do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    libboost-all-dev \
    wget \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

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

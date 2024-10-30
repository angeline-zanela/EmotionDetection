# Usa a imagem completa do Python 3.8 com Debian Bullseye para garantir compatibilidade
FROM python:3.8-bullseye

# Define o diretório de trabalho
WORKDIR /app

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
    libopenblas-dev \
    wget \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Cria e ativa um ambiente virtual para o Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia o arquivo requirements.txt com as dependências do Python
COPY requirements.txt .

# Instala as dependências do Python no ambiente virtual
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação para o container
COPY . .

# Baixa e descompacta o modelo de landmarks dlib
RUN wget -O shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Exponha a porta utilizada pelo Streamlit
EXPOSE 8501

# Comando de entrada para iniciar o Streamlit
CMD ["streamlit", "run", "app.py"]

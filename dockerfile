# Usa a imagem base oficial do Python 3.8
FROM python:3.8-slim

# Instala dependências essenciais do sistema
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo requirements.txt com as dependências do Python
COPY requirements.txt .

# Atualiza pip e instala as dependências do Python
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos de código para o container
COPY . .

# Baixa o modelo de landmarks dlib
RUN wget -O shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Exponha a porta que o Streamlit usa
EXPOSE 8501

# Comando de entrada para rodar o Streamlit
CMD ["streamlit", "run", "app.py"]

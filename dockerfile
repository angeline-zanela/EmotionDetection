# Use uma imagem base oficial do Python
FROM python:3.9-slim

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie o arquivo de requisitos para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o código da aplicação para o diretório de trabalho
COPY . /app

# Baixe o arquivo de preditores de landmarks faciais do dlib
RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    mv shape_predictor_68_face_landmarks.dat /app

# Instale dependências necessárias para o OpenCV e outras dependências do sistema
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

# Exponha a porta padrão do Streamlit
EXPOSE 8501

# Defina o comando de inicialização para rodar o Streamlit com o arquivo app.py
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

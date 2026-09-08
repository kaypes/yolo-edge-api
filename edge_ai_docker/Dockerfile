# O PDF pede bullseye (Debian 11), mas o repositório de segurança dele já
# expirou (EOL) -- apt-get update falha com "Release file ... is expired".
# bookworm (Debian 12) é o sucessor direto, ainda com repositório ativo; os
# wheels manylinux do TensorFlow/OpenCV funcionam nele sem alteração.
FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libgl1 \
    libglib2.0-0 \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# tensorflow==2.13.0 exige numpy>=1.22,<=1.24.3. O opencv-python mais recente
# (5.x) exige numpy>=2 no Python 3.10 -- pip install de tudo junto (como o PDF
# pede) causa conflito de resolução (ou falha, ou baixa dezenas de wheels em
# backtracking). Fixamos opencv-python==4.10.0.84 (pede só numpy>=1.21.2,
# tem wheel cp310 aarch64) e numpy==1.24.3 (o teto do tensorflow) para o
# ambiente resolver de primeira.
# "nano" no pip install é o do PDF -- na verdade é um pacote Django sem
# relação com o editor de texto (esse já foi instalado via apt acima).
# Inofensivo, mantido por fidelidade ao roteiro.
RUN pip install \
    tensorflow==2.13.0 \
    numpy==1.24.3 \
    opencv-python==4.10.0.84 \
    psutil \
    requests \
    pillow \
    nano

COPY . /app

CMD ["bash"]

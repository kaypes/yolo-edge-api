# O PDF pede bullseye (Debian 11), mas o repositório de segurança dele já
# expirou (EOL) -- apt-get update falha com "Release file ... is expired".
# bookworm (Debian 12) é o sucessor direto, ainda com repositório ativo (mesma
# correção da Aula 8).
FROM python:3.10-slim-bookworm

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install \
    tensorflow==2.13.0 \
    cryptography \
    requests \
    pillow \
    numpy

COPY . /app

CMD ["bash"]

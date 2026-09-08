import os

import tensorflow as tf
from cryptography.fernet import Fernet

# Recuperar chave da variável de ambiente
chave = os.getenv("MODEL_KEY")

if chave is None:
    raise ValueError("Variável MODEL_KEY não definida!")

# Criar objeto de descriptografia
cipher = Fernet(chave.encode())

# Ler modelo criptografado
with open("modelo/mobilenet_fp32_encrypted.bin", "rb") as f:
    dados_criptografados = f.read()

# Descriptografar conteúdo em memória
modelo_bytes = cipher.decrypt(dados_criptografados)

# Carregar modelo diretamente a partir dos bytes
interpreter = tf.lite.Interpreter(model_content=modelo_bytes)
interpreter.allocate_tensors()

print("Modelo MobileNet (ImageNet) carregado com sucesso em memória!")

from cryptography.fernet import Fernet

modelo_original = "modelo/mobilenet_fp32.tflite"
modelo_criptografado = "modelo/mobilenet_fp32_encrypted.bin"

# Gerar chave criptográfica
chave = Fernet.generate_key()

print("\nChave gerada (copie e guarde com segurança):")
print(chave.decode())

# Criar objeto de criptografia
cipher = Fernet(chave)

# Ler modelo original
with open(modelo_original, "rb") as f:
    dados = f.read()

# Criptografar conteúdo
dados_criptografados = cipher.encrypt(dados)

# Salvar modelo criptografado
with open(modelo_criptografado, "wb") as f:
    f.write(dados_criptografados)

print("\nModelo criptografado com sucesso!")
print("Arquivo salvo em:", modelo_criptografado)

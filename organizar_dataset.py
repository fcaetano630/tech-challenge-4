import os
import random
import shutil

# --- CONFIGURAÇÃO ---
# O script assume que você tem uma pasta 'train' com 'images' e 'labels' dentro
base_path = "dataset_medico"
train_img_dir = os.path.join(base_path, "train", "images")
train_lab_dir = os.path.join(base_path, "train", "labels")
val_img_dir = os.path.join(base_path, "val", "images")
val_lab_dir = os.path.join(base_path, "val", "labels")

# Criar as pastas de validação se não existirem
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lab_dir, exist_ok=True)

# --- EXECUÇÃO ---

# 1. Lista apenas os arquivos de imagem na pasta train/images
imagens = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 2. Define 20% para validação
quantidade_val = int(len(imagens) * 0.2)
imagens_selecionadas = random.sample(imagens, quantidade_val)

print(f"Movendo {len(imagens_selecionadas)} pares (imagem + txt) para a pasta 'val'...")

sucesso = 0
erros = 0

for img_nome in imagens_selecionadas:
    # Caminho de origem da imagem
    origem_img = os.path.join(train_img_dir, img_nome)
    
    # Descobre o nome do arquivo .txt (ex: foto1.jpg -> foto1.txt)
    txt_nome = os.path.splitext(img_nome)[0] + ".txt"
    origem_txt = os.path.join(train_lab_dir, txt_nome)
    
    # Verifica se o arquivo .txt realmente existe antes de mover
    if os.path.exists(origem_txt):
        # Move a imagem
        shutil.move(origem_img, os.path.join(val_img_dir, img_nome))
        # Move o label (.txt)
        shutil.move(origem_txt, os.path.join(val_lab_dir, txt_nome))
        sucesso += 1
    else:
        print(f"Aviso: O label para {img_nome} não foi encontrado em {train_lab_dir}")
        erros += 1

print(f"Concluído! {sucesso} pares movidos com sucesso. {erros} erros.")
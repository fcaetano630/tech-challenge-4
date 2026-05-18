import os
from ultralytics import YOLO
from collections import Counter

class InstrumentDetector:
    def __init__(self, model_path='models/medico_yolo.pt'):
        """
        Inicializa o detector com o modelo treinado.
        """
        # Caminho absoluto para evitar erros de diretório
        caminho_base = os.path.abspath(os.path.dirname(__file__))
        modelo_abs_path = os.path.join(caminho_base, model_path)
        
        if not os.path.exists(modelo_abs_path):
            raise FileNotFoundError(f"❌ Modelo não encontrado em: {modelo_abs_path}")
            
        self.model = YOLO(modelo_abs_path)
        print(f"✅ Modelo carregado: {model_path}")

    def analisar_pasta_automaticamente(self):
        """
        Varre a pasta data/images, detecta instrumentos e conta as quantidades.
        """
        caminho_base = os.path.abspath(os.getcwd())
        pasta_entrada = os.path.join(caminho_base, 'data', 'images')
        pasta_saida = os.path.join(caminho_base, 'data', 'results')

        os.makedirs(pasta_entrada, exist_ok=True)
        os.makedirs(pasta_saida, exist_ok=True)

        extensoes = ('.jpg', '.jpeg', '.png', '.webp')
        arquivos = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(extensoes)]

        if not arquivos:
            print(f"⚠️ Nenhuma imagem encontrada em: {pasta_entrada}")
            return

        print(f"🔍 Iniciando análise de {len(arquivos)} imagem(ns)...\n")

        for arquivo in arquivos:
            caminho_img = os.path.join(pasta_entrada, arquivo)
            
            # O predict do YOLO retorna uma lista de resultados (um por imagem)
            results = self.model.predict(
                source=caminho_img,
                conf=0.5,
                save=True,
                project=os.path.join(caminho_base, 'data'),
                name='results',
                exist_ok=True
            )

            # Lista para armazenar todos os nomes detectados nesta imagem
            objetos_detectados = []

            for r in results:
                for box in r.boxes:
                    nome_classe = r.names[int(box.cls[0])]
                    objetos_detectados.append(nome_classe)

            # Contagem das quantidades
            resumo_contagem = Counter(objetos_detectados)

            print(f"📊 Resultado para: {arquivo}")
            if not resumo_contagem:
                print("   ➔ Nenhum instrumento identificado.")
            else:
                for instrumento, qtd in resumo_contagem.items():
                    print(f"   ➔ {instrumento}: {qtd}")
            
            print(f"📂 Imagem salva em: data/results/{arquivo}\n" + "-"*40)

    def analisar_video(self, caminho_video):
        """
        Analisa um vídeo clínico em busca de instrumentos e salva o resultado anotado.
        """
        if not os.path.exists(caminho_video):
            raise FileNotFoundError(f"❌ Vídeo não encontrado: {caminho_video}")

        caminho_base = os.path.abspath(os.path.dirname(__file__))
        pasta_saida = os.path.join(caminho_base, 'data', 'results', 'video_results')
        os.makedirs(pasta_saida, exist_ok=True)

        print(f"🎥 Iniciando análise de vídeo: {os.path.basename(caminho_video)}")
        results = self.model.predict(
            source=caminho_video,
            conf=0.5,
            save=True,
            project=os.path.join(caminho_base, 'data'),
            name='video_results',
            exist_ok=True
        )

        objetos_detectados = []
        for r in results:
            for box in r.boxes:
                nome_classe = r.names[int(box.cls[0])]
                objetos_detectados.append(nome_classe)

        resumo_contagem = Counter(objetos_detectados)
        print(f"📊 Resultado para o vídeo: {os.path.basename(caminho_video)}")
        if not resumo_contagem:
            print("   ➔ Nenhum instrumento identificado no vídeo.")
        else:
            for instrumento, qtd in resumo_contagem.items():
                print(f"   ➔ {instrumento}: {qtd}")

        print(f"📂 Vídeo anotado salvo em: data/results/video_results\n" + "-"*40)
        return objetos_detectados

# --- EXECUÇÃO DO SCRIPT ---
if __name__ == "__main__":
    try:
        detector = InstrumentDetector()
        detector.analisar_pasta_automaticamente()
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
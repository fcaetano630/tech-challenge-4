import whisper
import os

class MedicalAudioAnalyzer:
    def __init__(self, model_size="base"):
        """
        Inicializa o Whisper localmente.
        """
        print(f"--- Carregando Modelo Whisper ({model_size}) ---")
        self.model = whisper.load_model(model_size)

    def processar_todos_os_videos(self):
        # Define o caminho da pasta de vídeos
        caminho_base = os.getcwd()
        pasta_videos = os.path.join(caminho_base, 'data', 'videos')

        if not os.path.exists(pasta_videos):
            print(f"❌ Pasta não encontrada: {pasta_videos}")
            return

        # Lista os arquivos suportados
        extensoes_suportadas = ('.mp4', '.mkv', '.avi', '.mov', '.mp3', '.wav')
        arquivos = [f for f in os.listdir(pasta_videos) if f.lower().endswith(extensoes_suportadas)]

        if not arquivos:
            print(f"⚠️ Nenhum arquivo de vídeo ou áudio encontrado em: {pasta_videos}")
            return

        print(f"📂 Encontrados {len(arquivos)} arquivos para processamento.\n")

        for arquivo in arquivos:
            caminho_completo = os.path.join(pasta_videos, arquivo)
            print(f"🎬 Processando: {arquivo}...")

            try:
                # Transcrição local (fp16=False para rodar melhor em CPUs sem GPU)
                result = self.model.transcribe(caminho_completo, language="pt", fp16=False)
                
                print(f"📝 Transcrição de {arquivo}:")
                print(f"'{result['text'].strip()}'")
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Erro ao processar {arquivo}: {e}")

if __name__ == "__main__":
    analyzer = MedicalAudioAnalyzer()
    analyzer.processar_todos_os_videos()
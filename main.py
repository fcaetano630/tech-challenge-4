import os
import sys
import shutil

# Importando as classes exatamente como você as definiu
try:
    from audio_module import MedicalAudioAnalyzer
    from vision_module import InstrumentDetector # Salve o código de visão com esse nome
    from agent_module import HospitalAgent
except ImportError as e:
    print(f"❌ Erro de importação: {e}. Verifique se os nomes dos arquivos .py estão corretos.")
    sys.exit()

def executar_fluxo_multimodal():
    print("\n" + "="*60)
    print("🏥 INICIANDO SISTEMA MULTIMODAL - TECH CHALLENGE 4".center(60))
    print("="*60)

    # Ensure ffmpeg is discoverable by subprocess calls (used by Whisper)
    def ensure_ffmpeg_on_path():
        if shutil.which('ffmpeg'):
            return True
        candidates = [
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'WinGet', 'Packages', 'Gyan.FFmpeg_Microsoft.WinGet.Source_8wekyb3d8bbwe', 'ffmpeg-8.1.1-full_build', 'bin'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'ffmpeg', 'bin'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Gyan', 'ffmpeg', 'bin'),
            os.path.join(os.environ.get('ProgramFiles', ''), 'ffmpeg', 'bin'),
            'C:\\ffmpeg\\bin',
        ]
        for p in candidates:
            ff = os.path.join(p, 'ffmpeg.exe')
            if os.path.exists(ff):
                os.environ['PATH'] = p + os.pathsep + os.environ.get('PATH', '')
                print(f"✅ Adicionado ffmpeg ao PATH a partir de: {p}")
                return True
        print("⚠️ ffmpeg não encontrado em locais comuns. Instale ffmpeg ou adicione-o ao PATH.")
        return False

    ensure_ffmpeg_on_path()

    # 1. Instanciando os Módulos
    try:
        audio_analyser = MedicalAudioAnalyzer(model_size="base")
        vision_detector = InstrumentDetector()
        agente = HospitalAgent()
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return

    # 2. Processamento de Áudio (Whisper)
    print("\n🎤 [AUDIO] Transcrevendo vídeo em 'data/videos'...")
    pasta_videos = os.path.join(os.getcwd(), 'data', 'videos')
    arquivos_videos = [f for f in os.listdir(pasta_videos) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]

    if not arquivos_videos:
        print("⚠️ Nenhum vídeo encontrado em 'data/videos'.")
        return

    caminho_video = os.path.join(pasta_videos, arquivos_videos[0])
    transcricao_final = audio_analyser.transcrever_video(caminho_video)

    # 3. Processamento de Visão em Vídeo (YOLO)
    print("\n🎥 [VIDEO] Analisando o vídeo para detecção de instrumentos...")
    try:
        objetos_video = vision_detector.analisar_video(caminho_video)
    except Exception as e:
        print(f"❌ Erro na análise do vídeo: {e}")
        objetos_video = []

    # 4. Processamento de Visão em Imagens (YOLO)
    print("\n👁️ [VISION] Detectando instrumentos em 'data/images'...")
    pasta_imagens = os.path.join(os.getcwd(), 'data', 'images')
    arquivos_img = [f for f in os.listdir(pasta_imagens) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    objetos_encontrados = []
    if arquivos_img:
        img_path = os.path.join(pasta_imagens, arquivos_img[0])
        results = vision_detector.model.predict(source=img_path, conf=0.5, save=False)
        for r in results:
            for box in r.boxes:
                nome_classe = r.names[int(box.cls[0])]
                objetos_encontrados.append(nome_classe)
    else:
        print("⚠️ Nenhuma imagem encontrada em 'data/images'.")

    # 5. Análise do Agente (Hugging Face)
    print("\n🧠 [AGENT] Cruzando dados e gerando relatório final...")
    relatorio = agente.analisar_atendimento(transcricao_final, objetos_encontrados, objetos_video)

    # 5. Output Final
    print("\n" + "📋 RELATÓRIO DE CONFORMIDADE MÉDICA".center(60))
    print("-" * 60)
    print(f"✅ STATUS TÉCNICO: {relatorio['Conformidade Técnica']}")
    print(f"🎭 ANÁLISE EMOCIONAL: {relatorio['Análise Emocional']}")
    print(f"🎬 ANÁLISE DE VÍDEO: {relatorio['Análise de Vídeo']}")
    print(f"📝 RESUMO: {relatorio['Resumo']}")
    print("\n🚨 ALERTAS DE SEGURANÇA:")
    for alerta in relatorio['Alertas de Saúde']:
        print(f"  - {alerta}")
    print("-" * 60)
    print("🎯 Processamento Multimodal Concluído.")

if __name__ == "__main__":
    executar_fluxo_multimodal()
import os
import sys

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

    # 1. Instanciando os Módulos
    try:
        audio_analyser = MedicalAudioAnalyzer(model_size="base")
        vision_detector = InstrumentDetector()
        agente = HospitalAgent()
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return

    # 2. Processamento de Áudio (Whisper)
    # Como seu código processa a pasta 'data/videos', vamos pegar o resultado
    print("\n🎤 [AUDIO] Transcrevendo arquivos em 'data/videos'...")
    # Dica: No seu audio_module.py, faça o método retornar o texto. 
    # Por enquanto, vamos simular a captura do resultado do vídeo de Papanicolaou:
    caminho_video = os.path.join(os.getcwd(), 'data', 'videos', 'Instrução Prática Para Coleta de Papanicolaou [jX8aDMQD8j4].mp4')
    
    if os.path.exists(caminho_video):
        # Aqui chamamos o modelo diretamente para pegar o texto para o Agente
        res = audio_analyser.model.transcribe(caminho_video, language="pt", fp16=False)
        transcricao_final = res['text']
    else:
        print("⚠️ Vídeo específico não encontrado para análise do Agente.")
        return

    # 3. Processamento de Visão (YOLO)
    print("\n👁️ [VISION] Detectando instrumentos em 'data/images'...")
    # Vamos capturar os objetos detectados na primeira imagem da pasta
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

    # 4. Análise do Agente (Hugging Face)
    print("\n🧠 [AGENT] Cruzando dados e gerando relatório final...")
    relatorio = agente.analisar_atendimento(transcricao_final, objetos_encontrados)

    # 5. Output Final
    print("\n" + "📋 RELATÓRIO DE CONFORMIDADE MÉDICA".center(60))
    print("-" * 60)
    print(f"✅ STATUS TÉCNICO: {relatorio['Conformidade Técnica']}")
    print(f"🎭 ANÁLISE EMOCIONAL: {relatorio['Análise Emocional']}")
    print(f"📝 RESUMO: {relatorio['Resumo']}")
    print("\n🚨 ALERTAS DE SEGURANÇA:")
    for alerta in relatorio['Alertas de Saúde']:
        print(f"  - {alerta}")
    print("-" * 60)
    print("🎯 Processamento Multimodal Concluído.")

if __name__ == "__main__":
    executar_fluxo_multimodal()
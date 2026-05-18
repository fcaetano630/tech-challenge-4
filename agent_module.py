import os
from dotenv import load_dotenv
from transformers import pipeline

# Carrega as variáveis do arquivo .env
load_dotenv()

class HospitalAgent:
    def __init__(self):
        # Busca a chave do .env
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("❌ Erro: HF_TOKEN não encontrado no arquivo .env")

        print("🤖 Inicializando Agente de IA (Hugging Face)...")
        
        # Inicializa o pipeline de análise de sentimento para Português
        self.sentiment_task = pipeline(
            "sentiment-analysis", 
            model="pysentimiento/robertuito-sentiment-analysis", 
            token=hf_token
        )

    def analisar_atendimento(self, texto_whisper, objetos_yolo_imagem, objetos_yolo_video=None):
        print("🧠 Analisando conformidade e riscos...")
        
        objetos_yolo_video = objetos_yolo_video or []
        todos_objetos = objetos_yolo_imagem + objetos_yolo_video

        # 1. Análise de Sentimento (focada em detectar tensão/medo)
        analise_sentimento = self.sentiment_task(texto_whisper, truncation=True, max_length=128)[0]
        
        # 2. Cruzamento Multimodal (O que foi dito vs O que foi visto)
        obrigatorios = [ "Speculum", "Slide", "Glove", "Cervical Brush", "Ayre Spatula", "Pozzi Forceps", "Kelly Forceps", "Cheron Forceps", "Needle Holder", "Straight Mayo Scissor", "Straight Dissection Clamp"]
        vistos = [obj for obj in set(todos_objetos) if obj in obrigatorios]
        faltantes = list(set(obrigatorios) - set(vistos))

        # 3. Detecção de Alertas de Saúde (Anomalias)
        alertas = []
        termos_risco = ["sangramento", "lesão", "dor", "medo", "ansiedade", "agressão", "casa", "machucado", "briga", "forçou", "trauma", "ansiosa"]
        
        for termo in termos_risco:
            if termo in texto_whisper.lower():
                alertas.append(f"🚨 ALERTA: Identificado termo de atenção: '{termo}'")

        if objetos_yolo_video and any(termo in texto_whisper.lower() for termo in termos_risco):
            alertas.append("🚨 ALERTA: Sinais verbais de risco alinhados a detecções visuais no vídeo.")

        if not objetos_yolo_video:
            alertas.append("⚠️ Vídeo analisado não apresentou instrumentos detectados, verifique a qualidade do vídeo ou possível anomalia no procedimento.")

        resumo_video = (
            f"Classes detectadas no vídeo: {', '.join(sorted(set(objetos_yolo_video)))}."
            if objetos_yolo_video else
            "Nenhum instrumento detectado no vídeo."
        )

        return {
            "Conformidade Técnica": "✅ OK" if not faltantes else f"⚠️ ATENÇÃO: Itens críticos não detectados visualmente: {faltantes}",
            "Análise Emocional": f"{analise_sentimento['label']} (Score: {analise_sentimento['score']:.2f})",
            "Análise de Vídeo": resumo_video,
            "Alertas de Saúde": alertas if alertas else ["Nenhuma anomalia crítica detectada."],
            "Resumo": "Processamento concluído com base no protocolo de saúde da mulher."
        }

if __name__ == "__main__":
    # Teste rápido do Agente
    try:
        # Simulando os dados que você já obteve nos módulos anteriores
        texto_teste = "A paciente apresenta muita dor e sangramento, além de relatar ansiedade."
        objetos_vistos = ["Luva", "Lâmina"] # Faltou o espéculo para o Papanicolaou
        
        agente = HospitalAgent()
        relatorio = agente.analisar_atendimento(texto_teste, objetos_vistos)
        
        print("\n" + "="*40)
        print("📝 RESULTADO DA ANÁLISE MULTIMODAL")
        print("="*40)
        for campo, info in relatorio.items():
            print(f"**{campo}**: {info}")
            
    except Exception as e:
        print(f"❌ Falha ao rodar o agente: {e}")
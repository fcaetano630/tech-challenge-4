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
        
        # Dicionários especializados para Saúde da Mulher
        self.termos_depressao_pos_parto = {
            "tristeza": 0.9, "choro": 0.95, "desesperança": 0.95, "isolamento": 0.85,
            "falta de interesse": 0.9, "fadiga": 0.75, "culpa": 0.85, "incapacidade": 0.9,
            "não consigo": 0.8, "sem energia": 0.8, "cansada": 0.7, "exausta": 0.75,
            "bonding": -0.8, "rejeição ao bebê": 0.95, "ódio": 0.9, "ambivalência": 0.7,
            "pensamentos negativos": 0.9, "suicida": 0.99, "morrer": 0.95, "desistir": 0.9,
            "houve mudança": 0.6, "piora": 0.8, "piorou": 0.8, "começou": 0.6
        }
        
        self.termos_violencia_domestica = {
            "bate": 0.99, "bato": 0.99, "bater": 0.99, "socos": 0.98, "tapas": 0.98,
            "chutes": 0.98, "abusa": 0.99, "abusou": 0.99, "agride": 0.98, "agrediu": 0.98,
            "controla": 0.95, "controlar": 0.95, "isolada": 0.9, "proíbe": 0.95,
            "machuca": 0.95, "machucou": 0.95, "machucado": 0.9, "hematomas": 0.95,
            "medo dele": 0.95, "tenho medo": 0.85, "ameaça": 0.95, "ameaçou": 0.95,
            "ciúmes": 0.7, "ciumento": 0.7, "agressivo": 0.85, "agressão": 0.9,
            "foge de casa": 0.8, "fugiu": 0.8, "casa insegura": 0.9, "não é seguro": 0.85,
            "força": 0.8, "forçou": 0.95, "sem consentimento": 0.99, "não quis": 0.8,
            "marido": 0.1, "companheiro": 0.1, "namorado": 0.1, "ex": 0.1  # contexto
        }
        
        self.sinais_visuais_violencia = {
            "olhar baixo": 0.8, "evita contato visual": 0.75, "assustada": 0.9,
            "tremendo": 0.85, "nervosa": 0.7, "tensa": 0.7, "encolhida": 0.8,
            "protetora": 0.6, "abraços defensivos": 0.75, "afastada do acompanhante": 0.7
        }
        
        self.indicadores_clinicos_femininos = {
            "hemorragia": 0.95, "sangramento excessivo": 0.95, "lóquios": 0.6,
            "infecção": 0.85, "febre": 0.8, "mastite": 0.8, "fissura": 0.6,
            "preeclâmpsia": 0.95, "pressão alta": 0.8, "edema": 0.7,
            "trombose": 0.95, "incontinência": 0.7, "dispareunia": 0.8,
            "recuperação": -0.7, "melhor": -0.6, "bem": -0.5, "ótimo": -0.8
        }

    
    def detectar_depressao_pos_parto(self, texto):
        """
        Detecta sinais de Depressão Pós-Parto (DPP) através de análise de termos e sentimento.
        Retorna score de risco e indicadores específicos.
        """
        texto_lower = texto.lower()
        score_dpp = 0
        indicadores = []
        
        # Análise de termos específicos
        for termo, peso in self.termos_depressao_pos_parto.items():
            if termo in texto_lower:
                score_dpp += peso
                indicadores.append(f"'{termo}'")
        
        # Análise de sentimento geral
        try:
            sentimento = self.sentiment_task(texto[:512], truncation=True)[0]
            if sentimento['label'] == 'NEG':
                score_dpp += sentimento['score'] * 0.8
        except:
            pass
        
        # Contextualizar com período pós-parto
        periodos_risco = ["semana", "mês", "meses", "pós", "depois", "após", "parto"]
        em_periodo_risco = any(p in texto_lower for p in periodos_risco)
        
        if score_dpp > 0:
            score_dpp = min(100, (score_dpp / 5) * 100)
        
        risco_level = "CRÍTICO" if score_dpp > 75 else "ALTO" if score_dpp > 50 else "MODERADO" if score_dpp > 25 else "BAIXO"
        
        return {
            "score": score_dpp,
            "risco_level": risco_level,
            "indicadores": indicadores,
            "em_periodo_pos_parto": em_periodo_risco
        }
    
    def detectar_violencia_domestica(self, texto, comportamento_visual=None):
        """
        Detecta sinais de Violência Doméstica através de análise verbal e visual.
        Retorna score de risco e evidências específicas.
        """
        texto_lower = texto.lower()
        score_violencia = 0
        evidencias_verbais = []
        evidencias_visuais = []
        
        # Análise de termos indicadores de violência
        for termo, peso in self.termos_violencia_domestica.items():
            if termo in texto_lower:
                # Evita falsos positivos com contexto (ex: "marido" sozinho não é indicador)
                if termo not in ["marido", "companheiro", "namorado", "ex"] or any(
                    risco in texto_lower for risco in ["bate", "abusa", "agride", "ameaça", "controla"]
                ):
                    score_violencia += peso
                    if peso > 0.8:
                        evidencias_verbais.append(f"'{termo}' (risco alto)")
                    else:
                        evidencias_verbais.append(f"'{termo}'")
        
        # Análise de comportamento visual (se fornecido)
        if comportamento_visual:
            for sinal, peso in self.sinais_visuais_violencia.items():
                if sinal.lower() in comportamento_visual.lower():
                    score_violencia += peso
                    evidencias_visuais.append(f"Sinal visual: {sinal}")
        
        # Normalizar score
        score_violencia = min(100, score_violencia * 10)
        
        risco_level = "CRÍTICO" if score_violencia > 80 else "ALTO" if score_violencia > 50 else "MODERADO" if score_violencia > 25 else "BAIXO"
        
        return {
            "score": score_violencia,
            "risco_level": risco_level,
            "evidencias_verbais": evidencias_verbais,
            "evidencias_visuais": evidencias_visuais,
            "requer_notificacao": score_violencia > 50
        }
    
    def analisar_complicacoes_clinicas(self, texto):
        """
        Detecta complicações clínicas específicas da saúde da mulher pós-parto.
        """
        texto_lower = texto.lower()
        complicacoes = []
        score_risco_clinico = 0
        
        for termo, peso in self.indicadores_clinicos_femininos.items():
            if termo in texto_lower:
                if peso > 0:  # Indicador de risco
                    complicacoes.append(f"⚠️ {termo.upper()}")
                    score_risco_clinico += peso
                else:  # Indicador positivo (recuperação)
                    complicacoes.append(f"✓ {termo}")
                    score_risco_clinico -= abs(peso) * 0.5
        
        score_risco_clinico = max(0, min(100, score_risco_clinico * 10))
        
        return {
            "complicacoes": complicacoes,
            "score_risco_clinico": score_risco_clinico,
            "requer_atencao_imediata": score_risco_clinico > 70
        }

    def analisar_atendimento(self, texto_whisper, objetos_yolo_imagem, objetos_yolo_video=None, comportamento_visual=None):
        print("🧠 Analisando conformidade clínica e riscos de saúde feminina...")
        
        objetos_yolo_video = objetos_yolo_video or []
        todos_objetos = objetos_yolo_imagem + objetos_yolo_video

        # 1. Análise Especializada para Saúde da Mulher
        analise_dpp = self.detectar_depressao_pos_parto(texto_whisper)
        analise_violencia = self.detectar_violencia_domestica(texto_whisper, comportamento_visual)
        analise_clinica = self.analisar_complicacoes_clinicas(texto_whisper)
        
        # 2. Análise de Sentimento Geral
        analise_sentimento = self.sentiment_task(texto_whisper[:512], truncation=True, max_length=128)[0]
        
        # 3. Cruzamento Multimodal (O que foi dito vs O que foi visto)
        obrigatorios = ["Speculum", "Slide", "Glove", "Cervical Brush", "Ayre Spatula", "Pozzi Forceps", 
                        "Kelly Forceps", "Cheron Forceps", "Needle Holder", "Straight Mayo Scissor", "Straight Dissection Clamp"]
        vistos = [obj for obj in set(todos_objetos) if obj in obrigatorios]
        faltantes = list(set(obrigatorios) - set(vistos))

        # 4. Sistema de Alertas Multimodal e Especializado
        alertas = []
        
        # Alertas de violência doméstica (CRÍTICO)
        if analise_violencia["score"] > 50:
            alertas.append(f"🚨 ALERTA CRÍTICO - VIOLÊNCIA DOMÉSTICA DETECTADA (Score: {analise_violencia['score']:.1f}%)")
            alertas.append(f"   Evidências: {', '.join(analise_violencia['evidencias_verbais'][:3])}")
            if analise_violencia['requer_notificacao']:
                alertas.append("   ⚠️ REQUER NOTIFICAÇÃO A AUTORIDADES")
        
        # Alertas de depressão pós-parto (ALTO)
        if analise_dpp["score"] > 50:
            alertas.append(f"🚨 ALERTA - DEPRESSÃO PÓS-PARTO INDICADA (Score: {analise_dpp['score']:.1f}%)")
            alertas.append(f"   Nível de Risco: {analise_dpp['risco_level']}")
            if analise_dpp['indicadores']:
                alertas.append(f"   Indicadores: {', '.join(analise_dpp['indicadores'][:4])}")
        
        # Alertas de complicações clínicas (ALTO)
        if analise_clinica["score_risco_clinico"] > 50:
            alertas.append(f"🚨 ALERTA CLÍNICO - COMPLICAÇÕES PÓS-PARTO (Score: {analise_clinica['score_risco_clinico']:.1f}%)")
            if analise_clinica['complicacoes']:
                alertas.append(f"   Complicações: {', '.join(analise_clinica['complicacoes'][:4])}")
            if analise_clinica['requer_atencao_imediata']:
                alertas.append("   ⚠️ REQUER ATENÇÃO MÉDICA IMEDIATA")
        
        # Alertas gerais de saúde
        termos_risco_geral = ["sangramento", "lesão", "infecção", "febre", "tremendo", "ansiedade"]
        for termo in termos_risco_geral:
            if termo in texto_whisper.lower():
                alertas.append(f"⚠️ Termo de atenção detectado: '{termo}'")

        if objetos_yolo_video and any(termo in texto_whisper.lower() for termo in termos_risco_geral):
            alertas.append("⚠️ Sinais verbais de risco alinhados a detecções visuais no vídeo.")

        if not objetos_yolo_video:
            alertas.append("ℹ️ Vídeo: Nenhum instrumento detectado - verifique qualidade ou anomalia no procedimento.")

        resumo_video = (
            f"Classes detectadas: {', '.join(sorted(set(objetos_yolo_video)))}."
            if objetos_yolo_video else
            "Nenhum instrumento detectado no vídeo."
        )
        
        # Score de risco geral
        score_risco_total = (analise_dpp["score"] + analise_violencia["score"] + analise_clinica["score_risco_clinico"]) / 3
        prioridade_atendimento = "CRÍTICO" if score_risco_total > 70 else "ALTO" if score_risco_total > 50 else "MODERADO" if score_risco_total > 25 else "NORMAL"

        return {
            "Conformidade Técnica": "✅ OK" if not faltantes else f"⚠️ ATENÇÃO: Itens não detectados: {faltantes}",
            "Análise Emocional": f"{analise_sentimento['label']} (Score: {analise_sentimento['score']:.2f})",
            "Depressão Pós-Parto": f"{analise_dpp['risco_level']} (Score: {analise_dpp['score']:.1f}%)",
            "Risco de Violência": f"{analise_violencia['risco_level']} (Score: {analise_violencia['score']:.1f}%)",
            "Complicações Clínicas": f"Score: {analise_clinica['score_risco_clinico']:.1f}%",
            "Análise de Vídeo": resumo_video,
            "Prioridade de Atendimento": prioridade_atendimento,
            "Score Risco Total": f"{score_risco_total:.1f}%",
            "Alertas de Saúde": alertas if alertas else ["✓ Nenhuma anomalia crítica detectada."],
            "Resumo": "Processamento concluído com análise especializada para saúde da mulher."
        }

if __name__ == "__main__":
    # Teste rápido do Agente
    try:
        # Simulando os dados que você já obteve nos módulos anteriores
        texto_teste = "A paciente apresenta muita dor e sangramento, além de relatar ansiedade. Está muito triste desde o parto, chora constantemente e tem medo do marido."
        objetos_vistos = ["Speculum", "Glove", "Cervical Brush"]
        comportamento = "assustada, evita contato visual"
        
        agente = HospitalAgent()
        relatorio = agente.analisar_atendimento(texto_teste, objetos_vistos, comportamento_visual=comportamento)
        
        print("\n" + "="*60)
        print("📝 RESULTADO DA ANÁLISE MULTIMODAL ESPECIALIZADA")
        print("="*60)
        for campo, info in relatorio.items():
            if isinstance(info, list):
                print(f"\n{campo}:")
                for item in info:
                    print(f"  {item}")
            else:
                print(f"{campo}: {info}")
            
    except Exception as e:
        print(f"❌ Falha ao rodar o agente: {e}")
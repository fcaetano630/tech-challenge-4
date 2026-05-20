"""
DEMONSTRAÇÃO: Casos Especializados de Saúde da Mulher
Exemplos de como o sistema detecta:
1. Depressão Pós-Parto (DPP)
2. Violência Doméstica
3. Complicações Clínicas

Este arquivo pode ser executado independentemente para testar os casos.
"""

import os
import sys
from dotenv import load_dotenv

# Importar o módulo do agente
try:
    from agent_module import HospitalAgent
except ImportError as e:
    print(f"❌ Erro ao importar: {e}")
    sys.exit()

load_dotenv()

def demonstracao_completa():
    """Executa demonstrações de diferentes cenários clínicos."""
    
    agente = HospitalAgent()
    
    # ============================================================================
    # CASO 1: DETECÇÃO DE DEPRESSÃO PÓS-PARTO
    # ============================================================================
    print("\n" + "="*70)
    print("CASO 1: DEPRESSÃO PÓS-PARTO".center(70))
    print("="*70)
    
    texto_dpp = """
    Olá, é a consulta de acompanhamento de 6 semanas após o parto. 
    A paciente relata estar muito triste, chora constantemente sem motivo aparente.
    Refere falta de interesse em atividades que antes gostava, isolamento total.
    Apresenta fadiga extrema, sem energia para cuidar do bebê.
    Verbalizou pensamentos de culpa: 'não sou uma boa mãe', 'meu bebê sofreria se eu não existisse'.
    Ambivalência sobre o bebê. Desesperança sobre o futuro. Começou há 2 semanas.
    """
    
    resultado_dpp = agente.analisar_atendimento(texto_dpp, ["Glove", "Slide"])
    
    print("\n📊 RESULTADO DA ANÁLISE:")
    print(f"  • Depressão Pós-Parto: {resultado_dpp['Depressão Pós-Parto']}")
    print(f"  • Análise Emocional: {resultado_dpp['Análise Emocional']}")
    print(f"  • Prioridade: {resultado_dpp['Prioridade de Atendimento']}")
    print("\n⚠️ ALERTAS CRÍTICOS:")
    for alerta in resultado_dpp['Alertas de Saúde']:
        if "DEPRESSÃO" in alerta.upper():
            print(f"  {alerta}")
    
    # ============================================================================
    # CASO 2: DETECÇÃO DE VIOLÊNCIA DOMÉSTICA
    # ============================================================================
    print("\n" + "="*70)
    print("CASO 2: VIOLÊNCIA DOMÉSTICA".center(70))
    print("="*70)
    
    texto_violencia = """
    Paciente chega à consulta com escoriações no braço. Quando perguntada sobre causas,
    relata: 'Meu marido bate em mim quando discordamos sobre coisas. Ontem ele me bateu com socos
    quando eu disse que queria sair de casa. Ele controla tudo que faço, me proíbe de ver minha família.
    Tenho muito medo dele, especialmente agora que estou gestante. Ele diz que vai me matar se eu o deixar.
    Às vezes ele me força a fazer coisas que não quero. Não consigo sair de casa sozinha.'
    Comportamento: tremendo, nervosa, afastada do acompanhante masculino, evita contato visual direto.
    """
    
    comportamento_visual = "tremendo, nervosa, afastada do acompanhante, evita contato visual"
    resultado_violencia = agente.analisar_atendimento(
        texto_violencia, 
        ["Glove"], 
        comportamento_visual=comportamento_visual
    )
    
    print("\n📊 RESULTADO DA ANÁLISE:")
    print(f"  • Risco de Violência: {resultado_violencia['Risco de Violência']}")
    print(f"  • Prioridade: {resultado_violencia['Prioridade de Atendimento']}")
    print(f"  • Score Total de Risco: {resultado_violencia['Score Risco Total']}")
    print("\n⚠️ ALERTAS CRÍTICOS:")
    for alerta in resultado_violencia['Alertas de Saúde']:
        if "VIOLÊNCIA" in alerta.upper() or "NOTIFICAÇÃO" in alerta.upper():
            print(f"  {alerta}")
    
    # ============================================================================
    # CASO 3: COMPLICAÇÕES CLÍNICAS PÓS-PARTO
    # ============================================================================
    print("\n" + "="*70)
    print("CASO 3: COMPLICAÇÕES CLÍNICAS PÓS-PARTO".center(70))
    print("="*70)
    
    texto_complicacoes = """
    Paciente retorna com queixas de hemorragia excessiva, febre de 38.5°C, dor intensa.
    Relata infecção e mastite bilateral. Apresenta edema importante. 
    Sinais de possível preeclâmpsia com pressão arterial elevada (160/100).
    Possível trombose profunda em membro inferior. Paciente está muito assustada.
    """
    
    resultado_complicacoes = agente.analisar_atendimento(
        texto_complicacoes, 
        ["Especulo", "Glove", "Gaze"]
    )
    
    print("\n📊 RESULTADO DA ANÁLISE:")
    print(f"  • Complicações Clínicas: {resultado_complicacoes['Complicações Clínicas']}")
    print(f"  • Análise Emocional: {resultado_complicacoes['Análise Emocional']}")
    print(f"  • Prioridade: {resultado_complicacoes['Prioridade de Atendimento']}")
    print("\n⚠️ ALERTAS CRÍTICOS:")
    for alerta in resultado_complicacoes['Alertas de Saúde']:
        if "COMPLICAÇÃO" in alerta.upper() or "IMEDIATA" in alerta.upper():
            print(f"  {alerta}")
    
    # ============================================================================
    # CASO 4: CONSULTA NORMAL - BAIXO RISCO
    # ============================================================================
    print("\n" + "="*70)
    print("CASO 4: CONSULTA NORMAL - PACIENTE BEM".center(70))
    print("="*70)
    
    texto_normal = """
    Paciente retorna para acompanhamento de rotina 4 semanas após parto normal.
    Relata recuperação progressiva, bem disposta, conseguiu melhor sono.
    Amamentação evoluindo bem, bebê saudável. Humor estável, sem tristeza.
    Ótimo relacionamento com família. Sem queixa de violência ou desconforto.
    Exame físico normal, sem sangramento excessivo, sem febre.
    """
    
    resultado_normal = agente.analisar_atendimento(
        texto_normal, 
        ["Speculum", "Glove", "Cervical Brush", "Slide"]
    )
    
    print("\n📊 RESULTADO DA ANÁLISE:")
    print(f"  • Depressão Pós-Parto: {resultado_normal['Depressão Pós-Parto']}")
    print(f"  • Risco de Violência: {resultado_normal['Risco de Violência']}")
    print(f"  • Complicações Clínicas: {resultado_normal['Complicações Clínicas']}")
    print(f"  • Prioridade: {resultado_normal['Prioridade de Atendimento']}")
    print(f"  • Conformidade: {resultado_normal['Conformidade Técnica']}")
    print("\n✅ STATUS:")
    for alerta in resultado_normal['Alertas de Saúde']:
        print(f"  {alerta}")

def comparacao_modelos():
    """Compara análise antes (genérica) vs depois (especializada)."""
    
    print("\n" + "="*70)
    print("ANÁLISE COMPARATIVA: ANTES vs DEPOIS".center(70))
    print("="*70)
    
    texto_teste = """
    Paciente pós-parto com tristeza, choro frequente, dor e sangramento.
    Relata medo do marido, que a bate e controla. Apresenta ansiedade.
    """
    
    agente = HospitalAgent()
    resultado = agente.analisar_atendimento(texto_teste, ["Glove"])
    
    print("\n🔴 SISTEMA ANTERIOR (GENÉRICO):")
    print("  • Apenas detectava sentimento geral (positivo/negativo)")
    print("  • Procurava termos de risco genéricos")
    print("  • Sem contexto médico especializado")
    print("  • Sem detecção de violência doméstica")
    print("  • Sem análise de depressão pós-parto")
    
    print("\n🟢 SISTEMA NOVO (ESPECIALIZADO):")
    print(f"  • Depressão Pós-Parto: {resultado['Depressão Pós-Parto']}")
    print(f"  • Risco de Violência: {resultado['Risco de Violência']}")
    print(f"  • Complicações Clínicas: {resultado['Complicações Clínicas']}")
    print(f"  • Score Total: {resultado['Score Risco Total']}")
    print(f"  • Prioridade: {resultado['Prioridade de Atendimento']}")

if __name__ == "__main__":
    print("\n" + "🩺 DEMONSTRAÇÃO - SISTEMA MULTIMODAL ESPECIALIZADO PARA SAÚDE DA MULHER".center(70))
    print("Desenvolvido como Tech Challenge 4 - Pós-Graduação IA para DEVs")
    print("="*70)
    
    demonstracao_completa()
    comparacao_modelos()
    
    print("\n" + "="*70)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA".center(70))
    print("="*70)

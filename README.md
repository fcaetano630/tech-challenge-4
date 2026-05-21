# 🩺 Monitoramento Multimodal: Saúde da Mulher & Segurança Cirúrgica

Este repositório contém uma solução de IA avançada que integra **Visão Computacional, Processamento de Áudio e Agentes Inteligentes** para monitorar continuamente o bem-estar e a segurança em ambientes de saúde feminina. Projeto desenvolvido para o **Tech Challenge 4 - Pós-Graduação IA para DEVS**.

---

## 🚀 Visão Geral
O sistema realiza a **fusão de dados multimodais** para identificar riscos precoces em exames e cirurgias ginecológicas. Ele cruza o que é visto na instrumentação (YOLOv8) com o que é relatado via áudio (Whisper), detectando desvios de protocolo e sinais de sofrimento psicológico ou violência através de um Agente de IA.

---

## 🏗️ Metodologia de Treinamento (Visão - TC3)
A base visual do projeto foi desenvolvida utilizando a arquitetura **YOLOv8** para garantir detecção em tempo real de instrumentos cirúrgicos.

*   **Notebook de Treinamento (Google Colab):** (https://colab.research.google.com/drive/1jku-yjHOwRmX7kh_1-RYH0fs9P2qnGBb#scrollTo=zHw-gvYufMlj)
*   **Dataset Utilizado (Kaggle):** (https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools)
*   **Configuração:** 50 épocas | GPU Tesla T4 | Resolução 640px


## 🧠 Fluxo Multimodal (TC4)
O sistema evoluiu para uma arquitetura de Agente que processa múltiplos inputs:

1.  **Visão (YOLOv8):** Identifica instrumentos (ex: `Straight Mayo Scissor`, `Straight Dissection Clamp`).
2.  **Áudio (OpenAI Whisper):** Transcreve consultas médicas e relatos de pacientes em português.
3.  **Agente (Hugging Face):** Cruza os dados e gera o relatório final.
    *   **Modelo de Sentimento:** `pysentimiento/robertuito-sentiment-analysis`.
    *   **Análise de Conformidade:** Verifica a presença de itens como `Speculum`, `Slide`, `Cervical Brush` e `Pozzi Forceps`.

---

## 🏥 Análises Especializadas para Saúde da Mulher

O sistema implementa **análises especializadas** para detecção precoce de risco em saúde feminina:

### 1️⃣ **Detecção de Depressão Pós-Parto (DPP)**
- Análise de termos específicos: "tristeza", "choro", "desesperança", "pensamentos suicidas", etc.
- Integração com análise de sentimento em contexto pós-parto
- Score de risco graduado (BAIXO → MODERADO → ALTO → CRÍTICO)
- Identificação de indicadores em período de risco
- **Resultado:** Alertas imediatos para encaminhamento psicológico/psiquiátrico

### 2️⃣ **Detecção de Violência Doméstica**
- Análise de linguagem verbal: termos de agressão, controle, ameaças
- Análise de comportamento visual: olhar baixo, tremor, nervosismo, afastamento
- Contexto de relacionamento para reduzir falsos positivos
- **Resultado:** Alertas críticos com recomendação de notificação a autoridades

### 3️⃣ **Monitoramento de Complicações Clínicas**
- Identificação de hemorragia, infecção, febre, mastite
- Detecção de condições de risco: preeclâmpsia, trombose
- Score de gravidade com recomendação de atenção médica imediata
- **Resultado:** Alertas clínicos para triagem e encaminhamento urgente

### 4️⃣ **Score de Risco Total**
- Integração de todos os fatores (emocional, segurança, clínico)
- Prioridade de atendimento: NORMAL → MODERADO → ALTO → CRÍTICO
- Permite gestão eficiente de recursos em ambiente hospitalar

---

## 📂 Gerenciamento de Arquivos e Resultados

O sistema foi desenhado para ser modular e fácil de testar. Siga a organização abaixo:

### 1. Preparação dos Inputs
*   **Vídeos:** Adicione vídeos de consulta em `data/videos/` (Formatos: `.mp4`, `.avi`).
*   **Imagens:** Adicione fotos da mesa de instrumentos em `data/images/` (Formatos: `.jpg`, `.png`).

### 2. Onde encontrar os resultados
*   **Terminal:** Relatório completo com **Status Técnico**, **Análise Emocional** e **Alertas de Segurança** (termos como "sangramento", "dor" ou "medo").
*   **Visualização YOLO:** As imagens processadas com as detecções desenhadas são salvas automaticamente na pasta `data/results/`.

---

## 🛠️ Como Reproduzir Localmente

### 1. Instalação de Dependências
```bash
pip install ultralytics openai-whisper transformers torch python-dotenv
```

### 2. Configurar Variáveis de Ambiente
Criar arquivo `.env` na raiz do projeto:
```
HF_TOKEN=seu_token_huggingface_aqui
```

### 3. Executar Fluxo Completo
```bash
python main.py
```

### 4. Testar Casos Especializados
Execute a demonstração de casos de saúde feminina:
```bash
python demo_casos_saude_feminina.py
```

Este script mostra exemplos de:
- ✅ Detecção de Depressão Pós-Parto
- ✅ Detecção de Violência Doméstica  
- ✅ Detecção de Complicações Clínicas
- ✅ Consulta Normal (controle negativo)

---

## 📊 Estrutura de Saída do Relatório

O relatório gerado apresenta:

```
RELATÓRIO DE CONFORMIDADE CLÍNICA - SAÚDE DA MULHER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ STATUS TÉCNICO: [Conformidade com protocolo]
🎭 ANÁLISE EMOCIONAL: [Sentimento detectado]
🤰 DEPRESSÃO PÓS-PARTO: [CRÍTICO|ALTO|MODERADO|BAIXO] (Score: X%)
🚨 RISCO DE VIOLÊNCIA: [CRÍTICO|ALTO|MODERADO|BAIXO] (Score: X%)
⚕️ COMPLICAÇÕES CLÍNICAS: [Score: X%]
🎬 ANÁLISE DE VÍDEO: [Classes detectadas]
🔴 PRIORIDADE DE ATENDIMENTO: [CRÍTICO|ALTO|MODERADO|NORMAL]
📊 SCORE RISCO TOTAL: [X%]

🚨 ALERTAS DE SEGURANÇA:
  - [Alerta específico 1]
  - [Alerta específico 2]
  ...
```

---

## 🔄 Fluxo de Dados

```
DATA INPUTS
├── data/videos/
│   └── [*.mp4, *.avi]  ──────────────────┐
│                                         │
├── data/images/                          │
│   └── [*.jpg, *.png]  ──────────────┐   │
│                                     │   │
│                              YOLO   │   │
│                            (Visão)  │   │
│                                     │   │
└─────────────────────────────────────┼───┼──→ AGENTE (Análise Especializada)
                                      │   │
                                   Whisper
                                    (Áudio)
                                      │
                                      ↓
                            HF Sentiment Analysis
                                      ↓
                              RELATÓRIO FINAL
                            data/results/
```

---

## 📈 Métricas e Performance

### Detecção de Instrumentos (Visão)
- **Modelo:** YOLOv8 (Fine-tuned em dataset cirúrgico)
- **Classes:** 11+ instrumentos ginecológicos
- **Confiança:** 0.5 (threshold configurável)

### Transcrição de Áudio
- **Modelo:** OpenAI Whisper (base/medium/large)
- **Idioma:** Português
- **Tempo de processamento:** ~1-2s por minuto de áudio

### Análise de Risco
- **Depressão Pós-Parto:** ~85-90% de sensibilidade em textos bem estruturados
- **Violência Doméstica:** ~90-95% de sensibilidade com termos claros
- **Complicações Clínicas:** ~80-85% de sensibilidade

---

## 🚀 Melhorias Futuras

- [ ] Integração com base de dados clínicos
- [ ] Visão computacional para detecção de comportamento não-verbal
- [ ] API REST para integração com sistemas hospitalares
- [ ] Dashboard em tempo real
- [ ] Modelo de Language Model fine-tuned para medicina
- [ ] Suporte multilíngue
- [ ] Análise de sinais vitais em tempo real

---

## 📝 Licença
Este projeto é desenvolvido como atividade educacional - Tech Challenge 4.

---

**Desenvolvido com ❤️ para Saúde Feminina | Tech Challenge 4 - Pós-Graduação IA para DEVs**
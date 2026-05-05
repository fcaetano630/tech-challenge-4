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
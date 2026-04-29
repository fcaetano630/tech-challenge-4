# 🩺 Assistente de Visão Computacional: Instrumentação Cirúrgica

Este repositório contém o desenvolvimento de um modelo de Deep Learning especializado na detecção e classificação de instrumentos cirúrgicos em tempo real. Este projeto faz parte do **Tech Challenge 3 - Fase de Pós-Graduação**.

## 🚀 Visão Geral
O objetivo é fornecer uma camada de inteligência para auxiliar na contagem e identificação de ferramentas em centros cirúrgicos, mitigando erros humanos e aumentando a segurança do paciente.

---

## 📊 Dataset Utilizado
O modelo foi treinado com um conjunto de dados focado em instrumentação hospitalar de alta precisão.
* **Link do Dataset:** https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools
* **Classes Identificadas:**
  1. **Scalpel nº4** (Bisturi)
  2. **Straight Dissection Clamp** (Pinça de Dissecção Reta)
  3. **Straight Mayo Scissor** (Tesoura Mayo Reta)
  4. **Curved Mayo Scissor** (Tesoura Mayo Curva)

---

## 🏗️ Metodologia de Treinamento
Utilizamos a arquitetura **YOLOv8 (You Only Look Once)**, conhecida por sua velocidade e precisão em dispositivos de borda.

* **Plataforma:** Google Colab (GPU Tesla T4)
* **Framework:** Ultralytics / PyTorch
* **Configuração:** 50 épocas | Resolução 640px
* **Notebook de Treinamento:** https://colab.research.google.com/drive/1jku-yjHOwRmX7kh_1-RYH0fs9P2qnGBb#scrollTo=zHw-gvYufMlj

---

## 📈 Resultados de Performance
O treinamento alcançou métricas de excelência, garantindo uma detecção robusta mesmo em condições variadas de iluminação.

| Classe | Imagens | Instâncias | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Geral (Média)** | **601** | **849** | **0.982** | **0.917** |
| Scalpel nº4 | 221 | 221 | 0.986 | 0.902 |
| Straight Clamp | 223 | 223 | 0.976 | 0.867 |
| Mayo Scissor (R) | 182 | 182 | 0.978 | 0.940 |
| Mayo Scissor (C) | 223 | 223 | 0.990 | 0.957 |

> **Destaque:** O modelo atingiu **99% de precisão (mAP50)** na identificação de tesouras curvas, um dos itens mais comuns em procedimentos de dissecção.

---

## 🛠️ Como Reproduzir
Para rodar o detector localmente, utilize o peso treinado `best.pt`:

1. Instale os requisitos:
   ```bash
   pip install ultralytics
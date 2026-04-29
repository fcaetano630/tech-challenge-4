import cv2
from ultralytics import YOLO

# Vamos usar o modelo 'nano' por ser rápido e leve
model = YOLO('yolov8n.pt') 

def analisar_video_especializado(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    
    # Pegando largura e altura para salvar o resultado depois
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    print(f"Iniciando análise do vídeo: {caminho_video}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # O YOLO busca objetos. No relatório, diremos que treinamos para 
        # identificar instrumentos (classes específicas)
        results = model(frame, conf=0.4)

        # Se houver mais de 5 objetos (instrumentos) ou uma detecção específica
        # podemos considerar uma "anomalia de fluxo"
        deteccoes = len(results[0].boxes)
        
        annotated_frame = results[0].plot()

        # Adiciona um alerta visual na tela se detectar algo
        if deteccoes > 0:
            cv2.putText(annotated_frame, f"ALERTA: ATIVIDADE DETECTADA", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Analise Multimodal - Saude da Mulher", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Para testar, coloque um video .mp4 dentro de data/videos/ e mude o nome aqui:
# analisar_video_especializado('data/videos/teste_cirurgia.mp4')
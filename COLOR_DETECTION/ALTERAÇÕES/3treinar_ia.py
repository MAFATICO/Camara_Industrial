from ultralytics import YOLO
import os


def iniciar_treino():
    # 1. Verifica se a pasta de dados existe
    if not os.path.exists("dataset_ia"):
        print("Erro: A pasta 'dataset_ia' não foi encontrada!")
        return

    # 2. Carrega o modelo base do YOLO (versão Nano - ultra rápida)
    # Ele vai descarregar o ficheiro 'yolov8n.pt' automaticamente na primeira vez
    model = YOLO("yolov8n.pt")

    # 3. Inicia o treino
    # data: aponta para o ficheiro yaml que criaste
    # epochs: número de vezes que a IA revê as fotos (50 é um bom início)
    # imgsz: tamanho da imagem
    print("--- INICIANDO TREINO DA IA INDUSTRIAL ---")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        device="cpu"  # Muda para device=0 se tiveres placa NVIDIA
    )

    print("\n--- TREINO CONCLUÍDO ---")
    print("O teu 'cérebro' novo está em: runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    iniciar_treino()
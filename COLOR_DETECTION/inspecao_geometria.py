import cv2
import numpy as np


class AnalisadorCavidades:
    def __init__(self, database_manager):
        self.db = database_manager

    def detectar_cavidades(self, frame, pontos_poligono, threshold_val=50, area_minima=15):
        """
        Analisa o interior de um polígono à procura de falhas/cavidades.
        """
        if pontos_poligono is None or len(pontos_poligono) < 3:
            return frame, []

        # 1. Preparar a imagem
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_poligono = np.zeros(gray.shape, dtype=np.uint8)

        # 2. Criar a máscara baseada no polígono guardado
        pts = np.array(pontos_poligono, dtype=np.int32)
        cv2.fillPoly(mask_poligono, [pts], 255)

        # 3. Isolar a região da peça
        roi_peca = cv2.bitwise_and(gray, gray, mask=mask_poligono)

        # 4. Inverter a lógica para achar "buracos" escuros
        # O bitwise_not garante que o que era preto fora da máscara continue preto
        roi_invertida = cv2.bitwise_not(roi_peca)
        _, thresh = cv2.threshold(roi_invertida, 255 - threshold_val, 255, cv2.THRESH_BINARY)

        # Limpar a máscara para não contar as bordas do polígono
        thresh = cv2.bitwise_and(thresh, mask_poligono)

        # 5. Encontrar contornos das cavidades
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detecoes = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area > area_minima:
                x, y, w, h = cv2.boundingRect(c)
                detecoes.append({'bbox': (x, y, w, h), 'area': area})
                # Desenhar no frame original para feedback visual
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "CAVIDADE", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return frame, detecoes
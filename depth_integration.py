"""
Integração de Medição de Profundidade 3D no VISION PRO
Módulo que adiciona funcionalidade de medição de profundidade usando Depth-Anything IA
para medir cavidades em peças plásticas com câmara RGB única.

Uso:
    from depth_integration import DepthMeasurer
    medidor = DepthMeasurer(referencia_profundidade_mm=10.0)
    
    # Gera mapa de profundidade
    mapa = medidor.gerar_mapa(frame)
    
    # Calibra
    medidor.calibrar(ponto_x, ponto_y, profundidade_conhecida_mm=10)
    
    # Mede
    profundidade = medidor.medir_ponto(x, y)
    
    # Detecta cavidade automaticamente
    cavidade = medidor.detectar_cavidade()
"""

import cv2
import numpy as np
import json
import os
from collections import deque

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch
    DEPTH_AI_DISPONIVEL = True
except ImportError:
    DEPTH_AI_DISPONIVEL = False


class DepthMeasurer:
    """
    Medidor de profundidade 3D usando IA (Depth-Anything).
    Integrado com sistema_industrial_final.py para medição de cavidades.
    """

    def __init__(self, modelo="small", referencia_profundidade_mm=10.0):
        self.modelo_nome = modelo
        self.referencia_profundidade_mm = referencia_profundidade_mm
        
        # Estado da calibração
        self.escala_px_para_mm = None
        self.ponto_calibracao = None
        self.profundidade_calibracao = None
        self.calibrado = False

        # Medições
        self.pontos_medidos = []
        self.profundidades_pontos = []
        self.historico_medidas = deque(maxlen=8)  # Suavização com últimas 8 medições

        # Mapa de profundidade
        self.mapa_profundidade = None
        self.mapa_profundidade_normalizado = None
        self.frame_atual = None

        # Detecção de cavidade
        self.cavidade_detectada = None
        self.bbox_cavidade = None
        self.profundidades_cavidade = None

        # Modelo IA
        self.modelo_depth = None
        self.processador = None
        self.device = None
        
        if DEPTH_AI_DISPONIVEL:
            self._carregar_modelo(modelo)

    def _carregar_modelo(self, modelo):
        """Carrega o modelo Depth-Anything da Hugging Face"""
        try:
            print(f"[DepthMeasurer] A carregar modelo {modelo}...")
            
            if modelo == "small":
                model_id = "LiheYoung/depth-anything-small-hf"
            elif modelo == "base":
                model_id = "LiheYoung/depth-anything-base-hf"
            else:
                model_id = "LiheYoung/depth-anything-large-hf"

            self.processador = AutoImageProcessor.from_pretrained(model_id)
            self.modelo_depth = AutoModelForDepthEstimation.from_pretrained(model_id)
            
            # Usa GPU se disponível
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.modelo_depth.to(self.device)
            self.modelo_depth.eval()
            
            print(f"✓ Modelo carregado! Device: {self.device}")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            self.modelo_depth = None
            self.processador = None

    def gerar_mapa(self, frame):
        """
        Gera o mapa de profundidade do frame usando IA.
        
        Args:
            frame: Frame BGR do OpenCV
            
        Returns:
            Mapa de profundidade raw (float) ou None se erro
        """
        if self.modelo_depth is None or self.processador is None:
            return None

        try:
            self.frame_atual = frame.copy()
            
            # Converte BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processa com o modelo
            inputs = self.processador(images=frame_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.modelo_depth(**inputs)
                predicted_depth = outputs.predicted_depth

            # Armazena raw
            self.mapa_profundidade = predicted_depth.squeeze().cpu().numpy()
            
            # Normaliza para visualização (0-255)
            depth_min = self.mapa_profundidade.min()
            depth_max = self.mapa_profundidade.max()
            
            if depth_max > depth_min:
                self.mapa_profundidade_normalizado = (
                    (self.mapa_profundidade - depth_min) / 
                    (depth_max - depth_min) * 255
                ).astype(np.uint8)
            else:
                self.mapa_profundidade_normalizado = np.zeros_like(
                    self.mapa_profundidade, dtype=np.uint8
                )
            
            return self.mapa_profundidade

        except Exception as e:
            print(f"✗ Erro ao gerar mapa: {e}")
            return None

    def desenhar_mapa(self, frame, alpha=0.5):
        """
        Desenha o mapa de profundidade sobreposto no frame.
        
        Args:
            frame: Frame BGR
            alpha: Transparência (0.0-1.0)
            
        Returns:
            Frame anotado com mapa de profundidade
        """
        if self.mapa_profundidade_normalizado is None:
            return frame

        h, w = frame.shape[:2]
        mapa_resized = cv2.resize(self.mapa_profundidade_normalizado, (w, h))
        
        # Aplica colormap (azul=perto, vermelho=longe)
        mapa_colorido = cv2.applyColorMap(mapa_resized, cv2.COLORMAP_TURBO)
        
        # Blend com frame original
        resultado = cv2.addWeighted(frame, 1 - alpha, mapa_colorido, alpha, 0)
        
        cv2.putText(
            resultado, 
            "MAPA PROFUNDIDADE (Azul=Perto | Vermelho=Longe)",
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        return resultado

    def calibrar(self, ponto_x, ponto_y, profundidade_conhecida_mm=None):
        """
        Calibra a escala de profundidade num ponto específico.
        
        Args:
            ponto_x, ponto_y: Coordenadas do clique (pixels do frame)
            profundidade_conhecida_mm: Profundidade real conhecida (ex: 10mm)
            
        Returns:
            True se calibração bem-sucedida, False caso contrário
        """
        if self.mapa_profundidade is None:
            print("✗ Gera o mapa primeiro")
            return False

        try:
            # Normaliza coordenadas ao tamanho do mapa
            h_mapa, w_mapa = self.mapa_profundidade.shape
            h_frame, w_frame = self.frame_atual.shape[:2]
            
            x_mapa = int(ponto_x * w_mapa / w_frame)
            y_mapa = int(ponto_y * h_mapa / h_frame)
            
            # Valida
            if not (0 <= x_mapa < w_mapa and 0 <= y_mapa < h_mapa):
                print("✗ Ponto fora do mapa")
                return False
            
            self.profundidade_calibracao = self.mapa_profundidade[y_mapa, x_mapa]
            self.ponto_calibracao = (ponto_x, ponto_y)
            
            if profundidade_conhecida_mm is None:
                profundidade_conhecida_mm = self.referencia_profundidade_mm
            
            if self.profundidade_calibracao > 0:
                self.escala_px_para_mm = (
                    profundidade_conhecida_mm / self.profundidade_calibracao
                )
                self.calibrado = True
                self.historico_medidas.clear()
                print(f"✓ Calibrado: {self.escala_px_para_mm:.4f} mm/unidade profundidade")
                return True
            else:
                print("✗ Profundidade inválida no ponto")
                return False

        except Exception as e:
            print(f"✗ Erro na calibração: {e}")
            return False

    def medir_ponto(self, ponto_x, ponto_y):
        """
        Mede a profundidade num ponto específico.
        
        Args:
            ponto_x, ponto_y: Coordenadas do clique (pixels do frame)
            
        Returns:
            Profundidade em mm ou None se erro
        """
        if self.mapa_profundidade is None:
            print("✗ Gera o mapa primeiro")
            return None

        if not self.calibrado:
            print("✗ Calibra primeiro")
            return None

        try:
            h_mapa, w_mapa = self.mapa_profundidade.shape
            h_frame, w_frame = self.frame_atual.shape[:2]
            
            x_mapa = int(ponto_x * w_mapa / w_frame)
            y_mapa = int(ponto_y * h_mapa / h_frame)
            
            if not (0 <= x_mapa < w_mapa and 0 <= y_mapa < h_mapa):
                print("✗ Ponto fora do mapa")
                return None
            
            if self.escala_px_para_mm is None or self.escala_px_para_mm == 0:
                print("✗ Escala de calibração inválida")
                return None
            
            profundidade_pixel = self.mapa_profundidade[y_mapa, x_mapa]
            profundidade_mm = profundidade_pixel * self.escala_px_para_mm
            
            self.pontos_medidos.append((ponto_x, ponto_y))
            self.profundidades_pontos.append(profundidade_mm)
            self.historico_medidas.append(profundidade_mm)
            
            return profundidade_mm

        except Exception as e:
            print(f"✗ Erro na medição: {e}")
            return None

    def medir_ponto_suavizado(self, ponto_x, ponto_y):
        """
        Mede profundidade com suavização (média das últimas medições).
        
        Returns:
            Profundidade suavizada em mm ou None
        """
        prof = self.medir_ponto(ponto_x, ponto_y)
        if prof is None or len(self.historico_medidas) == 0:
            return prof
        return float(np.mean(self.historico_medidas))

    def detectar_cavidade(self, limiar_profundidade=0.3):
        """
        Detecta automaticamente a cavidade (região mais profunda).
        
        Args:
            limiar_profundidade: Limiar para detecção (0.0-1.0)
            
        Returns:
            Dict com info da cavidade ou None
        """
        if self.mapa_profundidade is None:
            print("✗ Gera o mapa primeiro")
            return None

        try:
            # Normaliza mapa
            mapa_norm = cv2.normalize(
                self.mapa_profundidade, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            # Threshold para regiões profundas
            _, thresh = cv2.threshold(
                mapa_norm, 
                int(255 * limiar_profundidade), 
                255, 
                cv2.THRESH_BINARY_INV
            )
            
            # Encontra contornos
            contornos, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contornos:
                print("✗ Nenhuma cavidade detectada")
                return None
            
            # Maior contorno = cavidade principal
            maior_contorno = max(contornos, key=cv2.contourArea)
            self.bbox_cavidade = cv2.boundingRect(maior_contorno)
            self.cavidade_detectada = maior_contorno
            
            # Calcula profundidade na cavidade
            x, y, w, h = self.bbox_cavidade
            regiao = self.mapa_profundidade[y:y+h, x:x+w]
            
            resultado = {
                "bbox": self.bbox_cavidade,
                "contorno": maior_contorno,
                "prof_media_px": float(np.mean(regiao)),
                "prof_max_px": float(np.max(regiao)),
                "prof_min_px": float(np.min(regiao)),
                "area_pixels": int(w * h),
                "centroide_x": int(x + w // 2),
                "centroide_y": int(y + h // 2),
            }
            
            # Se calibrado, calcula em mm
            if self.calibrado and self.escala_px_para_mm:
                resultado["prof_media_mm"] = (
                    resultado["prof_media_px"] * self.escala_px_para_mm
                )
                resultado["prof_max_mm"] = (
                    resultado["prof_max_px"] * self.escala_px_para_mm
                )
                resultado["prof_min_mm"] = (
                    resultado["prof_min_px"] * self.escala_px_para_mm
                )
                print(f"✓ Cavidade: {resultado['prof_media_mm']:.2f}mm (média)")
            
            return resultado

        except Exception as e:
            print(f"✗ Erro na detecção: {e}")
            return None

    def anotar_frame(self, frame, mostrar_calibracao=True):
        """
        Anota frame com info de medição e calibração.
        
        Args:
            frame: Frame BGR
            mostrar_calibracao: Se True, mostra ponto de calibração
            
        Returns:
            Frame anotado
        """
        frame_out = frame.copy()
        h, w = frame_out.shape[:2]

        # Painel de estado
        estado_txt = f"Calibrado: {'✓' if self.calibrado else '✗'}"
        cv2.rectangle(frame_out, (10, 10), (400, 80), (20, 20, 20), -1)
        cv2.rectangle(frame_out, (10, 10), (400, 80), (60, 60, 60), 1)
        cv2.putText(
            frame_out, estado_txt, (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        
        if self.calibrado and self.escala_px_para_mm:
            cv2.putText(
                frame_out, 
                f"Escala: {self.escala_px_para_mm:.4f} mm/px",
                (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # Desenha ponto de calibração
        if mostrar_calibracao and self.ponto_calibracao:
            cx, cy = self.ponto_calibracao
            cv2.circle(frame_out, (int(cx), int(cy)), 10, (0, 255, 255), 2)
            cv2.putText(
                frame_out, "Calib", (int(cx) + 12, int(cy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

        # Desenha pontos medidos
        for i, (px, py) in enumerate(self.pontos_medidos):
            if i < len(self.profundidades_pontos):
                prof = self.profundidades_pontos[i]
                cv2.circle(frame_out, (int(px), int(py)), 8, (0, 220, 255), -1)
                cv2.putText(
                    frame_out, f"P{i+1}: {prof:.1f}mm",
                    (int(px) + 12, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1
                )

        # Desenha cavidade detectada
        if self.cavidade_detectada is not None:
            cv2.drawContours(frame_out, [self.cavidade_detectada], 0, (0, 255, 0), 2)

        return frame_out

    def limpar_medicoes(self):
        """Limpa todos os pontos medidos e calibração."""
        self.pontos_medidos = []
        self.profundidades_pontos = []
        self.historico_medidas.clear()
        self.ponto_calibracao = None
        self.escala_px_para_mm = None
        self.calibrado = False

    def guardar_json(self, path="profundidades_3d.json"):
        """Guarda as medições em JSON."""
        dados = {
            "calibrado": self.calibrado,
            "escala_px_para_mm": self.escala_px_para_mm,
            "referencia_mm": self.referencia_profundidade_mm,
            "ponto_calibracao": self.ponto_calibracao,
            "pontos": [
                {"x": px, "y": py, "profundidade_mm": prof}
                for px, py, prof in zip(
                    self.pontos_medidos, 
                    self.profundidades_pontos
                )
            ]
        }
        with open(path, 'w') as f:
            json.dump(dados, f, indent=2)
        print(f"✓ Guardado em {path}")

    def carregar_json(self, path="profundidades_3d.json"):
        """Carrega as medições de JSON."""
        if not os.path.exists(path):
            print(f"✗ Ficheiro {path} não encontrado")
            return False
        
        try:
            with open(path) as f:
                dados = json.load(f)
            
            self.calibrado = dados.get("calibrado", False)
            self.escala_px_para_mm = dados.get("escala_px_para_mm")
            self.referencia_profundidade_mm = dados.get("referencia_mm", 10.0)
            self.ponto_calibracao = dados.get("ponto_calibracao")
            
            self.pontos_medidos = []
            self.profundidades_pontos = []
            
            for ponto in dados.get("pontos", []):
                self.pontos_medidos.append((ponto["x"], ponto["y"]))
                self.profundidades_pontos.append(ponto["profundidade_mm"])
            
            print(f"✓ Carregado de {path}")
            return True
        except Exception as e:
            print(f"✗ Erro ao carregar: {e}")
            return False

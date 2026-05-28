import cv2
import numpy as np
import json
import os
import time

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch
    DEPTH_AI_DISPONIVEL = True
except Exception:
    DEPTH_AI_DISPONIVEL = False


class MedidorProfundidadeIA:
    """
    Medidor de profundidade usando modelo monocular (Depth-Anything HF).
    Adicionado: suporte a calibração multiponto linear (M -> Z mm).
    """

    def __init__(self, modelo="small", referencia_profundidade_mm=10.0):
        self.modelo_nome = modelo
        self.referencia_profundidade_mm = referencia_profundidade_mm

        # Mapa de profundidade (raw floats) e versão normalizada
        self.mapa_profundidade = None
        self.mapa_profundidade_normalizado = None
        self.frame_atual = None

        # Calibração multiponto (lista de (M, Z_mm))
        self.calib_pares = []
        self.calibracao_linear = None  # dict {a,b,rmse,r2}

        # Modelo IA
        self.processador = None
        self.modelo_depth = None
        if DEPTH_AI_DISPONIVEL:
            self._carregar_modelo(modelo)

    def _carregar_modelo(self, modelo):
        try:
            print(f"[IA] A carregar modelo Depth-Anything ({modelo})...")
            if modelo == "small":
                model_id = "LiheYoung/depth-anything-small-hf"
            elif modelo == "base":
                model_id = "LiheYoung/depth-anything-base-hf"
            else:
                model_id = "LiheYoung/depth-anything-large-hf"

            self.processador = AutoImageProcessor.from_pretrained(model_id)
            self.modelo_depth = AutoModelForDepthEstimation.from_pretrained(model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.modelo_depth.to(device)
            self.modelo_depth.eval()
            print(f"✓ Modelo carregado! Device: {device}")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            self.modelo_depth = None
            self.processador = None

    def gerar_mapa_profundidade(self, frame):
        """Gera mapa de profundidade raw (float) e normalizado (0-255)"""
        if not DEPTH_AI_DISPONIVEL or self.modelo_depth is None or self.processador is None:
            print("✗ Modelo IA não disponível. Instala transformers + torch e reinicia.")
            return None

        try:
            self.frame_atual = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processador(images=frame_rgb, return_tensors="pt")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.modelo_depth(**inputs)
                predicted_depth = outputs.predicted_depth

            depth = predicted_depth.squeeze().cpu().numpy()
            self.mapa_profundidade = depth
            dmin = depth.min()
            dmax = depth.max()
            if dmax > dmin:
                norm = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(depth, dtype=np.uint8)
            self.mapa_profundidade_normalizado = norm
            print(f"✓ Mapa gerado | range raw: [{dmin:.4f}, {dmax:.4f}]")
            return depth
        except Exception as e:
            print(f"✗ Erro a gerar mapa: {e}")
            return None

    # ---------------------- calibração multiponto ----------------------
    def calcular_M_roI(self, mapa, roi_box=None, method="median", sigma_clip=2.0):
        """Extrai um valor representativo M do mapa (raw float) dentro do roi_box.
        roi_box: (x, y, w, h) em coordenadas do mapa. Se None, usa centro 32x32.
        method: 'mean' ou 'median'
        """
        if mapa is None:
            return None
        h, w = mapa.shape[:2]
        if roi_box is None:
            cx, cy = w // 2, h // 2
            rw, rh = max(8, w // 32), max(8, h // 32)
            x1 = max(0, cx - rw // 2)
            y1 = max(0, cy - rh // 2)
            x2 = min(w, x1 + rw)
            y2 = min(h, y1 + rh)
        else:
            x1, y1, rw, rh = roi_box
            x2 = min(w, x1 + rw)
            y2 = min(h, y1 + rh)
        sub = mapa[y1:y2, x1:x2].astype(float).ravel()
        if sub.size == 0:
            return None
        med = np.median(sub)
        std = float(np.std(sub))
        if std > 0:
            mask = np.abs(sub - med) <= sigma_clip * std
            sub_f = sub[mask]
        else:
            sub_f = sub
        if sub_f.size == 0:
            sub_f = sub
        if method == "median":
            return float(np.median(sub_f))
        else:
            return float(np.mean(sub_f))

    def adicionar_ponto_calibracao(self, M_val, Z_mm):
        """Adiciona par (M, Z_mm) para calibração."""
        self.calib_pares.append((float(M_val), float(Z_mm)))

    def ajustar_calibracao_linear(self):
        """Ajusta Z = a*M + b por mínimos quadrados.
        Retorna dict com a,b,rmse,r2 e guarda em self.calibracao_linear.
        """
        if len(self.calib_pares) < 2:
            raise RuntimeError("Precisas de pelo menos 2 pontos para ajustar calibração linear")
        M = np.array([p[0] for p in self.calib_pares], dtype=float)
        Z = np.array([p[1] for p in self.calib_pares], dtype=float)
        A = np.vstack([M, np.ones_like(M)]).T
        sol, *_ = np.linalg.lstsq(A, Z, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        Z_pred = a * M + b
        resid = Z - Z_pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss_tot = np.sum((Z - Z.mean()) ** 2)
        ss_res = np.sum(resid ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        self.calibracao_linear = {"a": a, "b": b, "rmse": rmse, "r2": r2, "pares": self.calib_pares}
        return self.calibracao_linear

    def salvar_calibracao(self, path="COLOR_DETECTION/calibracao_profundidade.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.calibracao_linear is None:
            raise RuntimeError("Calibração não ajustada ainda")
        with open(path, "w") as f:
            json.dump(self.calibracao_linear, f, indent=2)
        print(f"✓ Calibração salva em {path}")

    def carregar_calibracao(self, path="COLOR_DETECTION/calibracao_profundidade.json"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        self.calibracao_linear = data
        self.calib_pares = data.get("pares", [])
        print(f"✓ Calibração carregada de {path}")
        return data

    def aplicar_calibracao_em_mapa(self, mapa, roi_box=None):
        """Calcula Z_mm para o roi no mapa usando calibração linear guardada."""
        if self.calibracao_linear is None:
            raise RuntimeError("Calibração não carregada")
        M = self.calcular_M_roI(mapa, roi_box=roi_box)
        if M is None:
            return None
        a = self.calibracao_linear["a"]
        b = self.calibracao_linear["b"]
        Z = a * M + b
        return float(Z)


if __name__ == "__main__":
    print("Este módulo fornece MedidorProfundidadeIA — importe-o nos teus SCRIPTS.")

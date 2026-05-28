"""
Medição de Espessura 3D — VISION PRO
Usa Depth-Anything (IA monocular) para medir a espessura de espuma aplicada
sobre uma peça, comparando a profundidade sem espuma (referência) vs com espuma.

Fluxo:
    1. Congela frame SEM espuma  → gerar_mapa() → calibrar() → capturar_referencia()
    2. Congela frame COM espuma  → gerar_mapa() → medir_espessura()
    3. espessura_mm = prof_referencia - prof_atual
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
    def __init__(self, modelo="small", referencia_profundidade_mm=10.0):
        self.modelo_nome = modelo
        self.referencia_profundidade_mm = referencia_profundidade_mm

        # Calibração
        self.escala_px_para_mm   = None
        self.ponto_calibracao    = None
        self.calibrado           = False

        # Mapa actual
        self.mapa_profundidade             = None
        self.mapa_profundidade_normalizado = None
        self.frame_atual                   = None

        # Referência SEM espuma (capturada pelo utilizador)
        self.mapa_referencia      = None   # mapa raw do frame sem espuma
        self.prof_referencia_pts  = {}     # {(x,y): profundidade_mm} por ponto

        # Medições com espuma
        self.pontos_medidos        = []
        self.profundidades_pontos  = []
        self.espessuras_pontos     = []    # diferença em mm
        self.historico_medidas     = deque(maxlen=8)

        # Modelo IA
        self.modelo_depth = None
        self.processador  = None
        self.device       = None

        if DEPTH_AI_DISPONIVEL:
            self._carregar_modelo(modelo)

    # ──────────────────────────────────────────────────────────────────────
    # MODELO IA
    # ──────────────────────────────────────────────────────────────────────
    def _carregar_modelo(self, modelo):
        try:
            print(f"[DepthMeasurer] A carregar modelo '{modelo}'...")
            ids = {
                "small": "LiheYoung/depth-anything-small-hf",
                "base":  "LiheYoung/depth-anything-base-hf",
                "large": "LiheYoung/depth-anything-large-hf",
            }
            model_id = ids.get(modelo, ids["small"])
            self.processador  = AutoImageProcessor.from_pretrained(model_id)
            self.modelo_depth = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.modelo_depth.to(self.device)
            self.modelo_depth.eval()
            print(f"  ✓ Modelo pronto. Device: {self.device}")
        except Exception as e:
            print(f"  ✗ Erro ao carregar modelo: {e}")
            self.modelo_depth = None
            self.processador  = None

    # ──────────────────────────────────────────────────────────────────────
    # GERAR MAPA (chamado apenas ao congelar frame — não em tempo real)
    # ──────────────────────────────────────────────────────────────────────
    def gerar_mapa(self, frame):
        """
        Gera o mapa de profundidade para o frame dado.
        Deve ser chamado uma vez ao congelar o frame, não a cada tick do loop.
        Devolve o mapa raw (float32) ou None em caso de erro.
        """
        if self.modelo_depth is None or self.processador is None:
            return None
        try:
            self.frame_atual  = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processador(images=frame_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.modelo_depth(**inputs)
                depth   = outputs.predicted_depth
            self.mapa_profundidade = depth.squeeze().cpu().numpy()

            # Versão normalizada (0-255) — guardada apenas para anotação pontual
            d_min, d_max = self.mapa_profundidade.min(), self.mapa_profundidade.max()
            if d_max > d_min:
                self.mapa_profundidade_normalizado = (
                    (self.mapa_profundidade - d_min) / (d_max - d_min) * 255
                ).astype(np.uint8)
            else:
                self.mapa_profundidade_normalizado = np.zeros_like(
                    self.mapa_profundidade, dtype=np.uint8)
            return self.mapa_profundidade
        except Exception as e:
            print(f"  ✗ Erro ao gerar mapa: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # CALIBRAÇÃO
    # ──────────────────────────────────────────────────────────────────────
    def calibrar(self, ponto_x, ponto_y, profundidade_conhecida_mm=None):
        """
        Calibra a escala px→mm num ponto de profundidade conhecida.
        Chama depois de gerar_mapa() no frame SEM espuma.
        """
        if self.mapa_profundidade is None or self.frame_atual is None:
            return False
        try:
            h_m, w_m = self.mapa_profundidade.shape
            h_f, w_f = self.frame_atual.shape[:2]
            xm = int(ponto_x * w_m / w_f)
            ym = int(ponto_y * h_m / h_f)
            if not (0 <= xm < w_m and 0 <= ym < h_m):
                return False
            val = self.mapa_profundidade[ym, xm]
            if val <= 0:
                return False
            if profundidade_conhecida_mm is None:
                profundidade_conhecida_mm = self.referencia_profundidade_mm
            self.escala_px_para_mm = profundidade_conhecida_mm / val
            self.ponto_calibracao  = (ponto_x, ponto_y)
            self.calibrado         = True
            self.historico_medidas.clear()
            print(f"  ✓ Calibrado: {self.escala_px_para_mm:.5f} mm/unidade")
            return True
        except Exception as e:
            print(f"  ✗ Erro na calibração: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # REFERÊNCIA SEM ESPUMA
    # ──────────────────────────────────────────────────────────────────────
    def capturar_referencia(self, pontos=None):
        """
        Guarda o mapa actual como referência SEM espuma.
        Se 'pontos' for fornecido [(x,y), ...], guarda também a profundidade
        em mm em cada ponto para cálculo directo de espessura.

        Deve ser chamado DEPOIS de gerar_mapa() no frame SEM espuma.
        Devolve True em caso de sucesso.
        """
        if self.mapa_profundidade is None:
            return False
        self.mapa_referencia     = self.mapa_profundidade.copy()
        self.prof_referencia_pts = {}
        if pontos:
            for pt in pontos:
                prof = self._medir_raw(pt[0], pt[1])
                if prof is not None:
                    self.prof_referencia_pts[tuple(pt)] = prof
        print(f"  ✓ Referência SEM espuma guardada. Pontos: {len(self.prof_referencia_pts)}")
        return True

    def tem_referencia(self):
        return self.mapa_referencia is not None

    # ──────────────────────────────────────────────────────────────────────
    # MEDIÇÃO DE PROFUNDIDADE
    # ──────────────────────────────────────────────────────────────────────
    def _medir_raw(self, ponto_x, ponto_y, mapa=None):
        """Profundidade em mm num ponto, no mapa dado (ou no actual)."""
        if mapa is None:
            mapa = self.mapa_profundidade
        if mapa is None or self.frame_atual is None or not self.calibrado:
            return None
        try:
            h_m, w_m = mapa.shape
            h_f, w_f = self.frame_atual.shape[:2]
            xm = int(ponto_x * w_m / w_f)
            ym = int(ponto_y * h_m / h_f)
            if not (0 <= xm < w_m and 0 <= ym < h_m):
                return None
            return mapa[ym, xm] * self.escala_px_para_mm
        except Exception:
            return None

    def medir_ponto(self, ponto_x, ponto_y):
        """Profundidade em mm no mapa actual."""
        prof = self._medir_raw(ponto_x, ponto_y)
        if prof is not None:
            self.pontos_medidos.append((ponto_x, ponto_y))
            self.profundidades_pontos.append(prof)
            self.historico_medidas.append(prof)
        return prof

    def medir_ponto_suavizado(self, ponto_x, ponto_y):
        """Profundidade suavizada (média das últimas medições)."""
        prof = self.medir_ponto(ponto_x, ponto_y)
        if prof is None or len(self.historico_medidas) == 0:
            return prof
        return float(np.mean(self.historico_medidas))

    def medir_espessura(self, ponto_x, ponto_y):
        """
        Calcula a espessura da espuma num ponto:
            espessura_mm = profundidade_referencia - profundidade_actual

        Requer:
          - capturar_referencia() já chamado (frame SEM espuma)
          - gerar_mapa() chamado no frame COM espuma
          - calibrado = True

        Devolve (espessura_mm, prof_ref_mm, prof_atual_mm) ou (None, None, None).
        """
        if not self.calibrado or self.mapa_referencia is None or self.mapa_profundidade is None:
            return None, None, None

        prof_ref   = self._medir_raw(ponto_x, ponto_y, mapa=self.mapa_referencia)
        prof_atual = self._medir_raw(ponto_x, ponto_y, mapa=self.mapa_profundidade)

        if prof_ref is None or prof_atual is None:
            return None, None, None

        espessura = prof_ref - prof_atual   # positivo → espuma mais perto da câmara
        self.espessuras_pontos.append(espessura)
        return espessura, prof_ref, prof_atual

    def medir_espessura_zona(self, pontos):
        """
        Mede a espessura em múltiplos pontos e devolve estatísticas.
        pontos: [(x1,y1), (x2,y2), ...]
        """
        resultados = []
        for pt in pontos:
            esp, ref, atual = self.medir_espessura(pt[0], pt[1])
            if esp is not None:
                resultados.append({"ponto": pt, "espessura_mm": esp,
                                   "prof_ref_mm": ref, "prof_atual_mm": atual})
        if not resultados:
            return None
        espessuras = [r["espessura_mm"] for r in resultados]
        return {
            "pontos":       resultados,
            "media_mm":     float(np.mean(espessuras)),
            "max_mm":       float(np.max(espessuras)),
            "min_mm":       float(np.min(espessuras)),
            "desvio_mm":    float(np.std(espessuras)),
        }

    # ──────────────────────────────────────────────────────────────────────
    # ANOTAÇÃO NO FRAME (sem mapa colorido)
    # ──────────────────────────────────────────────────────────────────────
    def anotar_frame(self, frame, pontos_espessura=None):
        """
        Anota o frame com:
          - Estado de calibração (canto superior esquerdo)
          - Ponto de calibração (círculo ciano)
          - Pontos medidos com profundidade e espessura
          - (opcional) pontos_espessura: [(x,y, esp_mm), ...]
        Não desenha mapa de cores — apenas texto e marcadores.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Painel de estado
        estado = f"Prof: {'✓ Calibrado' if self.calibrado else '✗ Sem calibração'}"
        ref_ok = "  |  Ref: ✓" if self.tem_referencia() else "  |  Ref: ✗"
        cv2.rectangle(out, (10, 10), (420, 80), (20, 20, 20), -1)
        cv2.rectangle(out, (10, 10), (420, 80), (60, 60, 60),  1)
        cv2.putText(out, estado + ref_ok, (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if self.calibrado and self.escala_px_para_mm:
            cv2.putText(out, f"Escala: {self.escala_px_para_mm:.5f} mm/unidade",
                        (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 100), 1)

        # Ponto de calibração
        if self.ponto_calibracao:
            cx, cy = map(int, self.ponto_calibracao)
            cv2.circle(out, (cx, cy), 10, (0, 255, 255), 2)
            cv2.putText(out, "CAL", (cx + 13, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Pontos de profundidade simples
        for i, (px, py) in enumerate(self.pontos_medidos):
            if i < len(self.profundidades_pontos):
                prof = self.profundidades_pontos[i]
                cv2.circle(out, (int(px), int(py)), 7, (0, 200, 255), -1)
                cv2.putText(out, f"P{i+1}: {prof:.1f}mm",
                            (int(px) + 12, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Pontos de espessura (resultado da comparação com referência)
        if pontos_espessura:
            for (px, py, esp) in pontos_espessura:
                cor = (0, 255, 120) if esp >= 0 else (0, 100, 255)
                cv2.circle(out, (int(px), int(py)), 9, cor, -1)
                cv2.putText(out, f"Esp: {esp:.1f}mm",
                            (int(px) + 13, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

        return out

    # ──────────────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────────────
    def limpar_medicoes(self):
        self.pontos_medidos       = []
        self.profundidades_pontos = []
        self.espessuras_pontos    = []
        self.historico_medidas.clear()
        self.ponto_calibracao     = None
        self.escala_px_para_mm    = None
        self.calibrado            = False

    def limpar_referencia(self):
        self.mapa_referencia     = None
        self.prof_referencia_pts = {}

    def guardar_json(self, path="espessura_espuma.json"):
        dados = {
            "calibrado":        self.calibrado,
            "escala_px_para_mm": self.escala_px_para_mm,
            "referencia_mm":    self.referencia_profundidade_mm,
            "ponto_calibracao": list(self.ponto_calibracao) if self.ponto_calibracao else None,
            "tem_referencia":   self.tem_referencia(),
            "medicoes": [
                {"x": px, "y": py, "profundidade_mm": prof}
                for (px, py), prof in zip(self.pontos_medidos, self.profundidades_pontos)
            ],
            "espessuras": self.espessuras_pontos,
        }
        with open(path, 'w') as f:
            json.dump(dados, f, indent=2)
        print(f"  ✓ Guardado em {path}")

    def carregar_json(self, path="espessura_espuma.json"):
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                dados = json.load(f)
            self.calibrado               = dados.get("calibrado", False)
            self.escala_px_para_mm       = dados.get("escala_px_para_mm")
            self.referencia_profundidade_mm = dados.get("referencia_mm", 10.0)
            pc = dados.get("ponto_calibracao")
            self.ponto_calibracao = tuple(pc) if pc else None
            self.pontos_medidos       = []
            self.profundidades_pontos = []
            for p in dados.get("medicoes", []):
                self.pontos_medidos.append((p["x"], p["y"]))
                self.profundidades_pontos.append(p["profundidade_mm"])
            self.espessuras_pontos = dados.get("espessuras", [])
            return True
        except Exception as e:
            print(f"  ✗ Erro ao carregar: {e}")
            return False
import cv2
import numpy as np
import json
import os

# ==============================================================================
# MEDIÇÃO DE ALTURA DE ESPUMA POR CÂMARA
# Funciona com câmara normal de portátil, sem hardware extra.
#
# PRINCÍPIO:
#   1. Calibras a escala: defines quantos mm tem a peça de referência
#   2. O sistema aprende a largura da peça em pixels → calcula px_por_mm
#   3. Em cada frame, deteta a peça e a espuma por cima
#   4. Mede a diferença de altura em pixels → converte para mm
# ==============================================================================

class FoamHeightMeasurer:
    """
    Medidor de altura de espuma por visão 2D (câmara top-down ou ligeiramente inclinada).

    Parâmetros
    ----------
    referencia_mm : float
        Dimensão real conhecida da peça (ex: largura = 100.0 mm)
    eixo_medicao : str
        'horizontal' → mede largura da peça para calibrar
        'vertical'   → mede altura da peça para calibrar
    """

    def __init__(self, referencia_mm: float = 100.0, eixo_medicao: str = "horizontal"):
        self.referencia_mm   = referencia_mm
        self.eixo_medicao    = eixo_medicao

        self.px_por_mm       = None   # calculado na calibração
        self.base_altura_px  = None   # altura do topo da peça sem espuma (referência)
        self.espuma_altura_px = None  # altura do topo da espuma (medição actual)

        # Parâmetros de segmentação (ajustáveis via UI)
        self.thresh_base   = 80    # threshold para detetar peça base
        self.thresh_espuma = 120   # threshold para detetar espuma
        self.min_area      = 3000  # área mínima de contorno válido (px²)

        # Estado
        self.calibrado        = False
        self.ref_capturada    = False
        self.altura_espuma_mm = 0.0
        self.historico_mm     = []   # últimas N medições para suavizar

        # CLAHE para normalizar iluminação
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ──────────────────────────────────────────────────────────────────────────
    # PRÉ-PROCESSAMENTO
    # ──────────────────────────────────────────────────────────────────────────
    def _pre_processar(self, frame):
        """Normaliza iluminação e converte para cinzento."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        frame_norm = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (7, 7), 0)

    # ──────────────────────────────────────────────────────────────────────────
    # DETEÇÃO DE CONTORNO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def _detetar_contorno_maior(self, gray, thresh_val):
        """Devolve o maior contorno encontrado acima do threshold dado."""
        _, bin_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,  kernel, iterations=1)
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, bin_img
        maior = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(maior) < self.min_area:
            return None, bin_img
        return maior, bin_img

    # ──────────────────────────────────────────────────────────────────────────
    # CALIBRAÇÃO — aprende px_por_mm da peça de referência
    # ──────────────────────────────────────────────────────────────────────────
    def calibrar(self, frame):
        """
        Chama com um frame da peça SEM espuma.
        Mede a dimensão da peça em pixels e calcula a escala px/mm.
        Devolve True se calibração foi bem sucedida.
        """
        gray   = self._pre_processar(frame)
        cnt, _ = self._detetar_contorno_maior(gray, self.thresh_base)
        if cnt is None:
            print("[Calibração] Não foi possível detetar a peça. Ajusta o threshold.")
            return False

        x, y, w, h = cv2.boundingRect(cnt)
        dimensao_px = w if self.eixo_medicao == "horizontal" else h

        self.px_por_mm = dimensao_px / self.referencia_mm
        self.calibrado = True
        print(f"[Calibração] OK → {dimensao_px}px = {self.referencia_mm}mm "
              f"→ escala = {self.px_por_mm:.3f} px/mm")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # CAPTURA DE REFERÊNCIA — topo da peça sem espuma
    # ──────────────────────────────────────────────────────────────────────────
    def capturar_referencia_base(self, frame):
        """
        Chama com a peça SEM espuma após calibrar.
        Guarda a posição do topo da peça (y mínimo do bounding rect).
        Devolve True se bem sucedido.
        """
        if not self.calibrado:
            print("[Referência] Calibra primeiro.")
            return False

        gray   = self._pre_processar(frame)
        cnt, _ = self._detetar_contorno_maior(gray, self.thresh_base)
        if cnt is None:
            print("[Referência] Peça não detetada.")
            return False

        _, y, _, h = cv2.boundingRect(cnt)
        # Topo da peça = y mínimo do bounding rect
        self.base_altura_px = y
        self.ref_capturada  = True
        print(f"[Referência] Topo da peça base capturado em y={y}px")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # MEDIÇÃO — altura da espuma em mm
    # ──────────────────────────────────────────────────────────────────────────
    def medir(self, frame):
        """
        Mede a altura da espuma no frame actual.
        Devolve (altura_mm, frame_anotado, detalhes_dict).
        """
        frame_out = frame.copy()
        detalhes  = {"altura_mm": 0.0, "status": "AGUARDAR", "erro": None}

        if not self.calibrado or not self.ref_capturada:
            cv2.putText(frame_out, "CALIBRA E CAPTURA REFERÊNCIA PRIMEIRO",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            detalhes["erro"] = "não calibrado"
            return 0.0, frame_out, detalhes

        gray = self._pre_processar(frame)

        # 1. Detetar peça base
        cnt_base, bin_base = self._detetar_contorno_maior(gray, self.thresh_base)
        if cnt_base is None:
            cv2.putText(frame_out, "PEÇA NÃO DETETADA", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            detalhes["erro"] = "peça não detetada"
            return 0.0, frame_out, detalhes

        # 2. Detetar espuma (threshold diferente — espuma é tipicamente mais clara ou mais escura)
        cnt_espuma, bin_espuma = self._detetar_contorno_maior(gray, self.thresh_espuma)

        # 3. Calcular altura
        _, y_base, _, h_base = cv2.boundingRect(cnt_base)

        if cnt_espuma is not None:
            _, y_esp, _, _ = cv2.boundingRect(cnt_espuma)
            # Diferença entre topo da espuma e topo da peça base de referência
            diff_px      = self.base_altura_px - y_esp
            altura_mm    = max(0.0, diff_px / self.px_por_mm)
        else:
            # Sem espuma detetada — altura zero
            altura_mm = 0.0

        # 4. Suavizar com média das últimas 5 medições
        self.historico_mm.append(altura_mm)
        if len(self.historico_mm) > 5:
            self.historico_mm.pop(0)
        altura_suavizada = np.mean(self.historico_mm)

        self.altura_espuma_mm = altura_suavizada

        # 5. Anotar frame
        # Contorno da peça base (azul)
        cv2.drawContours(frame_out, [cnt_base], -1, (255, 100, 0), 2)

        # Linha do topo de referência (verde)
        h_frame, w_frame = frame.shape[:2]
        cv2.line(frame_out, (0, self.base_altura_px), (w_frame, self.base_altura_px),
                 (0, 255, 0), 1)
        cv2.putText(frame_out, "BASE REF", (10, self.base_altura_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if cnt_espuma is not None:
            # Contorno espuma (laranja)
            cv2.drawContours(frame_out, [cnt_espuma], -1, (0, 165, 255), 2)
            _, y_esp, _, _ = cv2.boundingRect(cnt_espuma)
            # Linha do topo da espuma (amarelo)
            cv2.line(frame_out, (0, y_esp), (w_frame, y_esp), (0, 220, 255), 1)
            cv2.putText(frame_out, "TOPO ESPUMA", (10, y_esp - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
            # Seta de medição
            x_seta = w_frame - 60
            cv2.arrowedLine(frame_out, (x_seta, self.base_altura_px),
                            (x_seta, y_esp), (255, 255, 255), 2, tipLength=0.2)

        # Painel de resultados
        cv2.rectangle(frame_out, (10, 10), (380, 100), (20, 20, 20), -1)
        cv2.rectangle(frame_out, (10, 10), (380, 100), (60, 60, 60), 1)
        cv2.putText(frame_out, f"ALTURA ESPUMA: {altura_suavizada:.2f} mm",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        cv2.putText(frame_out, f"Escala: {self.px_por_mm:.3f} px/mm",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame_out, f"Ref: {self.referencia_mm}mm | Thresh base:{self.thresh_base} espuma:{self.thresh_espuma}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        detalhes["altura_mm"] = round(float(altura_suavizada), 2)
        detalhes["status"]    = "OK"
        return float(altura_suavizada), frame_out, detalhes

    # ──────────────────────────────────────────────────────────────────────────
    # GUARDAR / CARREGAR CALIBRAÇÃO
    # ──────────────────────────────────────────────────────────────────────────
    def guardar_calibracao(self, path="calibracao_espuma.json"):
        dados = {
            "referencia_mm":  self.referencia_mm,
            "eixo_medicao":   self.eixo_medicao,
            "px_por_mm":      self.px_por_mm,
            "base_altura_px": self.base_altura_px,
            "thresh_base":    self.thresh_base,
            "thresh_espuma":  self.thresh_espuma,
        }
        with open(path, "w") as f:
            json.dump(dados, f, indent=2)
        print(f"[Calibração] Guardada em {path}")

    def carregar_calibracao(self, path="calibracao_espuma.json"):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            dados = json.load(f)
        self.referencia_mm  = dados["referencia_mm"]
        self.eixo_medicao   = dados["eixo_medicao"]
        self.px_por_mm      = dados["px_por_mm"]
        self.base_altura_px = dados["base_altura_px"]
        self.thresh_base    = dados["thresh_base"]
        self.thresh_espuma  = dados["thresh_espuma"]
        self.calibrado      = True
        self.ref_capturada  = True
        print(f"[Calibração] Carregada de {path} — escala={self.px_por_mm:.3f} px/mm")
        return True


# ==============================================================================
# APLICAÇÃO DE DEMONSTRAÇÃO STANDALONE
# Corre este ficheiro directamente para testar sem o projeto principal.
# ==============================================================================
def main():
    print("=" * 60)
    print("  MEDIDOR DE ALTURA DE ESPUMA")
    print("  Controlos:")
    print("    C  → Calibrar (aponta para a peça SEM espuma)")
    print("    R  → Capturar referência base")
    print("    S  → Guardar calibração em ficheiro")
    print("    L  → Carregar calibração de ficheiro")
    print("    +/-→ Ajustar threshold da BASE")
    print("    [/]→ Ajustar threshold da ESPUMA")
    print("    Q  → Sair")
    print("=" * 60)

    ref_mm = input("Dimensão real conhecida da peça em mm (ex: 100): ").strip()
    try:
        ref_mm = float(ref_mm)
    except ValueError:
        ref_mm = 100.0
        print(f"Valor inválido — a usar {ref_mm}mm")

    eixo = input("Eixo de calibração [H=horizontal / V=vertical] (H): ").strip().upper()
    eixo = "vertical" if eixo == "V" else "horizontal"

    medidor = FoamHeightMeasurer(referencia_mm=ref_mm, eixo_medicao=eixo)

    # Tenta carregar calibração existente
    if medidor.carregar_calibracao():
        print("Calibração anterior carregada. Podes começar a medir directamente.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Câmara não encontrada!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Instrução no ecrã
        estado = ""
        if not medidor.calibrado:
            estado = "[C] Calibrar  |  [L] Carregar"
        elif not medidor.ref_capturada:
            estado = "[R] Capturar referência base"
        else:
            estado = "A medir...  [S] Guardar  [Q] Sair"

        altura_mm, frame_anot, _ = medidor.medir(frame)
        cv2.putText(frame_anot, estado, (10, frame_anot.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Medidor de Espuma | VISION PRO", frame_anot)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if medidor.calibrar(frame):
                print("✓ Calibração concluída. Agora captura a referência com [R].")
            else:
                print("✗ Falhou — garante que a peça está bem visível e ajusta o threshold.")
        elif key == ord('r'):
            if medidor.capturar_referencia_base(frame):
                print("✓ Referência base capturada. Sistema pronto a medir.")
        elif key == ord('s'):
            medidor.guardar_calibracao()
        elif key == ord('l'):
            medidor.carregar_calibracao()
        elif key == ord('+'):
            medidor.thresh_base = min(255, medidor.thresh_base + 5)
            print(f"Thresh base: {medidor.thresh_base}")
        elif key == ord('-'):
            medidor.thresh_base = max(0, medidor.thresh_base - 5)
            print(f"Thresh base: {medidor.thresh_base}")
        elif key == ord(']'):
            medidor.thresh_espuma = min(255, medidor.thresh_espuma + 5)
            print(f"Thresh espuma: {medidor.thresh_espuma}")
        elif key == ord('['):
            medidor.thresh_espuma = max(0, medidor.thresh_espuma - 5)
            print(f"Thresh espuma: {medidor.thresh_espuma}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
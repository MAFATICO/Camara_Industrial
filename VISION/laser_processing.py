import cv2
import numpy as np
import json
from sklearn.linear_model import RANSACRegressor

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
ANGULO_LASER = 34.92
historico_medidas = []

# =========================================================
# CLASSE PRINCIPAL
# =========================================================
class SistemaTriangulacao:
    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        # -------------------------------------------------
        # Desativar automatismos da webcam
        # -------------------------------------------------

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

        # -------------------------------------------------
        # Variáveis calibração
        # -------------------------------------------------

        self.y_referencia = None
        self.escala_mm_px = None
        self.roi = None

    # =====================================================
    # CAPTURA FRAME
    # =====================================================
    def obter_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    # =====================================================
    # ROI AUTOMÁTICA
    # =====================================================
    def encontrar_roi(self, mask):

        x, y, w, h = cv2.boundingRect(mask)
        margem = 20
        x = max(0, x - margem)
        y = max(0, y - margem)
        w += margem * 2
        h += margem * 2

        self.roi = (x, y, w, h)

        return self.roi

    # =====================================================
    # MÁSCARA HSV
    # =====================================================

    def obter_mascara_laser(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 80, 120])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 80, 120])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        # -------------------------------------------------
        # MORPHOLOGY
        # -------------------------------------------------
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask

    # =====================================================
    # REGRESSÃO LINEAR + RANSAC
    # =====================================================
    def calcular_linha_laser(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            return None

        X = xs.reshape(-1,1)
        Y = ys
        ransac = RANSACRegressor()
        ransac.fit(X, Y)
        a = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_

        return a, b

    # =====================================================
    # DESENHAR RETA
    # =====================================================
    def desenhar_linha(self, frame, a, b):
        x1 = 0
        y1 = int(a*x1 + b)

        x2 = frame.shape[1]
        y2 = int(a*x2 + b)

        cv2.line(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

    # =====================================================
    # CALIBRAR REFERÊNCIA
    # =====================================================
    def calibrar_referencia(self, frame):
        mask = self.obter_mascara_laser(frame)
        resultado = self.calcular_linha_laser(mask)

        if resultado is None:
            return False

        a, b = resultado
        x_ref = frame.shape[1] // 2
        self.y_referencia = a*x_ref + b
        print("REFERÊNCIA CALIBRADA")
        print("Y referência:", self.y_referencia)

        # ROI automática
        self.encontrar_roi(mask)

        return True

    # =====================================================
    # CALIBRAR ESCALA
    # =====================================================
    def calibrar_escala(self, frame, altura_bloco_mm):
        if self.y_referencia is None:
            return False

        mask = self.obter_mascara_laser(frame)
        resultado = self.calcular_linha_laser(mask)

        if resultado is None:
            return False

        a, b = resultado
        x_ref = frame.shape[1] // 2
        y_atual = a*x_ref + b
        delta_px = abs(self.y_referencia - y_atual)

        delta_mm_horizontal = (
            altura_bloco_mm *
            np.tan(np.radians(ANGULO_LASER))
        )

        self.escala_mm_px = (
            delta_mm_horizontal / delta_px
        )

        print("ESCALA CALIBRADA")
        print("Escala:", self.escala_mm_px)

        return True

    # =====================================================
    # MEDIR ESPUMA
    # =====================================================
    def medir_espuma(self, frame):
        if self.y_referencia is None:
            return None

        if self.escala_mm_px is None:
            return None

        # -------------------------------------------------
        # ROI
        # -------------------------------------------------
        if self.roi is not None:
            x, y, w, h = self.roi
            frame_roi = frame[y:y+h, x:x+w]

        else:
            frame_roi = frame

        # -------------------------------------------------
        # MÁSCARA
        # -------------------------------------------------
        mask = self.obter_mascara_laser(frame_roi)
        resultado = self.calcular_linha_laser(mask)
        if resultado is None:
            return None

        a, b = resultado
        x_ref = frame_roi.shape[1] // 2
        y_laser = a*x_ref + b
        delta_px = self.y_referencia - y_laser

        deslocamento_mm = (
            delta_px *
            self.escala_mm_px
        )

        altura_mm = (
            deslocamento_mm /
            np.tan(np.radians(ANGULO_LASER))
        )

        # -------------------------------------------------
        # MÉDIA MÓVEL
        # -------------------------------------------------
        historico_medidas.append(altura_mm)

        if len(historico_medidas) > 10:
            historico_medidas.pop(0)

        altura_filtrada = np.mean(historico_medidas)

        # -------------------------------------------------
        # DESENHAR
        # -------------------------------------------------
        self.desenhar_linha(frame_roi, a, b)
        cv2.putText(
            frame,
            f"Altura: {altura_filtrada:.3f} mm",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        return altura_filtrada

    # =====================================================
    # GUARDAR CALIBRAÇÃO
    # =====================================================
    def guardar_calibracao(self):

        dados = {
            "y_referencia": float(self.y_referencia),
            "escala_mm_px": float(self.escala_mm_px),
            "roi": self.roi
        }

        with open("calibracao.json", "w") as f:
            json.dump(dados, f, indent=4)

        print("CALIBRAÇÃO GUARDADA")

    # =====================================================
    # CARREGAR CALIBRAÇÃO
    # =====================================================
    def carregar_calibracao(self):

        with open("calibracao.json") as f:

            dados = json.load(f)
        self.y_referencia = dados["y_referencia"]
        self.escala_mm_px = dados["escala_mm_px"]
        self.roi = tuple(dados["roi"])

        print("CALIBRAÇÃO CARREGADA")

# =========================================================
# MAIN
# =========================================================
sistema = SistemaTriangulacao()
while True:
    frame = sistema.obter_frame()

    if frame is None:
        break

    # -----------------------------------------------------
    # MEDIR
    # -----------------------------------------------------
    sistema.medir_espuma(frame)

    # -----------------------------------------------------
    # MOSTRAR
    # -----------------------------------------------------
    cv2.imshow("Sistema Industrial", frame)
    tecla = cv2.waitKey(1) & 0xFF

    # -----------------------------------------------------
    # Q = SAIR
    # -----------------------------------------------------
    if tecla == ord('q'):
        break

    # -----------------------------------------------------
    # R = CALIBRAR REFERÊNCIA
    # -----------------------------------------------------
    elif tecla == ord('r'):

        sistema.calibrar_referencia(frame)

    # -----------------------------------------------------
    # E = CALIBRAR ESCALA
    # -----------------------------------------------------
    elif tecla == ord('e'):

        altura_bloco = 10.0

        sistema.calibrar_escala(
            frame,
            altura_bloco
        )

    # -----------------------------------------------------
    # G = GUARDAR
    # -----------------------------------------------------
    elif tecla == ord('g'):

        sistema.guardar_calibracao()

    # -----------------------------------------------------
    # C = CARREGAR
    # -----------------------------------------------------
    elif tecla == ord('c'):

        sistema.carregar_calibracao()

# =========================================================
# FECHAR
# =========================================================
sistema.cap.release()
cv2.destroyAllWindows()
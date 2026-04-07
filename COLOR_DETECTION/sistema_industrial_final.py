import sys
import cv2
import numpy as np
import os
import sqlite3
import json
import time
from PIL import Image

# --- CONFIGURAÇÃO DE BACKEND MATPLOTLIB ---
os.environ["QT_API"] = "pyside6"
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFrame,
                               QStackedWidget, QSlider, QScrollArea, QGridLayout,
                               QButtonGroup, QInputDialog, QMessageBox)

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES (Vindas do sistema_unificado.py)
# ==============================================================================
DB_NAME = "sistema_industrial.db"
DIR_DATASET = "dataset_ia"
COR_DESTAQUE_HEX = "#FF8C00"  # Laranja do novo menu
CORES_DETECCAO = {
    "AZUL": (np.array([85, 100, 50]), np.array([140, 255, 255]), (255, 0, 0)),
    "PRETO": (np.array([0, 0, 0]), np.array([180, 255, 50]), (60, 60, 60))
}


# ==============================================================================
# COMPONENTE DE VÍDEO PERSONALIZADO (Para capturar cliques)
# ==============================================================================
class VideoLabel(QLabel):
    clique_esquerdo = Signal(int, int)
    clique_direito = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clique_esquerdo.emit(event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.clique_direito.emit(event.x(), event.y())


# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================
class VisionProApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Inicialização de Estado ---
        if not os.path.exists(DIR_DATASET):
            os.makedirs(DIR_DATASET)

        self.inicializar_bd()

        self.peca_ativa_id = None
        self.peca_ativa_nome = "Nenhum"
        self.inspecao_ativa = False
        self.aba_ativa = "INSPECAO"

        # Variáveis de Processamento
        self.lista_poligonos = []
        self.ponto_atual = []
        self.frame_fixo_treino = None
        self.foto_referencia_processamento = None
        self.modelo_aprendido = None
        self.total_ok = 0
        self.total_fail = 0
        self.ultima_gravacao = 0
        self.valor_threshold = 100
        self.valor_sensibilidade = 800
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Mapeamento para cliques
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # --- Configuração UI ---
        self.setWindowTitle("VISION PRO | Terminal Industrial")
        self.resize(1400, 950)
        self._apply_style()
        self._init_ui()
        self._setup_shortcuts()

        # --- Hardware ---
        self.cap = cv2.VideoCapture(0)

        # Iniciar Timer de Loop (16ms ~ 60fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_loop)
        self.timer.start(16)

        # Carregar primeira peça
        self.carregar_dados_iniciais()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0F1115; }
            QFrame#Sidebar { background-color: #080808; border-right: 1px solid #1A1A1A; }
            QFrame#ControlBar { background-color: #16191E; border-top: 1px solid #2C313C; border-radius: 12px; }
            QFrame#StatCard { background-color: #1E2227; border-radius: 10px; border: 1px solid #333; min-height: 100px; }

            QLabel { color: #ABB2BF; font-family: 'Segoe UI'; }

            QPushButton#NavBtn { 
                background: transparent; border: none; color: #848da0; 
                text-align: left; padding: 12px 20px; font-weight: bold; font-size: 13px; 
            }
            QPushButton#NavBtn:hover { color: #FF8C00; background: #1A1A1A; }
            QPushButton#NavBtn[active="true"] { color: #FF8C00; background: #1A1A1A; border-left: 3px solid #FF8C00; }

            QPushButton#ModelBtn { 
                background: #111; border: 1px solid #222; color: #888; 
                padding: 10px; text-align: left; border-radius: 4px; margin-bottom: 2px; 
            }
            QPushButton#ModelBtn:checked { border-color: #FF8C00; color: #FF8C00; background: #1A1A1A; }

            QPushButton#ControlBtn { 
                background: #21252B; border: 1px solid #3E4451; color: white; 
                padding: 10px 15px; border-radius: 6px; font-weight: bold; font-size: 11px; 
            }
            QPushButton#ControlBtn:hover { border-color: #FF8C00; background: #2C313C; }

            QPushButton#DangerBtn { 
                background: #2A1B1B; border: 1px solid #4A2626; color: #FF6666; 
                padding: 10px 15px; border-radius: 6px; font-weight: bold; font-size: 11px; 
            }
            QPushButton#DangerBtn:hover { background: #3D2222; border-color: #FF5555; }

            QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }
            QSlider::handle:horizontal { background: #FF8C00; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
        """)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(260)
        side_v = QVBoxLayout(self.sidebar)

        logo = QLabel("VISION PRO")
        logo.setStyleSheet("color:#FF8C00; font-size:22px; font-weight:bold; margin:20px;")
        side_v.addWidget(logo)

        # Menus Laterais
        self.nav_buttons = {}
        abas = [("INSPEÇÃO", "INSPECAO", 0),
                ("TREINAR", "TREINAR", 0),
                ("REFERÊNCIAS", "REFERENCIAS", 0),
                ("IA / DATASET", "IA / DATASET", 0),
                ("ESTATÍSTICAS", "ESTATISTICAS", 1),
                ("COMANDOS", "COMANDOS", 2)]

        for nome, chave, idx in abas:
            btn = QPushButton(nome)
            btn.setObjectName("NavBtn")
            btn.clicked.connect(lambda chk=False, k=chave, i=idx: self.mudar_aba(k, i))
            side_v.addWidget(btn)
            self.nav_buttons[chave] = btn

        side_v.addSpacing(25)
        side_v.addWidget(QLabel("MODELOS NA BD"), alignment=Qt.AlignCenter)

        h_m = QHBoxLayout()
        b_add = QPushButton("+ NOVO");
        b_add.setObjectName("ControlBtn");
        b_add.clicked.connect(self.adicionar_novo_modelo)
        b_del = QPushButton("- APAGAR");
        b_del.setObjectName("DangerBtn");
        b_del.clicked.connect(self.eliminar_modelo_atual)
        h_m.addWidget(b_add);
        h_m.addWidget(b_del)
        side_v.addLayout(h_m)

        scroll = QScrollArea();
        scroll.setWidgetResizable(True);
        scroll.setStyleSheet("background:transparent; border:none;")
        self.model_container = QWidget();
        self.model_layout = QVBoxLayout(self.model_container);
        self.model_layout.setAlignment(Qt.AlignTop)
        self.model_group = QButtonGroup(self);
        self.model_group.setExclusive(True)
        scroll.setWidget(self.model_container)
        side_v.addWidget(scroll)
        side_v.addStretch()

        # --- ÁREA CENTRAL ---
        self.pages = QStackedWidget()

        # 1. PÁGINA DE VÍDEO (Inspeção, Treino, Referências, IA)
        self.page_video = QWidget()
        video_v = QVBoxLayout(self.page_video)

        self.lbl_nome_peca_top = QLabel("...")
        self.lbl_nome_peca_top.setStyleSheet("font-size:18px; color:#FF8C00; font-weight:bold;")
        video_v.addWidget(self.lbl_nome_peca_top)

        self.video_container = QFrame();
        self.video_container.setStyleSheet("background:#000; border-radius:10px;")
        v_box = QVBoxLayout(self.video_container)
        self.video_label = VideoLabel()
        self.video_label.clique_esquerdo.connect(self.gerir_clique_esquerdo)
        self.video_label.clique_direito.connect(self.gerir_clique_direito)
        v_box.addWidget(self.video_label)
        video_v.addWidget(self.video_container, stretch=10)

        # Barra de Controlos Inferior
        self.bottom_bar = QFrame();
        self.bottom_bar.setObjectName("ControlBar");
        self.bottom_bar.setFixedHeight(140)
        b_lay = QHBoxLayout(self.bottom_bar)

        slid_v = QVBoxLayout()
        slid_v.addWidget(QLabel("THRESHOLD"))
        self.slider_thresh = QSlider(Qt.Horizontal);
        self.slider_thresh.setRange(0, 255);
        self.slider_thresh.setValue(self.valor_threshold)
        self.slider_thresh.valueChanged.connect(self.set_thresh)
        slid_v.addWidget(self.slider_thresh)
        slid_v.addWidget(QLabel("SENSIBILIDADE"))
        self.slider_sens = QSlider(Qt.Horizontal);
        self.slider_sens.setRange(0, 10000);
        self.slider_sens.setValue(self.valor_sensibilidade)
        self.slider_sens.valueChanged.connect(self.set_sens)
        slid_v.addWidget(self.slider_sens)
        b_lay.addLayout(slid_v, stretch=2)

        draw_h = QHBoxLayout()
        btns_ctrl = [("↺ VOLTAR (Z)", self.retroceder_ponto, "ControlBtn"),
                     ("🗑️ LIMPAR (D)", self.reset_poligonos_db, "DangerBtn"),
                     ("💾 SALVAR (S)", self.comando_salvar_s, "ControlBtn"),
                     ("▶ INSPEÇÃO (I)", self.toggle_inspecao, "ControlBtn")]
        for t, f, style in btns_ctrl:
            b = QPushButton(t);
            b.setObjectName(style);
            b.clicked.connect(f);
            draw_h.addWidget(b)
        b_lay.addLayout(draw_h, stretch=3)
        video_v.addWidget(self.bottom_bar)

        # 2. PÁGINA DE ESTATÍSTICAS
        self.page_stats = QWidget()
        stats_v = QVBoxLayout(self.page_stats);
        stats_v.setContentsMargins(30, 30, 30, 30)
        self.lbl_stats_title = QLabel("ESTATÍSTICAS DO MODELO");
        self.lbl_stats_title.setStyleSheet("font-size:18px; color:#FF8C00; font-weight:bold;")
        stats_v.addWidget(self.lbl_stats_title)

        cards_h = QHBoxLayout()
        self.card_ok = self._create_card("PEÇAS OK", "#2ECC71");
        cards_h.addWidget(self.card_ok)
        self.card_nok = self._create_card("PEÇAS NOK", "#E74C3C");
        cards_h.addWidget(self.card_nok)
        stats_v.addLayout(cards_h)

        self.fig, self.ax = plt.subplots(figsize=(5, 5));
        self.fig.patch.set_facecolor('#0F1115')
        self.canvas = FigureCanvas(self.fig);
        stats_v.addWidget(self.canvas)

        # 3. PÁGINA DE COMANDOS
        self.page_cmds = QWidget()
        cmd_grid = QGridLayout(self.page_cmds)
        comandos_txt = ["[ S ] - SALVAR REFERÊNCIA / POLÍGONO", "[ I ] - INICIAR / PARAR INSPEÇÃO",
                        "[ Z ] - RETROCEDER PONTO", "[ D ] - LIMPAR DESENHOS",
                        "[ P ] - CAPTURAR PARA IA", "[ R ] - RESETAR CONTADORES",
                        "[ N ] - RENOMEAR MODELO"]
        for i, txt in enumerate(comandos_txt):
            btn = QPushButton(txt);
            btn.setObjectName("ControlBtn");
            btn.setMinimumHeight(60)
            cmd_grid.addWidget(btn, i // 2, i % 2)

        self.pages.addWidget(self.page_video)
        self.pages.addWidget(self.page_stats)
        self.pages.addWidget(self.page_cmds)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.pages)

    def _create_card(self, title, color):
        card = QFrame();
        card.setObjectName("StatCard");
        v = QVBoxLayout(card)
        v.addWidget(QLabel(title), alignment=Qt.AlignCenter)
        lbl = QLabel("0");
        lbl.setStyleSheet(f"font-size:28px; font-weight:bold; color:{color};");
        v.addWidget(lbl, alignment=Qt.AlignCenter)
        card.lbl = lbl
        return card

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("S"), self, self.comando_salvar_s)
        QShortcut(QKeySequence("I"), self, self.toggle_inspecao)
        QShortcut(QKeySequence("Z"), self, self.retroceder_ponto)
        QShortcut(QKeySequence("D"), self, self.reset_poligonos_db)
        QShortcut(QKeySequence("P"), self, self.capturar_para_ia)
        QShortcut(QKeySequence("R"), self, self.reset_contadores)
        QShortcut(QKeySequence("N"), self, self.renomear_peca)

    # ==========================================================================
    # LÓGICA DE BASE DE DADOS E GESTÃO DE MODELOS
    # ==========================================================================
    def inicializar_bd(self):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS pecas
                          (
                              id             INTEGER PRIMARY KEY,
                              nome           TEXT,
                              poligonos      TEXT,
                              pontos_medicao TEXT,
                              medida_ideal   REAL,
                              tolerancia     REAL,
                              escala         REAL,
                              foto_ref       BLOB,
                              aprovadas      INTEGER DEFAULT 0,
                              rejeitadas     INTEGER DEFAULT 0,
                              bordas_ref     INTEGER DEFAULT 0
                          )''')
        conn.commit();
        conn.close()

    def carregar_dados_iniciais(self):
        self.atualizar_lista_modelos()
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM pecas LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row: self.carregar_dados_peca(row[0])

    def carregar_dados_peca(self, id_p):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pecas WHERE id=?", (id_p,))
        p = cursor.fetchone()
        conn.close()

        if p:
            self.peca_ativa_id = p[0]
            self.peca_ativa_nome = p[1]
            self.lista_poligonos = json.loads(p[2]) if p[2] else []
            self.total_ok = p[8] if p[8] is not None else 0
            self.total_fail = p[9] if p[9] is not None else 0

            if p[7]:
                nparr = np.frombuffer(p[7], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.foto_referencia_processamento = img
                self.processar_referencia(img)
            else:
                self.foto_referencia_processamento = None
                self.modelo_aprendido = None

            self.atualizar_ui_labels()

    def atualizar_ui_labels(self):
        txt = f"MODELO: {self.peca_ativa_nome} | ID: {self.peca_ativa_id}"
        self.lbl_nome_peca_top.setText(txt)
        self.lbl_stats_title.setText(f"ESTATÍSTICAS: {self.peca_ativa_nome}")
        self.card_ok.lbl.setText(str(self.total_ok))
        self.card_nok.lbl.setText(str(self.total_fail))
        self.update_pie(self.total_ok, self.total_fail)

    def atualizar_lista_modelos(self):
        for i in reversed(range(self.model_layout.count())):
            self.model_layout.itemAt(i).widget().setParent(None)

        conn = sqlite3.connect(DB_NAME)
        data = conn.execute("SELECT id, nome FROM pecas").fetchall()
        conn.close()

        for pid, nome in data:
            btn = QPushButton(f"📦 ID {pid:02} | {nome[:12]}")
            btn.setObjectName("ModelBtn")
            btn.setCheckable(True)
            if pid == self.peca_ativa_id: btn.setChecked(True)
            btn.clicked.connect(lambda chk=False, x=pid: self.carregar_dados_peca(x))
            self.model_group.addButton(btn)
            self.model_layout.addWidget(btn)

    def update_pie(self, ok, nok):
        self.ax.clear()
        if ok == 0 and nok == 0:
            self.ax.text(0.5, 0.5, "Sem Dados", color="white", ha="center")
        else:
            self.ax.pie([ok, nok], labels=["OK", "NOK"], colors=["#2ECC71", "#E74C3C"],
                        autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.4})
            self.ax.text(0, 0, f"TOTAL\n{ok + nok}", color='white', ha='center', va='center', fontweight='bold')
        self.canvas.draw()

    # ==========================================================================
    # LÓGICA DE VISÃO COMPUTACIONAL (Sistema Unificado)
    # ==========================================================================
    def processar_referencia(self, img):
        if img is None: return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe_img = self.clahe.apply(blur)
        _, thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos: self.modelo_aprendido = max(contornos, key=cv2.contourArea)

    def executar_inspecao(self, frame):
        status_final = "OK";
        motivo_fail = ""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filt = self.clahe.apply(gray)
        _, thresh = cv2.threshold(filt, self.valor_threshold, 255, cv2.THRESH_BINARY_INV)

        for poly in self.lista_poligonos:
            pts = np.array(poly)
            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, [pts], 255)
            zona_isolada = cv2.bitwise_and(thresh, mask)
            cnts, _ = cv2.findContours(zona_isolada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            zona_ok = False
            if cnts:
                c_max = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c_max) > self.valor_sensibilidade:
                    x_b, y_b, w_b, h_b = cv2.boundingRect(c_max)
                    if 0.4 < (float(w_b) / h_b) < 2.5:
                        zona_ok = True
                        cv2.drawContours(frame, [c_max], -1, (0, 255, 0), 2)

            if not zona_ok:
                status_final = "FAIL";
                motivo_fail = "Zona Falha"
                cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

        if status_final == "OK":
            hsv = cv2.cvtColor(cv2.bilateralFilter(frame, 9, 75, 75), cv2.COLOR_BGR2HSV)
            objetos = []
            for nome_cor, (low, high, _) in CORES_DETECCAO.items():
                m = cv2.inRange(hsv, low, high)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) > 3000: objetos.append((cv2.contourArea(c), c, nome_cor))

            if objetos:
                objetos.sort(key=lambda x: x[0], reverse=True)
                area, c_prin, cor_prin = objetos[0]
                if self.modelo_aprendido is not None:
                    if cv2.matchShapes(self.modelo_aprendido, c_prin, 1, 0.0) > 0.25:
                        status_final = "FAIL";
                        motivo_fail = "Forma Errada"

                cor_rect = (0, 255, 0) if status_final == "OK" else (0, 0, 255)
                cv2.drawContours(frame, [c_prin], -1, cor_rect, 2)
                cv2.putText(frame, f"{cor_prin} {motivo_fail}", (c_prin[0][0][0], c_prin[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_rect, 2)

        return status_final

    # ==========================================================================
    # LOOP PRINCIPAL E RENDEREZAÇÃO
    # ==========================================================================
    def atualizar_loop(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display_frame = frame.copy()

        if self.aba_ativa == "TREINAR":
            if self.frame_fixo_treino is not None:
                display_frame = self.frame_fixo_treino.copy()
                for poly in self.lista_poligonos: cv2.polylines(display_frame, [np.array(poly)], True, (100, 100, 100),
                                                                1)
                for pt in self.ponto_atual: cv2.circle(display_frame, pt, 4, (0, 255, 255), -1)
                if len(self.ponto_atual) > 1: cv2.polylines(display_frame, [np.array(self.ponto_atual)], False,
                                                            (0, 255, 255), 2)
                cv2.putText(display_frame, "MODO TREINO: CONGELADO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "PRESSIONE [S] PARA CONGELAR", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

        elif self.aba_ativa == "INSPECAO":
            if self.inspecao_ativa:
                status = self.executar_inspecao(display_frame)
                cv2.putText(display_frame, f"STATUS: {status}", (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if status == "OK" else (0, 0, 255), 3)
                if time.time() - self.ultima_gravacao > 1.2:
                    if status == "OK":
                        self.total_ok += 1
                    else:
                        self.total_fail += 1
                    self.ultima_gravacao = time.time()
                    self.salvar_progresso_bd()
                    self.atualizar_ui_labels()
            else:
                cv2.putText(display_frame, "INSPECAO PAUSADA [I]", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)

        elif self.aba_ativa == "REFERENCIAS":
            if self.foto_referencia_processamento is not None:
                display_frame = self.foto_referencia_processamento.copy()
                cv2.putText(display_frame, "REFERENCIA ATUAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 140, 0),
                            2)
            else:
                display_frame = np.zeros_like(frame);
                cv2.putText(display_frame, "SEM FOTO DE REF", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (100, 100, 100), 2)

        elif self.aba_ativa == "IA / DATASET":
            imgs = [f for f in os.listdir(DIR_DATASET) if f.startswith(f"peca_{self.peca_ativa_id}_")]
            cv2.putText(display_frame, f"DATASET: {len(imgs)} IMAGENS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

        self.exibir_frame(display_frame)

    def exibir_frame(self, frame):
        w_lbl = self.video_container.width() - 20
        h_lbl = self.video_container.height() - 20
        if w_lbl <= 0 or h_lbl <= 0: return

        h_img, w_img = frame.shape[:2]
        self.scale_factor = min(w_lbl / w_img, h_lbl / h_img)

        new_w = int(w_img * self.scale_factor)
        new_h = int(h_img * self.scale_factor)

        self.offset_x = (self.video_label.width() - new_w) / 2
        self.offset_y = (self.video_label.height() - new_h) / 2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w_img, h_img, w_img * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    # ==========================================================================
    # EVENTOS E COMANDOS
    # ==========================================================================
    def gerir_clique_esquerdo(self, x, y):
        if self.aba_ativa == "TREINAR" and self.frame_fixo_treino is not None:
            real_x = int((x - self.offset_x) / self.scale_factor)
            real_y = int((y - self.offset_y) / self.scale_factor)
            h, w = self.frame_fixo_treino.shape[:2]
            if 0 <= real_x < w and 0 <= real_y < h:
                self.ponto_atual.append((real_x, real_y))

    def gerir_clique_direito(self, x, y):
        self.ponto_atual = []

    def mudar_aba(self, chave, idx):
        self.aba_ativa = chave
        self.pages.setCurrentIndex(idx)
        self.frame_fixo_treino = None
        self.ponto_atual = []
        # Atualizar estilo botões
        for k, btn in self.nav_buttons.items():
            btn.setProperty("active", k == chave)
            btn.style().unpolish(btn);
            btn.style().polish(btn)

    def set_thresh(self, v):
        self.valor_threshold = v

    def set_sens(self, v):
        self.valor_sensibilidade = v

    def toggle_inspecao(self):
        self.inspecao_ativa = not self.inspecao_ativa

    def comando_salvar_s(self):
        if self.aba_ativa == "TREINAR":
            if self.frame_fixo_treino is None:
                ret, f = self.cap.read()
                if ret: self.frame_fixo_treino = cv2.flip(f, 1)
            else:
                if len(self.ponto_atual) > 2:
                    self.lista_poligonos.append(list(self.ponto_atual))
                    self.ponto_atual = []
                    self.salvar_progresso_bd()
                    QMessageBox.information(self, "Sucesso", "Polígono guardado.")
                elif QMessageBox.question(self, "Referência", "Salvar imagem como referência?") == QMessageBox.Yes:
                    conn = sqlite3.connect(DB_NAME)
                    _, buf = cv2.imencode(".jpg", self.frame_fixo_treino)
                    conn.execute("UPDATE pecas SET foto_ref=? WHERE id=?", (buf.tobytes(), self.peca_ativa_id))
                    conn.commit();
                    conn.close()
                    self.foto_referencia_processamento = self.frame_fixo_treino.copy()
                    self.processar_referencia(self.frame_fixo_treino)
                self.frame_fixo_treino = None

    def salvar_progresso_bd(self):
        conn = sqlite3.connect(DB_NAME)
        conn.execute("UPDATE pecas SET poligonos=?, aprovadas=?, rejeitadas=? WHERE id=?",
                     (json.dumps(self.lista_poligonos), self.total_ok, self.total_fail, self.peca_ativa_id))
        conn.commit();
        conn.close()

    def reset_poligonos_db(self):
        if QMessageBox.question(self, "Limpar", "Apagar desenhos do modelo?") == QMessageBox.Yes:
            self.lista_poligonos = [];
            self.ponto_atual = []
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET poligonos=? WHERE id=?", (json.dumps([]), self.peca_ativa_id))
            conn.commit();
            conn.close()

    def retroceder_ponto(self):
        if self.ponto_atual:
            self.ponto_atual.pop()
        elif self.lista_poligonos:
            self.lista_poligonos.pop()

    def capturar_para_ia(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            ts = int(time.time() * 1000)
            path = os.path.join(DIR_DATASET, f"peca_{self.peca_ativa_id}_{ts}.jpg")
            cv2.imwrite(path, frame)

    def reset_contadores(self):
        self.total_ok = 0;
        self.total_fail = 0
        self.salvar_progresso_bd();
        self.atualizar_ui_labels()

    def renomear_peca(self):
        novo, ok = QInputDialog.getText(self, "Renomear", "Novo nome:", text=self.peca_ativa_nome)
        if ok and novo:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET nome=? WHERE id=?", (novo, self.peca_ativa_id))
            conn.commit();
            conn.close()
            self.peca_ativa_nome = novo;
            self.atualizar_lista_modelos();
            self.atualizar_ui_labels()

    def adicionar_novo_modelo(self):
        conn = sqlite3.connect(DB_NAME);
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM pecas");
        res = cursor.fetchone()[0]
        nxt = (res + 1) if res else 1
        cursor.execute("INSERT INTO pecas (id, nome, poligonos) VALUES (?, ?, ?)", (nxt, f"Modelo {nxt:02d}", "[]"))
        conn.commit();
        conn.close();
        self.atualizar_lista_modelos()

    def eliminar_modelo_atual(self):
        if QMessageBox.question(self, "Eliminar", f"Eliminar {self.peca_ativa_nome}?") == QMessageBox.Yes:
            conn = sqlite3.connect(DB_NAME);
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pecas WHERE id=?", (self.peca_ativa_id,))
            conn.commit();
            conn.close();
            self.carregar_dados_iniciais()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionProApp()
    window.show()
    sys.exit(app.exec())

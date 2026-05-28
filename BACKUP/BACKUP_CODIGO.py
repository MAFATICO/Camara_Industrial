import sys
import cv2
import numpy as np
import os
import sqlite3
import json
import time
import struct
from collections import Counter, deque

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from VISION.roi_inspection import ROIInspection
from SCRIPTS.falhas_manager import FalhasManager

from CORE.depth_integration import DepthMeasurer, DEPTH_AI_DISPONIVEL

os.environ["QT_API"] = "pyside6"
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFrame,
                               QStackedWidget, QScrollArea, QGridLayout,
                               QButtonGroup, QInputDialog, QMessageBox, QLineEdit,
                               QDoubleSpinBox)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_NAME     = os.path.join(BASE_DIR, "sistema_industrial.db")
DIR_DATASET = os.path.join(BASE_DIR, "DATASET")

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

CORES_DETECCAO = {
    "AZUL":  (np.array([85, 100, 50]),  np.array([140, 255, 255]), (255, 0, 0)),
    "PRETO": (np.array([0, 0, 0]),      np.array([150, 50, 50]),   (60, 60, 60)),
}


# ==============================================================================
# VIDEO LABEL
# ==============================================================================
class VideoLabel(QLabel):
    clique_esquerdo = Signal(int, int)
    clique_direito  = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clique_esquerdo.emit(event.x(), event.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.clique_direito.emit(event.x(), event.y())
        super().mousePressEvent(event)


# ==============================================================================
# LOGIN
# ==============================================================================
class LoginScreen(QWidget):
    login_sucesso = Signal(str)

    def __init__(self):
        super().__init__()
        self.user_input = None
        self.pass_input = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0F1115;")
        container = QFrame()
        container.setFixedWidth(400)
        container.setStyleSheet("""
            QFrame { background-color: #16191E; border-radius: 15px; border: 1px solid #2C313C; padding: 20px; }
            QLineEdit { background-color: #0F1115; border: 1px solid #3E4451; color: white;
                        padding: 10px; border-radius: 5px; margin-bottom: 10px; }
            QPushButton { padding: 12px; border-radius: 5px; font-weight: bold; }
        """)
        c_lay = QVBoxLayout(container)
        logo = QLabel("VISION PRO | LOGIN")
        logo.setStyleSheet("color:#FF8C00; font-size:24px; font-weight:bold; margin-bottom:20px; border:none;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_lay.addWidget(logo)
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Utilizador")
        c_lay.addWidget(self.user_input)
        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("Palavra-passe")
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        c_lay.addWidget(self.pass_input)
        btn_login = QPushButton("ENTRAR")
        btn_login.setStyleSheet("background-color: #FF8C00; color: white;")
        btn_login.clicked.connect(self.autenticar)
        c_lay.addWidget(btn_login)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #2C313C;")
        c_lay.addWidget(line)
        lbl_gestao = QLabel("GESTÃO DE UTILIZADORES")
        lbl_gestao.setStyleSheet("border:none; color:#888; margin-top:10px;")
        c_lay.addWidget(lbl_gestao)
        h_btns = QHBoxLayout()
        btn_reg = QPushButton("REGISTAR")
        btn_reg.setStyleSheet("background-color: #21252B; color: #ABB2BF; border: 1px solid #3E4451;")
        btn_reg.clicked.connect(self.registar)
        btn_del = QPushButton("ELIMINAR")
        btn_del.setStyleSheet("background-color: #2A1B1B; color: #FF6666; border: 1px solid #4A2626;")
        btn_del.clicked.connect(self.eliminar)
        h_btns.addWidget(btn_reg)
        h_btns.addWidget(btn_del)
        c_lay.addLayout(h_btns)
        btn_exit = QPushButton("SAIR DA APLICAÇÃO")
        btn_exit.setStyleSheet("background-color: #1A1A1A; color: #888; border: 1px solid #333; margin-top: 10px;")
        btn_exit.clicked.connect(QApplication.instance().quit)
        c_lay.addWidget(btn_exit)
        layout.addWidget(container)

    def autenticar(self):
        user, pw = self.user_input.text(), self.pass_input.text()
        try:
            conn = sqlite3.connect(DB_NAME)
            res  = conn.execute("SELECT * FROM usuarios WHERE username=? AND password=?", (user, pw)).fetchone()
            conn.close()
            if res:
                self.login_sucesso.emit(user)
            else:
                QMessageBox.critical(self, "Erro", "Utilizador ou palavra-passe incorretos!")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Erro de Base de Dados", str(e))

    def registar(self):
        user, pw = self.user_input.text(), self.pass_input.text()
        if not user or not pw:
            QMessageBox.warning(self, "Aviso", "Preencha ambos os campos!"); return
        try:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", (user, pw))
            conn.commit(); conn.close()
            QMessageBox.information(self, "Sucesso", f"Utilizador '{user}' registado!")
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Erro", "Este utilizador já existe!")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Erro", str(e))

    def eliminar(self):
        user = self.user_input.text()
        if not user:
            QMessageBox.warning(self, "Aviso", "Introduza o nome do utilizador a eliminar."); return
        if QMessageBox.question(self, "Confirmar", f"Deseja eliminar '{user}'?") == QMessageBox.StandardButton.Yes:
            try:
                conn = sqlite3.connect(DB_NAME)
                conn.execute("DELETE FROM usuarios WHERE username=?", (user,))
                conn.commit(); conn.close()
                QMessageBox.information(self, "Sucesso", "Utilizador eliminado.")
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Erro", str(e))


class StatCard(QFrame):
    def __init__(self):
        super().__init__()
        self.lbl = None


# ==============================================================================
# APP PRINCIPAL
# ==============================================================================
class VisionProApp(QMainWindow):
    def __init__(self):
        super().__init__()

        if not os.path.exists(DIR_DATASET):
            os.makedirs(DIR_DATASET)

        self.inicializar_bd()
        self.falhas_manager       = FalhasManager(DB_NAME)
        self.operador_autenticado = "admin"

        self.peca_ativa_id   = None
        self.peca_ativa_nome = "Nenhum"
        self.inspecao_ativa  = False
        self.aba_ativa       = "INSPECAO"

        self.lista_poligonos = []
        self.ponto_atual     = []
        self.frame_fixo_treino             = None
        self.foto_referencia_processamento = None
        self.modelo_aprendido              = None
        self.total_ok                      = 0
        self.total_fail                    = 0
        self.ultimo_motivo_falha           = "Sem falhas registadas."
        self.falhas_por_modelo             = {}
        self.ultima_gravacao               = 0
        self.valor_threshold               = 100
        self.valor_sensibilidade           = 800
        self.clahe         = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.roi_inspector = ROIInspection(db_path=DB_NAME)
        self.historico_status = deque(maxlen=7)

        self.scale_factor = 1.0
        self.offset_x     = 0
        self.offset_y     = 0

        # ── MEDIÇÃO DE DISTÂNCIA 2D ──────────────────────────────────────────
        self.medicao_pontos        = []
        self.medicao_referencia_mm = 100.0
        self.medicao_px_por_mm     = None
        self.medicao_resultado_mm  = None
        self.medicao_historico     = deque(maxlen=8)
        self.medicao_calibrado     = False
        self.frame_fixo_medicao    = None

        # ── PROFUNDIDADE / ESPESSURA DE ESPUMA ──────────────────────────────
        # DepthMeasurer instanciado sempre; o modelo IA só carrega se
        # 'transformers' e 'torch' estiverem instalados.
        self.depth_measurer = DepthMeasurer(modelo="small",
                                            referencia_profundidade_mm=10.0)
        # Último resultado de espessura para mostrar no frame
        # Lista de (x, y, espessura_mm) calculados no frame COM espuma
        self.depth_espessuras_display = []

        self.setWindowTitle("VISION PRO | Terminal Industrial")
        self.resize(1400, 950)
        self._apply_style()
        self._init_ui()
        self._setup_shortcuts()

        self.cap   = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_loop)
        self.timer.start(16)

        self.carregar_dados_iniciais()

    # ──────────────────────────────────────────────────────────────────────────
    # ESTILOS
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0F1115; }
            QFrame#Sidebar    { background-color: #080808; border-right: 1px solid #1A1A1A; }
            QFrame#ControlBar { background-color: #16191E; border-top: 1px solid #2C313C; border-radius: 12px; }
            QFrame#StatCard   { background-color: #1E2227; border-radius: 10px; border: 1px solid #333; min-height: 100px; }
            QLabel { color: #ABB2BF; font-family: 'Segoe UI'; }
            QPushButton#NavBtn { background: transparent; border: none; color: #848da0;
                text-align: left; padding: 12px 20px; font-weight: bold; font-size: 13px; }
            QPushButton#NavBtn:hover { color: #FF8C00; background: #1A1A1A; }
            QPushButton#NavBtn[active="true"] { color: #FF8C00; background: #1A1A1A; border-left: 3px solid #FF8C00; }
            QPushButton#ModelBtn { background: #111; border: 1px solid #222; color: #888;
                padding: 10px; text-align: left; border-radius: 4px; margin-bottom: 2px; }
            QPushButton#ModelBtn:checked { border-color: #FF8C00; color: #FF8C00; background: #1A1A1A; }
            QPushButton#ControlBtn { background: #21252B; border: 1px solid #3E4451; color: white;
                padding: 10px 15px; border-radius: 6px; font-weight: bold; font-size: 11px; }
            QPushButton#ControlBtn:hover { border-color: #FF8C00; background: #2C313C; }
            QPushButton#DangerBtn { background: #2A1B1B; border: 1px solid #4A2626; color: #FF6666;
                padding: 10px 15px; border-radius: 6px; font-weight: bold; font-size: 11px; }
            QPushButton#DangerBtn:hover { background: #3D2222; border-color: #FF5555; }
            QPushButton#GreenBtn { background: #1A2E1A; border: 1px solid #2ECC71; color: #2ECC71;
                padding: 10px 15px; border-radius: 6px; font-weight: bold; font-size: 11px; }
            QPushButton#GreenBtn:hover { background: #223A22; border-color: #27AE60; }
            QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }
            QSlider::handle:horizontal { background: #FF8C00; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QLineEdit { background-color: #0F1115; border: 1px solid #3E4451; color: white; padding: 8px; border-radius: 4px; }
            QDoubleSpinBox { background-color: #0F1115; border: 1px solid #3E4451; color: white; padding: 6px; border-radius: 4px; }
        """)

    # ──────────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────────
    def _init_ui(self):
        self.main_stack = QStackedWidget()
        self.setCentralWidget(self.main_stack)

        self.login_screen = LoginScreen()
        self.login_screen.login_sucesso.connect(self.mostrar_app_principal)
        self.main_stack.addWidget(self.login_screen)

        self.app_container = QWidget()
        main_layout = QHBoxLayout(self.app_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.main_stack.addWidget(self.app_container)

        # ── SIDEBAR ──────────────────────────────────────────────────────────
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(260)
        side_v = QVBoxLayout(self.sidebar)

        logo = QLabel("VISION PRO")
        logo.setStyleSheet("color:#FF8C00; font-size:22px; font-weight:bold; margin:20px;")
        side_v.addWidget(logo)

        ai_status = "✓ IA Profundidade activa" if DEPTH_AI_DISPONIVEL else "⚠ Depth-AI não instalada"
        ai_color  = "#2ECC71" if DEPTH_AI_DISPONIVEL else "#E67E22"
        lbl_ai = QLabel(ai_status)
        lbl_ai.setStyleSheet(f"color:{ai_color}; font-size:11px; margin-left:20px; margin-bottom:5px;")
        side_v.addWidget(lbl_ai)

        self.nav_buttons = {}
        abas = [("INSPEÇÃO",     "INSPECAO",    0),
                ("TREINAR",      "TREINAR",     0),
                ("REFERÊNCIAS",  "REFERENCIAS", 0),
                ("IA / DATASET", "IA / DATASET",0),
                ("MEDIÇÃO",      "MEDICAO",     0),
                ("ESTATÍSTICAS", "ESTATISTICAS",1),
                ("COMANDOS",     "COMANDOS",    2),
                ("FALHAS",       "FALHAS",      3)]
        for nome, chave, idx in abas:
            btn = QPushButton(nome)
            btn.setObjectName("NavBtn")
            btn.clicked.connect(lambda chk=False, k=chave, i=idx: self.mudar_aba(k, i))
            side_v.addWidget(btn)
            self.nav_buttons[chave] = btn

        btn_logout = QPushButton("SAIR / BLOQUEAR")
        btn_logout.setObjectName("NavBtn")
        btn_logout.setStyleSheet("color: #FF5555;")
        btn_logout.clicked.connect(lambda: self.main_stack.setCurrentIndex(0))
        side_v.addSpacing(10)
        side_v.addWidget(btn_logout)
        side_v.addSpacing(25)
        side_v.addWidget(QLabel("MODELOS NA BD"), alignment=Qt.AlignmentFlag.AlignCenter)

        h_m = QHBoxLayout()
        b_add = QPushButton("+ NOVO");  b_add.setObjectName("ControlBtn"); b_add.clicked.connect(self.adicionar_novo_modelo)
        b_del = QPushButton("- APAGAR"); b_del.setObjectName("DangerBtn");  b_del.clicked.connect(self.eliminar_modelo_atual)
        h_m.addWidget(b_add); h_m.addWidget(b_del)
        side_v.addLayout(h_m)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:transparent; border:none;")
        self.model_container = QWidget()
        self.model_layout    = QVBoxLayout(self.model_container)
        self.model_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.model_group = QButtonGroup(self)
        self.model_group.setExclusive(True)
        scroll.setWidget(self.model_container)
        side_v.addWidget(scroll)
        side_v.addStretch()

        # ── ÁREA CENTRAL ─────────────────────────────────────────────────────
        self.pages = QStackedWidget()

        # Página 0 — Vídeo
        self.page_video = QWidget()
        video_v = QVBoxLayout(self.page_video)
        self.lbl_nome_peca_top = QLabel("...")
        self.lbl_nome_peca_top.setStyleSheet("font-size:18px; color:#FF8C00; font-weight:bold;")
        video_v.addWidget(self.lbl_nome_peca_top)
        self.video_container = QFrame()
        self.video_container.setStyleSheet("background:#000; border-radius:10px;")
        v_box = QVBoxLayout(self.video_container)
        self.video_label = VideoLabel()
        self.video_label.clique_esquerdo.connect(self.gerir_clique_esquerdo)
        self.video_label.clique_direito.connect(self.gerir_clique_direito)
        v_box.addWidget(self.video_label)
        video_v.addWidget(self.video_container, stretch=10)

        # Barra inferior dinâmica
        self.bottom_bar = QFrame()
        self.bottom_bar.setObjectName("ControlBar")
        self.bottom_bar.setFixedHeight(160)
        b_lay = QHBoxLayout(self.bottom_bar)
        self.ctrl_stack = QStackedWidget()

        # -- Controlos normais --
        ctrl_normal = QWidget()
        cn_lay = QHBoxLayout(ctrl_normal)
        draw_h = QHBoxLayout()
        btns_ctrl = [
            ("↺ VOLTAR (Z)",    self.retroceder_ponto,   "ControlBtn"),
            ("🗑️ LIMPAR (D)",   self.reset_poligonos_db, "DangerBtn"),
            ("💾 SALVAR (S)",   self.comando_salvar_s,   "ControlBtn"),
            ("▶ INSPEÇÃO (I)",  self.toggle_inspecao,    "ControlBtn"),
            ("📸 CAPTURAR (P)", self.capturar_para_ia,   "ControlBtn"),
            ("🔄 RESET (R)",    self.reset_contadores,   "ControlBtn"),
            ("✏ RENOMEAR (N)",  self.renomear_peca,      "ControlBtn"),
        ]
        for t, f, style in btns_ctrl:
            b = QPushButton(t); b.setObjectName(style); b.clicked.connect(f); draw_h.addWidget(b)
        cn_lay.addLayout(draw_h)
        self.ctrl_stack.addWidget(ctrl_normal)   # índice 0

        # -- Controlos de medição + espessura de espuma --
        ctrl_medicao = QWidget()
        cm_lay = QVBoxLayout(ctrl_medicao)
        cm_lay.setContentsMargins(6, 4, 6, 4)
        cm_lay.setSpacing(5)

        # Linha 1: parâmetros
        linha1 = QHBoxLayout()

        # Referência distância 2D
        col_dist = QVBoxLayout()
        col_dist.addWidget(QLabel("Ref. distância (mm):"))
        self.spin_ref_mm = QDoubleSpinBox()
        self.spin_ref_mm.setRange(1.0, 9999.0); self.spin_ref_mm.setValue(100.0)
        self.spin_ref_mm.setSuffix(" mm"); self.spin_ref_mm.setFixedWidth(110)
        self.spin_ref_mm.valueChanged.connect(lambda v: setattr(self, 'medicao_referencia_mm', v))
        col_dist.addWidget(self.spin_ref_mm)
        linha1.addLayout(col_dist)

        # Referência profundidade conhecida (para calibrar DepthMeasurer)
        col_prof = QVBoxLayout()
        col_prof.addWidget(QLabel("Ref. prof. conhecida (mm):"))
        self.spin_depth_ref = QDoubleSpinBox()
        self.spin_depth_ref.setRange(0.1, 999.0); self.spin_depth_ref.setValue(10.0)
        self.spin_depth_ref.setSuffix(" mm"); self.spin_depth_ref.setFixedWidth(130)
        self.spin_depth_ref.valueChanged.connect(
            lambda v: setattr(self.depth_measurer, 'referencia_profundidade_mm', v))
        col_prof.addWidget(self.spin_depth_ref)
        linha1.addLayout(col_prof)

        # Estado
        col_estado = QVBoxLayout()
        col_estado.addWidget(QLabel("Estado:"))
        self.lbl_medicao_estado = QLabel("Sem calibração")
        self.lbl_medicao_estado.setStyleSheet("color:#FF8C00; font-weight:bold; font-size:11px;")
        self.lbl_medicao_estado.setWordWrap(True)
        col_estado.addWidget(self.lbl_medicao_estado)
        linha1.addLayout(col_estado, stretch=2)
        cm_lay.addLayout(linha1)

        # Linha 2: botões
        linha2 = QHBoxLayout(); linha2.setSpacing(4)

        btn_congelar = QPushButton("❄ CONGELAR / PONTOS  (S)")
        btn_congelar.setObjectName("ControlBtn")
        btn_congelar.clicked.connect(self.medicao_congelar_frame)
        linha2.addWidget(btn_congelar)

        btn_cal_dist = QPushButton("📐 CALIBRAR DISTÂNCIA (C)")
        btn_cal_dist.setObjectName("ControlBtn")
        btn_cal_dist.clicked.connect(self.medicao_calibrar)
        linha2.addWidget(btn_cal_dist)

        # Calibrar profundidade — clique no ponto de profundidade conhecida
        self.btn_cal_prof = QPushButton("🔬 CALIBRAR PROF. (V)")
        self.btn_cal_prof.setObjectName("ControlBtn")
        self.btn_cal_prof.clicked.connect(self.depth_calibrar)
        self.btn_cal_prof.setEnabled(DEPTH_AI_DISPONIVEL)
        linha2.addWidget(self.btn_cal_prof)

        # Capturar referência SEM espuma
        self.btn_cap_ref = QPushButton("📦 CAPTURAR SEM ESPUMA (F)")
        self.btn_cap_ref.setObjectName("GreenBtn")
        self.btn_cap_ref.clicked.connect(self.depth_capturar_referencia)
        self.btn_cap_ref.setEnabled(DEPTH_AI_DISPONIVEL)
        linha2.addWidget(self.btn_cap_ref)

        # Medir espessura COM espuma
        self.btn_medir_esp = QPushButton("🧽 MEDIR ESPESSURA (E)")
        self.btn_medir_esp.setObjectName("ControlBtn")
        self.btn_medir_esp.clicked.connect(self.depth_medir_espessura)
        self.btn_medir_esp.setEnabled(DEPTH_AI_DISPONIVEL)
        linha2.addWidget(self.btn_medir_esp)

        btn_medir_cont = QPushButton("📏 MEDIR CONTÍNUO (M)")
        btn_medir_cont.setObjectName("ControlBtn")
        btn_medir_cont.clicked.connect(self.toggle_inspecao)
        linha2.addWidget(btn_medir_cont)

        btn_limpar = QPushButton("🗑 LIMPAR (D)")
        btn_limpar.setObjectName("DangerBtn")
        btn_limpar.clicked.connect(self.medicao_limpar_pontos)
        linha2.addWidget(btn_limpar)

        cm_lay.addLayout(linha2)
        self.ctrl_stack.addWidget(ctrl_medicao)  # índice 1

        b_lay.addWidget(self.ctrl_stack)
        video_v.addWidget(self.bottom_bar)

        # Página 1 — Estatísticas
        self.page_stats = QWidget()
        stats_v = QVBoxLayout(self.page_stats)
        stats_v.setContentsMargins(30, 30, 30, 30)
        self.lbl_stats_title = QLabel("ESTATÍSTICAS DO MODELO")
        self.lbl_stats_title.setStyleSheet("font-size:18px; color:#FF8C00; font-weight:bold;")
        stats_v.addWidget(self.lbl_stats_title)
        cards_h = QHBoxLayout()
        self.card_ok  = self._create_card("PEÇAS OK",  "#2ECC71")
        self.card_nok = self._create_card("PEÇAS NOK", "#E74C3C")
        cards_h.addWidget(self.card_ok); cards_h.addWidget(self.card_nok)
        stats_v.addLayout(cards_h)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor('#0F1115')
        self.canvas = FigureCanvas(self.fig)
        stats_v.addWidget(self.canvas)

        # Página 2 — Comandos
        self.page_cmds = QWidget()
        cmd_grid = QGridLayout(self.page_cmds)
        comandos_txt = [
            "[ S ] - SALVAR / CONGELAR",        "[ I ] - INICIAR / PARAR INSPEÇÃO",
            "[ Z ] - RETROCEDER PONTO",          "[ D ] - LIMPAR PONTOS / POLÍGONOS",
            "[ P ] - CAPTURAR PARA IA",          "[ R ] - RESETAR CONTADORES",
            "[ N ] - RENOMEAR MODELO",           "",
            "[ M ] - MEDIR CONTINUAMENTE",       "[ C ] - CALIBRAR DISTÂNCIA 2D",
            "[ V ] - CALIBRAR PROFUNDIDADE",     "[ F ] - CAPTURAR REFERÊNCIA SEM ESPUMA",
            "[ E ] - MEDIR ESPESSURA DE ESPUMA", "",
        ]
        for i, txt in enumerate(comandos_txt):
            btn = QPushButton(txt); btn.setObjectName("ControlBtn")
            btn.setMinimumHeight(60); cmd_grid.addWidget(btn, i // 2, i % 2)

        # Página 3 — Falhas
        self.page_falhas = QWidget()
        falhas_v = QVBoxLayout(self.page_falhas)
        falhas_v.setContentsMargins(30, 30, 30, 30)
        self.lbl_falhas_title = QLabel("FALHAS DO MODELO")
        self.lbl_falhas_title.setStyleSheet("font-size:18px; color:#FF8C00; font-weight:bold;")
        falhas_v.addWidget(self.lbl_falhas_title)
        falhas_v.addWidget(QLabel("Última falha detetada:"))
        self.lbl_ultimo_motivo = QLabel("Sem falhas registadas.")
        self.lbl_ultimo_motivo.setWordWrap(True)
        self.lbl_ultimo_motivo.setStyleSheet("font-size:16px; color:#E74C3C; font-weight:bold;")
        falhas_v.addWidget(self.lbl_ultimo_motivo)
        falhas_v.addSpacing(20)
        falhas_v.addWidget(QLabel("Resumo por motivo (modelo ativo):"))
        self.lbl_resumo_falhas = QLabel("Sem dados ainda.")
        self.lbl_resumo_falhas.setWordWrap(True)
        self.lbl_resumo_falhas.setStyleSheet("font-size:14px; color:#ABB2BF;")
        falhas_v.addWidget(self.lbl_resumo_falhas)
        falhas_v.addStretch()

        self.pages.addWidget(self.page_video)    # 0
        self.pages.addWidget(self.page_stats)    # 1
        self.pages.addWidget(self.page_cmds)     # 2
        self.pages.addWidget(self.page_falhas)   # 3

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.pages)

    def mostrar_app_principal(self, username):
        self.operador_autenticado = username
        self.main_stack.setCurrentIndex(1)
        self.login_screen.pass_input.clear()

    def _create_card(self, title, color):
        card = StatCard(); card.setObjectName("StatCard")
        v = QVBoxLayout(card)
        v.addWidget(QLabel(title), alignment=Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("0")
        lbl.setStyleSheet(f"font-size:28px; font-weight:bold; color:{color};")
        v.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)
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
        QShortcut(QKeySequence("C"), self, self.medicao_calibrar)
        QShortcut(QKeySequence("V"), self, self.depth_calibrar)
        QShortcut(QKeySequence("F"), self, self.depth_capturar_referencia)
        QShortcut(QKeySequence("E"), self, self.depth_medir_espessura)

    # ──────────────────────────────────────────────────────────────────────────
    # BASE DE DADOS
    # ──────────────────────────────────────────────────────────────────────────
    def inicializar_bd(self):
        try:
            conn   = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS pecas (
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
                bordas_ref     INTEGER DEFAULT 0,
                usuarios       INTEGER DEFAULT 0,
                px_por_mm      REAL,
                ref_mm         REAL,
                depth_escala   REAL,
                depth_ref_mm   REAL
            )''')
            for col, tipo in [("px_por_mm","REAL"),("ref_mm","REAL"),
                               ("depth_escala","REAL"),("depth_ref_mm","REAL")]:
                try:
                    cursor.execute(f"ALTER TABLE pecas ADD COLUMN {col} {tipo}")
                except sqlite3.OperationalError:
                    pass
            cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                username TEXT PRIMARY KEY, password TEXT)''')
            cursor.execute("SELECT COUNT(*) FROM usuarios")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO usuarios VALUES (?, ?)", ("admin", "1234"))
            conn.commit(); conn.close()
        except sqlite3.Error as e:
            print(f"ERRO BD: {e}")

    def carregar_dados_iniciais(self):
        self.atualizar_lista_modelos()
        conn = sqlite3.connect(DB_NAME)
        row  = conn.execute("SELECT id FROM pecas LIMIT 1").fetchone()
        conn.close()
        if row:
            self.carregar_dados_peca(row[0])

    def carregar_dados_peca(self, id_p):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pecas WHERE id=?", (id_p,))
        p = cursor.fetchone()
        conn.close()
        if not p: return
        self.peca_ativa_id   = p[0]
        self.peca_ativa_nome = p[1]
        self.roi_inspector.carregar_poligonos_peca(self.peca_ativa_id)
        self.lista_poligonos = self.roi_inspector.poligonos
        self.historico_status.clear()
        self.total_ok   = p[8] if p[8] is not None else 0
        self.total_fail = p[9] if p[9] is not None else 0
        self.ultimo_motivo_falha = "Sem falhas registadas."

        px_por_mm = p[12] if len(p) > 12 else None
        ref_mm    = p[13] if len(p) > 13 else None
        if px_por_mm and ref_mm:
            self.medicao_px_por_mm     = px_por_mm
            self.medicao_referencia_mm = ref_mm
            self.medicao_calibrado     = True
            self.spin_ref_mm.setValue(ref_mm)
        else:
            self.medicao_px_por_mm = None
            self.medicao_calibrado = False

        depth_escala = p[14] if len(p) > 14 else None
        depth_ref_mm = p[15] if len(p) > 15 else None
        if depth_escala and depth_ref_mm:
            self.depth_measurer.escala_px_para_mm           = depth_escala
            self.depth_measurer.referencia_profundidade_mm  = depth_ref_mm
            self.depth_measurer.calibrado                   = True
            self.spin_depth_ref.setValue(depth_ref_mm)
        else:
            self.depth_measurer.escala_px_para_mm = None
            self.depth_measurer.calibrado         = False

        self._atualizar_estado_medicao()

        pontos_json = p[3] if p[3] else "[]"
        try:
            pts = json.loads(pontos_json)
            self.medicao_pontos = [tuple(pt) for pt in pts] if pts else []
        except Exception:
            self.medicao_pontos = []

        if p[7]:
            nparr = np.frombuffer(p[7], np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.foto_referencia_processamento = img
            self.processar_referencia(img)
        else:
            self.foto_referencia_processamento = None
            self.modelo_aprendido = None

        self.atualizar_ui_labels()

    def _atualizar_estado_medicao(self):
        partes = []
        if self.medicao_calibrado and self.medicao_px_por_mm:
            partes.append(f"Dist: {self.medicao_px_por_mm:.2f} px/mm ✓")
        else:
            partes.append("Dist: ✗")
        if self.depth_measurer.calibrado:
            valor = struct.unpack('f', self.depth_measurer.escala_px_para_mm)[0]

            partes.append(f"Prof: {valor:.5f} mm/u ✓")
        else:
            partes.append("Prof: ✗")
        if self.depth_measurer.tem_referencia():
            partes.append("Ref espuma: ✓")
        else:
            partes.append("Ref espuma: ✗")
        self.lbl_medicao_estado.setText("  |  ".join(partes))

    def atualizar_ui_labels(self):
        txt = f"MODELO: {self.peca_ativa_nome} | ID: {self.peca_ativa_id}"
        self.lbl_nome_peca_top.setText(txt)
        self.lbl_stats_title.setText(f"ESTATÍSTICAS: {self.peca_ativa_nome}")
        self.lbl_falhas_title.setText(f"FALHAS: {self.peca_ativa_nome}")
        self.card_ok.lbl.setText(str(self.total_ok))
        self.card_nok.lbl.setText(str(self.total_fail))
        self.update_pie(self.total_ok, self.total_fail)
        self.atualizar_painel_falhas()

    def atualizar_painel_falhas(self):
        self.lbl_ultimo_motivo.setText(self.ultimo_motivo_falha)
        if self.peca_ativa_id is None:
            self.lbl_resumo_falhas.setText("Sem modelo selecionado."); return
        falhas_modelo = self.falhas_por_modelo.get(self.peca_ativa_id, {})
        if not falhas_modelo:
            self.lbl_resumo_falhas.setText("Sem dados ainda."); return
        linhas = [f"- {m}: {c}" for m, c in sorted(falhas_modelo.items(), key=lambda x: x[1], reverse=True)]
        self.lbl_resumo_falhas.setText("\n".join(linhas))

    def atualizar_lista_modelos(self):
        for i in reversed(range(self.model_layout.count())):
            item = self.model_layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                self.model_group.removeButton(w)
                w.setParent(None); w.deleteLater()
        conn = sqlite3.connect(DB_NAME)
        data = conn.execute("SELECT id, nome FROM pecas").fetchall()
        conn.close()
        for pid, nome in data:
            btn = QPushButton(f"📦 ID {pid:02} | {nome[:12]}")
            btn.setObjectName("ModelBtn"); btn.setCheckable(True)
            if pid == self.peca_ativa_id: btn.setChecked(True)
            btn.clicked.connect(lambda chk=False, x=pid: self.carregar_dados_peca(x))
            self.model_group.addButton(btn)
            self.model_layout.addWidget(btn)

    def update_pie(self, ok, nok):
        self.ax.clear()
        if ok == 0 and nok == 0:
            self.ax.text(0.5, 0.5, "Sem Dados", color="white", ha="center")
        else:
            self.ax.pie([ok, nok], labels=["OK","NOK"], colors=["#2ECC71","#E74C3C"],
                        autopct='%1.1f%%', pctdistance=0.78, labeldistance=1.05, startangle=90,
                        textprops={'color':'white','fontsize':10},
                        wedgeprops={'width':0.4,'edgecolor':'#0F1115'})
            self.ax.text(0, 0, f"TOTAL\n{ok+nok}", color='white', ha='center', va='center', fontweight='bold')
        self.ax.set_aspect('equal'); self.canvas.draw()

    def salvar_progresso_bd(self):
        if self.peca_ativa_id is None: return
        conn = sqlite3.connect(DB_NAME)
        conn.execute("UPDATE pecas SET poligonos=?, aprovadas=?, rejeitadas=? WHERE id=?",
                     (json.dumps(self.lista_poligonos), self.total_ok, self.total_fail, self.peca_ativa_id))
        conn.commit(); conn.close()

    # ──────────────────────────────────────────────────────────────────────────
    # MEDIÇÃO DE DISTÂNCIA 2D
    # ──────────────────────────────────────────────────────────────────────────
    def medicao_congelar_frame(self):
        if self.frame_fixo_medicao is None:
            ret, f = self.cap.read()
            if ret:
                self.frame_fixo_medicao = cv2.flip(f, 1)
                self.medicao_pontos     = []
                # Gera mapa de profundidade para este frame (apenas uma vez)
                if DEPTH_AI_DISPONIVEL:
                    self.lbl_medicao_estado.setText("A gerar mapa de profundidade...")
                    QApplication.processEvents()
                    self.depth_measurer.gerar_mapa(self.frame_fixo_medicao)
                self.lbl_medicao_estado.setText("Frame congelado. Clica nos pontos.")
        else:
            self.frame_fixo_medicao = None
            self.lbl_medicao_estado.setText("Descongelado.")

    def medicao_calibrar(self):
        if len(self.medicao_pontos) < 2:
            QMessageBox.warning(self, "Medição", "Define 2 pontos primeiro."); return
        p1, p2  = self.medicao_pontos[0], self.medicao_pontos[1]
        dist_px = np.linalg.norm(np.array(p1) - np.array(p2))
        if dist_px < 5:
            QMessageBox.warning(self, "Medição", "Pontos demasiado próximos."); return
        self.medicao_px_por_mm = dist_px / self.medicao_referencia_mm
        self.medicao_calibrado = True
        self.medicao_historico.clear()
        if self.peca_ativa_id is not None:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET px_por_mm=?, ref_mm=?, pontos_medicao=? WHERE id=?",
                         (self.medicao_px_por_mm, self.medicao_referencia_mm,
                          json.dumps([list(p) for p in self.medicao_pontos]), self.peca_ativa_id))
            conn.commit(); conn.close()
        self.frame_fixo_medicao = None
        self._atualizar_estado_medicao()

    def medicao_limpar_pontos(self):
        self.medicao_pontos           = []
        self.frame_fixo_medicao       = None
        self.medicao_resultado_mm     = None
        self.depth_espessuras_display = []
        self.depth_measurer.limpar_medicoes()
        self._atualizar_estado_medicao()

    def medicao_calcular_distancia(self, frame):
        if not self.medicao_calibrado or len(self.medicao_pontos) < 2:
            cv2.putText(frame, "Congela (S) → define 2 pontos → calibra (C)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
            return None, frame
        p1 = tuple(map(int, self.medicao_pontos[0]))
        p2 = tuple(map(int, self.medicao_pontos[1]))
        dist_px = np.linalg.norm(np.array(p1) - np.array(p2))
        dist_mm = dist_px / self.medicao_px_por_mm
        self.medicao_historico.append(dist_mm)
        dist_suav = float(np.mean(self.medicao_historico))
        cv2.line(frame, p1, p2, (0, 220, 255), 2)
        cv2.circle(frame, p1, 7, (0, 140, 255), -1)
        cv2.putText(frame, "P1", (p1[0]+10, p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        cv2.circle(frame, p2, 7, (255, 220, 0), -1)
        cv2.putText(frame, "P2", (p2[0]+10, p2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        cv2.rectangle(frame, (mid[0]-5, mid[1]-25), (mid[0]+200, mid[1]+8), (20,20,20), -1)
        cv2.putText(frame, f"{dist_suav:.2f} mm", (mid[0], mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.rectangle(frame, (10,10), (370,75), (20,20,20), -1)
        cv2.rectangle(frame, (10,10), (370,75), (60,60,60), 1)
        cv2.putText(frame, f"DISTANCIA: {dist_suav:.2f} mm", (18,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,255), 2)
        cv2.putText(frame, f"Escala: {self.medicao_px_por_mm:.3f} px/mm | ref: {self.medicao_referencia_mm}mm",
                    (18,66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
        return dist_suav, frame

    # ──────────────────────────────────────────────────────────────────────────
    # PROFUNDIDADE / ESPESSURA DE ESPUMA
    # ──────────────────────────────────────────────────────────────────────────
    def depth_calibrar(self):
        """
        Calibra o DepthMeasurer num ponto de profundidade conhecida.
        Deve ser chamado após congelar o frame SEM espuma (mapa já gerado).
        Usa o último ponto clicado pelo utilizador.
        """
        if not DEPTH_AI_DISPONIVEL:
            QMessageBox.warning(self, "Profundidade", "Instala 'transformers' e 'torch' primeiro."); return
        if self.depth_measurer.mapa_profundidade is None:
            QMessageBox.warning(self, "Profundidade",
                                "Congela primeiro o frame SEM espuma (S) para gerar o mapa."); return
        if not self.medicao_pontos:
            QMessageBox.warning(self, "Profundidade",
                                "Clica num ponto de profundidade conhecida no frame congelado."); return
        px, py  = self.medicao_pontos[-1]
        ref_mm  = self.spin_depth_ref.value()
        sucesso = self.depth_measurer.calibrar(px, py, profundidade_conhecida_mm=ref_mm)
        if sucesso:
            if self.peca_ativa_id is not None:
                conn = sqlite3.connect(DB_NAME)
                conn.execute("UPDATE pecas SET depth_escala=?, depth_ref_mm=? WHERE id=?",
                             (self.depth_measurer.escala_px_para_mm, ref_mm, self.peca_ativa_id))
                conn.commit(); conn.close()
            self._atualizar_estado_medicao()
        else:
            QMessageBox.warning(self, "Profundidade", "Calibração falhou — ponto inválido no mapa.")

    def depth_capturar_referencia(self):
        """
        Guarda o mapa actual como referência SEM espuma.
        Deve ser chamado com o frame congelado e o mapa gerado.
        """
        if not DEPTH_AI_DISPONIVEL:
            QMessageBox.warning(self, "Espuma", "IA não disponível."); return
        if not self.depth_measurer.calibrado:
            QMessageBox.warning(self, "Espuma", "Calibra a profundidade primeiro (V)."); return
        if self.depth_measurer.mapa_profundidade is None:
            QMessageBox.warning(self, "Espuma",
                                "Congela o frame SEM espuma (S) para gerar o mapa primeiro."); return
        sucesso = self.depth_measurer.capturar_referencia(pontos=self.medicao_pontos or None)
        if sucesso:
            self.frame_fixo_medicao = None   # descongela
            self._atualizar_estado_medicao()
            QMessageBox.information(self, "Referência", "✓ Referência SEM espuma guardada!\n"
                                    "Agora coloca a espuma e clica em 'MEDIR ESPESSURA (E)'.")
        else:
            QMessageBox.warning(self, "Espuma", "Não foi possível guardar a referência.")

    def depth_medir_espessura(self):
        """
        Congela o frame COM espuma, gera o mapa e calcula a espessura
        em todos os pontos definidos pelo utilizador.
        Mostra os resultados numa janela e no vídeo.
        """
        if not DEPTH_AI_DISPONIVEL:
            QMessageBox.warning(self, "Espuma", "IA não disponível."); return
        if not self.depth_measurer.tem_referencia():
            QMessageBox.warning(self, "Espuma",
                                "Ainda não capturaste a referência SEM espuma (F)."); return
        if not self.depth_measurer.calibrado:
            QMessageBox.warning(self, "Espuma", "Calibra a profundidade primeiro (V)."); return

        # Congela o frame actual COM espuma e gera o mapa
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Espuma", "Não foi possível capturar o frame."); return
        frame = cv2.flip(frame, 1)

        self.lbl_medicao_estado.setText("A calcular mapa COM espuma...")
        QApplication.processEvents()
        self.depth_measurer.gerar_mapa(frame)

        # Define pontos a medir — usa os pontos guardados ou uma grelha central
        if self.medicao_pontos:
            pontos_para_medir = self.medicao_pontos
        else:
            # Grelha 3×3 centrada no frame se não houver pontos definidos
            h, w = frame.shape[:2]
            pontos_para_medir = [
                (int(w * x), int(h * y))
                for x in [0.25, 0.5, 0.75]
                for y in [0.25, 0.5, 0.75]
            ]

        resultado = self.depth_measurer.medir_espessura_zona(pontos_para_medir)
        if resultado is None:
            QMessageBox.warning(self, "Espuma", "Nenhuma medição válida nos pontos definidos."); return

        # Actualiza lista para mostrar no vídeo
        self.depth_espessuras_display = [
            (r["ponto"][0], r["ponto"][1], r["espessura_mm"])
            for r in resultado["pontos"]
        ]

        msg = (f"ESPESSURA DE ESPUMA\n"
               f"{'─'*35}\n"
               f"Média   : {resultado['media_mm']:+.2f} mm\n"
               f"Máximo  : {resultado['max_mm']:+.2f} mm\n"
               f"Mínimo  : {resultado['min_mm']:+.2f} mm\n"
               f"Desvio  : {resultado['desvio_mm']:.2f} mm\n"
               f"Pontos  : {len(resultado['pontos'])}\n\n"
               f"(+) = espuma mais perto da câmara (normal)\n"
               f"(-) = espuma afasta a peça (verificar)")
        QMessageBox.information(self, "Resultado — Espessura de Espuma", msg)
        self._atualizar_estado_medicao()

    # ──────────────────────────────────────────────────────────────────────────
    # VISÃO COMPUTACIONAL
    # ──────────────────────────────────────────────────────────────────────────
    def processar_referencia(self, img):
        if img is None: return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe_img = self.clahe.apply(blur)
        _, thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = res[0] if len(res) == 2 else res[1]
        if contornos: self.modelo_aprendido = max(contornos, key=cv2.contourArea)

    def executar_inspecao(self, frame):
        if not self.roi_inspector.poligonos:
            return "AGUARDAR", "Sem ROIs definidas para o modelo ativo", {}
        resultado_roi = self.roi_inspector.processar_frame_completo(frame)
        status_final  = resultado_roi.get("status_global", "AGUARDAR")
        motivo_fail   = resultado_roi.get("motivo", "")
        frame_com_rois = self.roi_inspector.desenhar_rois_em_frame(frame)
        frame[:] = frame_com_rois
        for idx, dados in resultado_roi.get("rois_dados", {}).items():
            cor = (0,200,0) if dados.get("status")=="OK" else ((0,200,255) if dados.get("status")=="AGUARDAR" else (0,0,255))
            pts = None
            if self.roi_inspector.poligonos:
                try:
                    roi_num = int(idx.split("_")[1])
                    if roi_num < len(self.roi_inspector.poligonos):
                        pts = np.array(self.roi_inspector.poligonos[roi_num], dtype=np.int32)
                except (ValueError, IndexError): pts = None
            if pts is not None and len(pts) > 2:
                cv2.polylines(frame, [pts], True, cor, 2)
        if status_final != "OK":
            return status_final, motivo_fail, resultado_roi
        mascara_uniao_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
        for dados in resultado_roi.get("rois_dados", {}).values():
            mascara = dados.get("mascara")
            if mascara is not None:
                mascara_uniao_roi = cv2.bitwise_or(mascara_uniao_roi, mascara)
        hsv = cv2.cvtColor(cv2.bilateralFilter(frame, 9, 75, 75), cv2.COLOR_BGR2HSV)
        objetos = []
        for nome_cor, (low, high, _) in CORES_DETECCAO.items():
            m = cv2.inRange(hsv, low, high)
            m = cv2.bitwise_and(m, m, mask=mascara_uniao_roi)
            res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = res[0] if len(res) == 2 else res[1]
            for c in cnts:
                if cv2.contourArea(c) > 1000:
                    objetos.append((cv2.contourArea(c), c, nome_cor))
        if not objetos:
            return "AGUARDAR", "Sem peça na zona ROI", resultado_roi
        objetos.sort(key=lambda x: x[0], reverse=True)
        _, c_prin, cor_prin = objetos[0]
        if self.modelo_aprendido is not None and cv2.matchShapes(self.modelo_aprendido, c_prin, 1, 0.0) > 0.25:
            status_final = "FAIL"; motivo_fail = "Forma Errada"
        cor_rect = (0,255,0) if status_final == "OK" else (0,0,255)
        cv2.drawContours(frame, [c_prin], -1, cor_rect, 2)
        cv2.putText(frame, f"{cor_prin} {motivo_fail}".strip(),
                    (c_prin[0][0][0], c_prin[0][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_rect, 2)
        return status_final, motivo_fail, resultado_roi

    def estabilizar_status(self, status):
        if status == "AGUARDAR":
            self.historico_status.clear(); return status
        self.historico_status.append(status)
        if len(self.historico_status) < 3: return status
        return Counter(self.historico_status).most_common(1)[0][0]

    # ──────────────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def atualizar_loop(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        display_frame = frame.copy()

        if self.aba_ativa == "TREINAR":
            if self.frame_fixo_treino is not None:
                display_frame = self.frame_fixo_treino.copy()
                for poly in self.lista_poligonos:
                    cv2.polylines(display_frame, [np.array(poly)], True, (100,100,100), 1)
                for pt in self.ponto_atual:
                    cv2.circle(display_frame, pt, 4, (0,255,255), -1)
                if len(self.ponto_atual) > 1:
                    cv2.polylines(display_frame, [np.array(self.ponto_atual)], False, (0,255,255), 2)
                cv2.putText(display_frame, "MODO TREINO: CONGELADO", (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(display_frame, "PRESSIONE [S] PARA CONGELAR", (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        elif self.aba_ativa == "INSPECAO":
            if self.inspecao_ativa:
                status_bruto, motivo_fail, resultado_roi = self.executar_inspecao(display_frame)
                status     = self.estabilizar_status(status_bruto)
                cor_status = (0,255,255) if status=="AGUARDAR" else ((0,255,0) if status=="OK" else (0,0,255))
                cv2.putText(display_frame, f"STATUS: {status}", (w-300,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, cor_status, 3)
                if motivo_fail:
                    cv2.putText(display_frame, motivo_fail, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)
                if resultado_roi and resultado_roi.get("rois_dados"):
                    y = 70
                    for nome_roi, dados in resultado_roi["rois_dados"].items():
                        txt_roi = f"{nome_roi}: {dados.get('status','N/A')} | Fill={dados.get('fill_ratio',0):.3f}"
                        cv2.putText(display_frame, txt_roi, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
                        y += 18
                if time.time() - self.ultima_gravacao > 1.2:
                    if status == "OK":
                        self.total_ok += 1
                        if self.peca_ativa_id is not None:
                            self.falhas_manager.registar_falha(self.peca_ativa_id,
                                self.operador_autenticado, "OK", "---", display_frame)
                    elif status != "AGUARDAR":
                        self.total_fail += 1
                        self.ultimo_motivo_falha = motivo_fail or "Falha sem motivo identificado"
                        if self.peca_ativa_id is not None:
                            self.falhas_manager.registar_falha(self.peca_ativa_id,
                                self.operador_autenticado, "FAIL", self.ultimo_motivo_falha, display_frame)
                            fm = self.falhas_por_modelo.setdefault(self.peca_ativa_id, {})
                            fm[self.ultimo_motivo_falha] = fm.get(self.ultimo_motivo_falha, 0) + 1
                    self.ultima_gravacao = time.time()
                    self.salvar_progresso_bd()
                    self.atualizar_ui_labels()
            else:
                cv2.putText(display_frame, "INSPECAO PAUSADA [I]", (w//2-150, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        elif self.aba_ativa == "REFERENCIAS":
            if self.foto_referencia_processamento is not None:
                display_frame = self.foto_referencia_processamento.copy()
                cv2.putText(display_frame, "REFERENCIA ATUAL", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,140,0), 2)
            else:
                display_frame = np.zeros_like(frame)
                cv2.putText(display_frame, "SEM FOTO DE REF", (w//2-100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)

        elif self.aba_ativa == "IA / DATASET":
            imgs = [f for f in os.listdir(DIR_DATASET) if f.startswith(f"peca_{self.peca_ativa_id}_")]
            cv2.putText(display_frame, f"DATASET: {len(imgs)} IMAGENS", (30,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        elif self.aba_ativa == "MEDICAO":
            if self.frame_fixo_medicao is not None:
                # ── Frame congelado ───────────────────────────────────────────
                display_frame = self.frame_fixo_medicao.copy()
                cv2.putText(display_frame, "CONGELADO — clica nos pontos  |  S=descongelar",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cores_pts = [(0,140,255),(255,220,0)]
                for i, pt in enumerate(self.medicao_pontos[:2]):
                    cv2.circle(display_frame, tuple(map(int,pt)), 6, cores_pts[i], -1)
                    cv2.putText(display_frame, f"P{i+1}", (int(pt[0])+12, int(pt[1])-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cores_pts[i], 2)
                if len(self.medicao_pontos) == 2:
                    p1 = tuple(map(int, self.medicao_pontos[0]))
                    p2 = tuple(map(int, self.medicao_pontos[1]))
                    cv2.line(display_frame, p1, p2, (0,220,255), 2)
                    dist_px = np.linalg.norm(np.array(p1)-np.array(p2))
                    cv2.putText(display_frame,
                                f"{dist_px:.0f} px — [C] calibra dist  [V] calibra prof  [F] guarda ref s/espuma",
                                (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                # Anotação de profundidade (sem mapa colorido)
                if DEPTH_AI_DISPONIVEL:
                    display_frame = self.depth_measurer.anotar_frame(display_frame)

            else:
                # ── Modo ao vivo ──────────────────────────────────────────────
                if self.inspecao_ativa:
                    # Medição 2D contínua
                    _, display_frame = self.medicao_calcular_distancia(display_frame)
                    # Redesenha pontos fixos
                    cores_pts = [(0,140,255),(255,220,0)]
                    for i, pt in enumerate(self.medicao_pontos[:2]):
                        cv2.circle(display_frame, tuple(map(int,pt)), 6, cores_pts[i], -1)
                else:
                    # Parado — mostra pontos e último resultado de espessura
                    for i, pt in enumerate(self.medicao_pontos[:2]):
                        cv2.circle(display_frame, tuple(map(int,pt)), 6,
                                   [(0,140,255),(255,220,0)][i], -1)

                # Mostra espessuras calculadas (persistem até nova medição)
                if self.depth_espessuras_display:
                    display_frame = self.depth_measurer.anotar_frame(
                        display_frame,
                        pontos_espessura=self.depth_espessuras_display)

                # Instrução se não há nada activo
                if not self.inspecao_ativa and not self.depth_espessuras_display:
                    cv2.putText(display_frame,
                                "[S] Congelar  [M] Medir contínuo  [E] Medir espessura espuma",
                                (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,120), 1)

        self.exibir_frame(display_frame)

    def exibir_frame(self, frame):
        w_lbl = self.video_container.width()  - 20
        h_lbl = self.video_container.height() - 20
        if w_lbl <= 0 or h_lbl <= 0: return
        h_img, w_img  = frame.shape[:2]
        self.scale_factor = min(w_lbl/w_img, h_lbl/h_img)
        new_w = int(w_img * self.scale_factor)
        new_h = int(h_img * self.scale_factor)
        self.offset_x = (self.video_label.width()  - new_w) / 2
        self.offset_y = (self.video_label.height() - new_h) / 2
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w_img, h_img, w_img*3, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(new_w, new_h,
               Qt.AspectRatioMode.KeepAspectRatio,
               Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pix)

    # ──────────────────────────────────────────────────────────────────────────
    # CLIQUES
    # ──────────────────────────────────────────────────────────────────────────
    def gerir_clique_esquerdo(self, x, y):
        real_x = int((x - self.offset_x) / self.scale_factor)
        real_y = int((y - self.offset_y) / self.scale_factor)

        if self.aba_ativa == "TREINAR" and self.frame_fixo_treino is not None:
            h, w = self.frame_fixo_treino.shape[:2]
            if 0 <= real_x < w and 0 <= real_y < h:
                self.ponto_atual.append((real_x, real_y))

        elif self.aba_ativa == "MEDICAO" and self.frame_fixo_medicao is not None:
            h, w = self.frame_fixo_medicao.shape[:2]
            if 0 <= real_x < w and 0 <= real_y < h:
                if len(self.medicao_pontos) >= 2:
                    self.medicao_pontos = []
                self.medicao_pontos.append((real_x, real_y))
                n = len(self.medicao_pontos)
                msg = f"P{n} em ({real_x}, {real_y})."
                # Se IA disponível e calibrada, mostra profundidade do ponto
                if DEPTH_AI_DISPONIVEL and self.depth_measurer.calibrado:
                    prof = self.depth_measurer.medir_ponto(real_x, real_y)
                    if prof is not None:
                        msg += f" Prof: {prof:.1f}mm"
                self.lbl_medicao_estado.setText(
                    msg + (" Clica P2." if n == 1 else " [C]=dist  [V]=prof  [F]=ref espuma"))

    def gerir_clique_direito(self, _x, _y):
        if self.aba_ativa == "MEDICAO":
            self.medicao_pontos = []
            self.depth_measurer.pontos_medidos        = []
            self.depth_measurer.profundidades_pontos  = []
            self.lbl_medicao_estado.setText("Pontos apagados. Clica P1.")
        else:
            self.ponto_atual = []

    # ──────────────────────────────────────────────────────────────────────────
    # COMANDOS
    # ──────────────────────────────────────────────────────────────────────────
    def mudar_aba(self, chave, idx):
        self.aba_ativa = chave
        self.pages.setCurrentIndex(idx)
        self.frame_fixo_treino = None
        self.ponto_atual       = []
        self.ctrl_stack.setCurrentIndex(1 if chave == "MEDICAO" else 0)
        for k, btn in self.nav_buttons.items():
            btn.setProperty("active", k == chave)
            btn.style().unpolish(btn); btn.style().polish(btn)

    def toggle_inspecao(self):
        self.inspecao_ativa = not self.inspecao_ativa
        self.historico_status.clear()

    def comando_salvar_s(self):
        if self.aba_ativa == "MEDICAO":
            self.medicao_congelar_frame(); return
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro."); return
        if self.aba_ativa == "TREINAR":
            if self.frame_fixo_treino is None:
                ret, f = self.cap.read()
                if ret: self.frame_fixo_treino = cv2.flip(f, 1)
                return
            if len(self.ponto_atual) > 2:
                self.lista_poligonos.append(list(self.ponto_atual))
                self.ponto_atual = []
                self.salvar_progresso_bd()
                self.roi_inspector.carregar_poligonos_peca(self.peca_ativa_id)
                self.lista_poligonos = self.roi_inspector.poligonos
            else:
                resp = QMessageBox.question(self, "Referência",
                    "Deseja salvar esta imagem como REFERÊNCIA ou apenas DESCONGELAR?\n\n"
                    "(Sim = Salvar Ref | Não = Descongelar | Cancelar = Nada)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                if resp == QMessageBox.StandardButton.Yes:
                    conn = sqlite3.connect(DB_NAME)
                    _, buf = cv2.imencode(".jpg", self.frame_fixo_treino)
                    conn.execute("UPDATE pecas SET foto_ref=? WHERE id=?", (buf.tobytes(), self.peca_ativa_id))
                    conn.commit(); conn.close()
                    self.foto_referencia_processamento = self.frame_fixo_treino.copy()
                    self.processar_referencia(self.frame_fixo_treino)
                    self.frame_fixo_treino = None
                elif resp == QMessageBox.StandardButton.No:
                    self.frame_fixo_treino = None; self.ponto_atual = []

    def reset_poligonos_db(self):
        if self.aba_ativa == "MEDICAO":
            self.medicao_limpar_pontos(); return
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro."); return
        if QMessageBox.question(self, "Limpar", "Apagar desenhos do modelo?") == QMessageBox.Yes:
            self.lista_poligonos = []; self.ponto_atual = []
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET poligonos=? WHERE id=?", (json.dumps([]), self.peca_ativa_id))
            conn.commit(); conn.close()
            self.roi_inspector.carregar_poligonos_peca(self.peca_ativa_id)
            self.lista_poligonos = self.roi_inspector.poligonos

    def retroceder_ponto(self):
        if self.aba_ativa == "MEDICAO":
            if self.medicao_pontos: self.medicao_pontos.pop()
        elif self.ponto_atual:
            self.ponto_atual.pop()
        elif self.lista_poligonos:
            self.lista_poligonos.pop()

    def capturar_para_ia(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro."); return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            path  = os.path.join(DIR_DATASET, f"peca_{self.peca_ativa_id}_{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, frame)

    def reset_contadores(self):
        self.total_ok = self.total_fail = 0
        self.salvar_progresso_bd(); self.atualizar_ui_labels()

    def renomear_peca(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro."); return
        novo, ok = QInputDialog.getText(self, "Renomear", "Novo nome:", text=self.peca_ativa_nome)
        if ok and novo:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET nome=? WHERE id=?", (novo, self.peca_ativa_id))
            conn.commit(); conn.close()
            self.peca_ativa_nome = novo
            self.atualizar_lista_modelos(); self.atualizar_ui_labels()

    def adicionar_novo_modelo(self):
        conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM pecas")
        res = cursor.fetchone()[0]; nxt = (res+1) if res else 1
        cursor.execute("INSERT INTO pecas (id, nome, poligonos) VALUES (?, ?, ?)",
                       (nxt, f"Modelo {nxt:02d}", "[]"))
        conn.commit(); conn.close(); self.atualizar_lista_modelos()

    def eliminar_modelo_atual(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Nenhum modelo selecionado."); return
        if QMessageBox.question(self, "Eliminar", f"Eliminar {self.peca_ativa_nome}?") == QMessageBox.Yes:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("DELETE FROM pecas WHERE id=?", (self.peca_ativa_id,))
            conn.commit(); conn.close(); self.carregar_dados_iniciais()


if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = VisionProApp()
    window.showFullScreen()
    sys.exit(app.exec())

    # trocar a inspeção da peça pela medição da profundidade
    # trocar o botão de inspeção que (I) esta a dar conflito com o botao de medição (M)
    # verificar se a medição funciona diretio em casa
    # ajustar/usar a medição_espuma.py para a medição da profundidade e melhorar depois
    # ajustar a posição dos botoes na aba de medição
    # remover o menu de falhas
    # remover a escala de px para mm

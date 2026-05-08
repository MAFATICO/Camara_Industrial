import sys
import cv2
import numpy as np
import os
import sqlite3
import json
import time
from collections import Counter, deque

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from COLOR_DETECTION.roi_inspection import ROIInspection

# --- CONFIGURAÇÃO DE BACKEND MATPLOTLIB ---
os.environ["QT_API"] = "pyside6"
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFrame,
                               QStackedWidget, QSlider, QScrollArea, QGridLayout,
                               QButtonGroup, QInputDialog, QMessageBox, QLineEdit)

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "COLOR_DETECTION/sistema_industrial.db")
DIR_DATASET = os.path.join(BASE_DIR, "COLOR_DETECTION/dataset_ia")

# Garante que o diretório da base de dados existe
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
CORES_DETECCAO = {
    "AZUL": (np.array([85, 100, 50]), np.array([140, 255, 255]), (255, 0, 0)),
    "PRETO": (np.array([0, 0, 0]), np.array([150, 50, 50]), (60, 60, 60)) #array da cor preta
}


# ==============================================================================
# COMPONENTE DE VÍDEO PERSONALIZADO (Para capturar cliques)
# ==============================================================================
class VideoLabel(QLabel):
    clique_esquerdo = Signal(int, int)
    clique_direito = Signal(int, int)

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
# GESTÃO DE UTILIZADORES E ECRÃ DE LOGIN
# ==============================================================================
class LoginScreen(QWidget):
    login_sucesso = Signal()

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
            QFrame { 
                background-color: #16191E; 
                border-radius: 15px; 
                border: 1px solid #2C313C; 
                padding: 20px;
            }
            QLineEdit {
                background-color: #0F1115;
                border: 1px solid #3E4451;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            QPushButton {
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
            }
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
        user = self.user_input.text()
        pw = self.pass_input.text()
        
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM usuarios WHERE username=? AND password=?", (user, pw))
            res = cursor.fetchone()
            conn.close()

            if res:
                self.login_sucesso.emit()
            else:
                QMessageBox.critical(self, "Erro", "Utilizador ou palavra-passe incorretos!")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Erro de Base de Dados", f"Erro ao autenticar: {e}")

    def registar(self):
        user = self.user_input.text()
        pw = self.pass_input.text()
        if not user or not pw:
            QMessageBox.warning(self, "Aviso", "Preencha ambos os campos!")
            return

        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", (user, pw))
            conn.commit()
            QMessageBox.information(self, "Sucesso", f"Utilizador '{user}' registado!")
            conn.close()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Erro", "Este utilizador já existe!")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Erro", f"Erro ao registar: {e}")

    def eliminar(self):
        user = self.user_input.text()
        if not user:
            QMessageBox.warning(self, "Aviso", "Introduza o nome do utilizador a eliminar.")
            return
        
        confirm = QMessageBox.question(self, "Confirmar", f"Deseja eliminar o utilizador '{user}'?")
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM usuarios WHERE username=?", (user,))
                if cursor.rowcount > 0:
                    conn.commit()
                    QMessageBox.information(self, "Sucesso", "Utilizador eliminado.")
                else:
                    QMessageBox.warning(self, "Erro", "Utilizador não encontrado.")
                conn.close()
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Erro", f"Erro ao eliminar: {e}")


# ==============================================================================
# COMPONENTES AUXILIARES
# ==============================================================================
class StatCard(QFrame):
    def __init__(self):
        super().__init__()
        self.lbl = None

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
        self.ultimo_motivo_falha = "Sem falhas registadas."
        self.falhas_por_modelo = {}
        self.ultima_gravacao = 0
        self.valor_threshold = 100
        self.valor_sensibilidade = 800
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.roi_inspector = ROIInspection(db_path=DB_NAME)
        self.historico_status = deque(maxlen=7)

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

            QLineEdit {
                background-color: #0F1115;
                border: 1px solid #3E4451;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
        """)

    def _init_ui(self):
        # Widget Central que contém o Stack Principal (Login vs App)
        self.main_stack = QStackedWidget()
        self.setCentralWidget(self.main_stack)

        # 1. ECRÃ DE LOGIN
        self.login_screen = LoginScreen()
        self.login_screen.login_sucesso.connect(self.mostrar_app_principal)
        self.main_stack.addWidget(self.login_screen)

        # 2. APP PRINCIPAL
        self.app_container = QWidget()
        main_layout = QHBoxLayout(self.app_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.main_stack.addWidget(self.app_container)

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
                ("COMANDOS", "COMANDOS", 2),
                ("FALHAS", "FALHAS", 3)]

        for nome, chave, idx in abas:
            btn = QPushButton(nome)
            btn.setObjectName("NavBtn")
            btn.clicked.connect(lambda chk=False, k=chave, i=idx: self.mudar_aba(k, i))
            side_v.addWidget(btn)
            self.nav_buttons[chave] = btn
        
        # Botão Logout
        btn_logout = QPushButton("SAIR / BLOQUEAR")
        btn_logout.setObjectName("NavBtn")
        btn_logout.setStyleSheet("color: #FF5555;")
        btn_logout.clicked.connect(lambda: self.main_stack.setCurrentIndex(0))
        side_v.addSpacing(10)
        side_v.addWidget(btn_logout)

        side_v.addSpacing(25)
        side_v.addWidget(QLabel("MODELOS NA BD"), alignment=Qt.AlignmentFlag.AlignCenter)

        h_m = QHBoxLayout()
        b_add = QPushButton("+ NOVO")
        b_add.setObjectName("ControlBtn")
        b_add.clicked.connect(self.adicionar_novo_modelo)
        b_del = QPushButton("- APAGAR")
        b_del.setObjectName("DangerBtn")
        b_del.clicked.connect(self.eliminar_modelo_atual)
        h_m.addWidget(b_add)
        h_m.addWidget(b_del)
        side_v.addLayout(h_m)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:transparent; border:none;")
        self.model_container = QWidget()
        self.model_layout = QVBoxLayout(self.model_container)
        self.model_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.model_group = QButtonGroup(self)
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

        self.video_container = QFrame()
        self.video_container.setStyleSheet("background:#000; border-radius:10px;")
        v_box = QVBoxLayout(self.video_container)
        self.video_label = VideoLabel()
        self.video_label.clique_esquerdo.connect(self.gerir_clique_esquerdo)
        self.video_label.clique_direito.connect(self.gerir_clique_direito)
        v_box.addWidget(self.video_label)
        video_v.addWidget(self.video_container, stretch=10)

        # Barra de Controlos Inferior
        self.bottom_bar = QFrame()
        self.bottom_bar.setObjectName("ControlBar")
        self.bottom_bar.setFixedHeight(140)
        b_lay = QHBoxLayout(self.bottom_bar)

        slid_v = QVBoxLayout()
        slid_v.addWidget(QLabel("THRESHOLD"))
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(self.valor_threshold)
        self.slider_thresh.valueChanged.connect(self.set_thresh)
        slid_v.addWidget(self.slider_thresh)
        slid_v.addWidget(QLabel("SENSIBILIDADE"))
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(0, 10000)
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
            b = QPushButton(t)
            b.setObjectName(style)
            b.clicked.connect(f)
            draw_h.addWidget(b)
        b_lay.addLayout(draw_h, stretch=3)
        video_v.addWidget(self.bottom_bar)

        # 2. PÁGINA DE ESTATÍSTICAS
        self.page_stats = QWidget()
        stats_v = QVBoxLayout(self.page_stats)
        stats_v.setContentsMargins(30, 30, 30, 30)
        self.lbl_stats_title = QLabel("ESTATÍSTICAS DO MODELO")
        self.lbl_stats_title.setStyleSheet("font-size:18px; color:#FF8C00; font-weight:bold;")
        stats_v.addWidget(self.lbl_stats_title)

        cards_h = QHBoxLayout()
        self.card_ok = self._create_card("PEÇAS OK", "#2ECC71")
        cards_h.addWidget(self.card_ok)
        self.card_nok = self._create_card("PEÇAS NOK", "#E74C3C")
        cards_h.addWidget(self.card_nok)
        stats_v.addLayout(cards_h)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor('#0F1115')
        self.canvas = FigureCanvas(self.fig)
        stats_v.addWidget(self.canvas)

        # 3. PÁGINA DE COMANDOS
        self.page_cmds = QWidget()
        cmd_grid = QGridLayout(self.page_cmds)
        comandos_txt = ["[ S ] - SALVAR REFERÊNCIA / POLÍGONO", "[ I ] - INICIAR / PARAR INSPEÇÃO",
                        "[ Z ] - RETROCEDER PONTO", "[ D ] - LIMPAR DESENHOS",
                        "[ P ] - CAPTURAR PARA IA", "[ R ] - RESETAR CONTADORES",
                        "[ N ] - RENOMEAR MODELO"]
        for i, txt in enumerate(comandos_txt):
            btn = QPushButton(txt)
            btn.setObjectName("ControlBtn")
            btn.setMinimumHeight(60)
            cmd_grid.addWidget(btn, i // 2, i % 2)

        # 4. PÁGINA DE FALHAS
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

        self.pages.addWidget(self.page_video)
        self.pages.addWidget(self.page_stats)
        self.pages.addWidget(self.page_cmds)
        self.pages.addWidget(self.page_falhas)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.pages)

    def mostrar_app_principal(self):
        self.main_stack.setCurrentIndex(1)
        self.login_screen.pass_input.clear()

    def _create_card(self, title, color):
        card = StatCard()
        card.setObjectName("StatCard")
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

    # ==========================================================================
    # LÓGICA DE BASE DE DADOS E GESTÃO DE MODELOS
    # ==========================================================================
    def inicializar_bd(self):
        """Garante que todas as tabelas necessárias existem no sistema."""
        try:
            print(f"Tentando ligar à base de dados em: {DB_NAME}")
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # 1. Criar tabela de Peças se não existir
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
            
            # 2. Criar tabela de Utilizadores se não existir
            cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios
                              (
                                  username TEXT PRIMARY KEY,
                                  password TEXT
                              )''')
            
            # 3. Verificar se o admin existe
            cursor.execute("SELECT COUNT(*) FROM usuarios")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", ("admin", "1234"))
                print("-> Utilizador 'admin' (pass: 1234) criado.")
            
            conn.commit()
            conn.close()
            print("-> Base de dados verificada com sucesso.")
            
        except sqlite3.Error as e:
            print(f"!!! ERRO NA BASE DE DADOS: {e}")
            QMessageBox.critical(self, "Erro de Sistema", f"Falha ao carregar base de dados:\n{e}")

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
            self.roi_inspector.carregar_poligonos_peca(self.peca_ativa_id)
            self.lista_poligonos = self.roi_inspector.poligonos
            self.historico_status.clear()
            self.total_ok = p[8] if p[8] is not None else 0
            self.total_fail = p[9] if p[9] is not None else 0
            self.ultimo_motivo_falha = "Sem falhas registadas."

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
        self.lbl_falhas_title.setText(f"FALHAS: {self.peca_ativa_nome}")
        self.card_ok.lbl.setText(str(self.total_ok))
        self.card_nok.lbl.setText(str(self.total_fail))
        self.update_pie(self.total_ok, self.total_fail)
        self.atualizar_painel_falhas()

    def atualizar_painel_falhas(self):
        self.lbl_ultimo_motivo.setText(self.ultimo_motivo_falha)
        if self.peca_ativa_id is None:
            self.lbl_resumo_falhas.setText("Sem modelo selecionado.")
            return

        falhas_modelo = self.falhas_por_modelo.get(self.peca_ativa_id, {})
        if not falhas_modelo:
            self.lbl_resumo_falhas.setText("Sem dados ainda.")
            return

        linhas = [f"- {motivo}: {contagem}" for motivo, contagem in sorted(falhas_modelo.items(), key=lambda x: x[1], reverse=True)]
        self.lbl_resumo_falhas.setText("\n".join(linhas))

    def atualizar_lista_modelos(self):
        # Limpar layout e remover widgets do grupo de botões
        for i in reversed(range(self.model_layout.count())):
            item = self.model_layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                self.model_group.removeButton(w)
                w.setParent(None)
                w.deleteLater()

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
                        autopct='%1.1f%%', pctdistance=0.78, labeldistance=1.05, startangle=90,
                        textprops={'color': 'white', 'fontsize': 10},
                        wedgeprops={'width': 0.4, 'edgecolor': '#0F1115'})
            self.ax.text(0, 0, f"TOTAL\n{ok + nok}", color='white', ha='center', va='center', fontweight='bold')
        self.ax.set_aspect('equal')
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
        res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = res[0] if len(res) == 2 else res[1]
        if contornos: self.modelo_aprendido = max(contornos, key=cv2.contourArea)

    def executar_inspecao(self, frame):
        if not self.roi_inspector.poligonos:
            return "AGUARDAR", "Sem ROIs definidas para o modelo ativo", {}

        resultado_roi = self.roi_inspector.processar_frame_completo(frame)
        status_final = resultado_roi.get("status_global", "AGUARDAR")
        motivo_fail = resultado_roi.get("motivo", "")

        frame_com_rois = self.roi_inspector.desenhar_rois_em_frame(frame)
        frame[:] = frame_com_rois

        # Realce visual por ROI
        for idx, dados in resultado_roi.get("rois_dados", {}).items():
            cor = (0, 200, 0) if dados.get("status") == "OK" else ((0, 200, 255) if dados.get("status") == "AGUARDAR" else (0, 0, 255))
            pts = None
            if self.roi_inspector.poligonos:
                try:
                    roi_num = int(idx.split("_")[1])
                    if roi_num < len(self.roi_inspector.poligonos):
                        pts = np.array(self.roi_inspector.poligonos[roi_num], dtype=np.int32)
                except (ValueError, IndexError):
                    pts = None
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
            status_final = "FAIL"
            motivo_fail = "Forma Errada"

        cor_rect = (0, 255, 0) if status_final == "OK" else (0, 0, 255)
        cv2.drawContours(frame, [c_prin], -1, cor_rect, 2)
        cv2.putText(
            frame, f"{cor_prin} {motivo_fail}".strip(), (c_prin[0][0][0], c_prin[0][0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_rect, 2
        )
        return status_final, motivo_fail, resultado_roi

    def estabilizar_status(self, status):
        if status == "AGUARDAR":
            self.historico_status.clear()
            return status
        self.historico_status.append(status)
        if len(self.historico_status) < 3:
            return status
        return Counter(self.historico_status).most_common(1)[0][0]

    # ==========================================================================
    # LOOP PRINCIPAL E RENDERIZAÇÃO
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
                status_bruto, motivo_fail, resultado_roi = self.executar_inspecao(display_frame)
                status = self.estabilizar_status(status_bruto)
                cor_status = (0, 255, 255) if status == "AGUARDAR" else ((0, 255, 0) if status == "OK" else (0, 0, 255))
                cv2.putText(display_frame, f"STATUS: {status}", (w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_status, 3)
                if motivo_fail:
                    cv2.putText(display_frame, motivo_fail, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)
                if resultado_roi and resultado_roi.get("rois_dados"):
                    y = 70
                    for nome_roi, dados in resultado_roi["rois_dados"].items():
                        txt_roi = f"{nome_roi}: {dados.get('status', 'N/A')} | Fill={dados.get('fill_ratio', 0):.3f}"
                        cv2.putText(display_frame, txt_roi, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                        y += 18
                if time.time() - self.ultima_gravacao > 1.2:
                    if status == "AGUARDAR":
                        pass
                    elif status == "OK":
                        self.total_ok += 1
                    else:
                        self.total_fail += 1
                        self.ultimo_motivo_falha = motivo_fail or "Falha sem motivo identificado"
                        if self.peca_ativa_id is not None:
                            falhas_modelo = self.falhas_por_modelo.setdefault(self.peca_ativa_id, {})
                            falhas_modelo[self.ultimo_motivo_falha] = falhas_modelo.get(self.ultimo_motivo_falha, 0) + 1
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
                display_frame = np.zeros_like(frame)
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
        qimg = QImage(rgb.data, w_img, h_img, w_img * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
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

    def gerir_clique_direito(self, _x, _y):
        self.ponto_atual = []

    def mudar_aba(self, chave, idx):
        self.aba_ativa = chave
        self.pages.setCurrentIndex(idx)
        self.frame_fixo_treino = None
        self.ponto_atual = []
        # Atualizar estilo botões
        for k, btn in self.nav_buttons.items():
            btn.setProperty("active", k == chave)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def set_thresh(self, v):
        self.valor_threshold = v

    def set_sens(self, v):
        self.valor_sensibilidade = v

    def toggle_inspecao(self):
        self.inspecao_ativa = not self.inspecao_ativa
        self.historico_status.clear()

    def comando_salvar_s(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro.")
            return
        
        if self.aba_ativa == "TREINAR":
            # Se não está congelado, congela o frame atual
            if self.frame_fixo_treino is None:
                ret, f = self.cap.read()
                if ret: self.frame_fixo_treino = cv2.flip(f, 1)
                return

            # Se está congelado:
            # Caso 1: Tem pontos suficientes para um polígono -> Salva o polígono e MANTÉM congelado
            if len(self.ponto_atual) > 2:
                self.lista_poligonos.append(list(self.ponto_atual))
                self.ponto_atual = []
                self.salvar_progresso_bd()
                self.roi_inspector.carregar_poligonos_peca(self.peca_ativa_id)
                self.lista_poligonos = self.roi_inspector.poligonos
                # Não limpamos self.frame_fixo_treino aqui para permitir desenhar mais ROIs
            
            # Caso 2: Poucos pontos -> Pergunta se quer salvar como referência ou apenas descongelar
            else:
                resp = QMessageBox.question(self, "Referência", 
                                          "Deseja salvar esta imagem como REFERÊNCIA ou apenas DESCONGELAR?\n\n(Sim = Salvar Ref | Não = Descongelar | Cancelar = Nada)",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                
                if resp == QMessageBox.StandardButton.Yes:
                    conn = sqlite3.connect(DB_NAME)
                    _, buf = cv2.imencode(".jpg", self.frame_fixo_treino)
                    conn.execute("UPDATE pecas SET foto_ref=? WHERE id=?", (buf.tobytes(), self.peca_ativa_id))
                    conn.commit()
                    conn.close()
                    self.foto_referencia_processamento = self.frame_fixo_treino.copy()
                    self.processar_referencia(self.frame_fixo_treino)
                    self.frame_fixo_treino = None
                elif resp == QMessageBox.StandardButton.No:
                    self.frame_fixo_treino = None
                    self.ponto_atual = []

    def salvar_progresso_bd(self):
        if self.peca_ativa_id is None:
            return
        conn = sqlite3.connect(DB_NAME)
        conn.execute("UPDATE pecas SET poligonos=?, aprovadas=?, rejeitadas=? WHERE id=?",
                     (json.dumps(self.lista_poligonos), self.total_ok, self.total_fail, self.peca_ativa_id))
        conn.commit()
        conn.close()

    def reset_poligonos_db(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro.")
            return
        if QMessageBox.question(self, "Limpar", "Apagar desenhos do modelo?") == QMessageBox.Yes:
            self.lista_poligonos = []
            self.ponto_atual = []
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET poligonos=? WHERE id=?", (json.dumps([]), self.peca_ativa_id))
            conn.commit()
            conn.close()
            self.roi_inspector.carregar_poligonos_peca(self.peca_ativa_id)
            self.lista_poligonos = self.roi_inspector.poligonos

    def retroceder_ponto(self):
        if self.ponto_atual:
            self.ponto_atual.pop()
        elif self.lista_poligonos:
            self.lista_poligonos.pop()

    def capturar_para_ia(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro.")
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            ts = int(time.time() * 1000)
            path = os.path.join(DIR_DATASET, f"peca_{self.peca_ativa_id}_{ts}.jpg")
            cv2.imwrite(path, frame)

    def reset_contadores(self):
        self.total_ok = 0
        self.total_fail = 0
        self.salvar_progresso_bd()
        self.atualizar_ui_labels()

    def renomear_peca(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Crie ou selecione um modelo primeiro.")
            return
        novo, ok = QInputDialog.getText(self, "Renomear", "Novo nome:", text=self.peca_ativa_nome)
        if ok and novo:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET nome=? WHERE id=?", (novo, self.peca_ativa_id))
            conn.commit()
            conn.close()
            self.peca_ativa_nome = novo
            self.atualizar_lista_modelos()
            self.atualizar_ui_labels()

    def adicionar_novo_modelo(self):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM pecas")
        res = cursor.fetchone()[0]
        nxt = (res + 1) if res else 1
        cursor.execute("INSERT INTO pecas (id, nome, poligonos) VALUES (?, ?, ?)", (nxt, f"Modelo {nxt:02d}", "[]"))
        conn.commit()
        conn.close()
        self.atualizar_lista_modelos()

    def eliminar_modelo_atual(self):
        if self.peca_ativa_id is None:
            QMessageBox.warning(self, "Sem modelo", "Nenhum modelo selecionado para eliminar.")
            return
        if QMessageBox.question(self, "Eliminar", f"Eliminar {self.peca_ativa_nome}?") == QMessageBox.Yes:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pecas WHERE id=?", (self.peca_ativa_id,))
            conn.commit()
            conn.close()
            self.carregar_dados_iniciais()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionProApp()
    window.showFullScreen()
    sys.exit(app.exec())

# analisar uma peça 10 vezes e apena é rejeitada de der nok mais do qeu 6 vezes
# trocar o menu das falhas para aparecer uma foto da peça que esta NOK, o motivo e o estado atual OK ou NOK, mostrar a hora da inspeção
# melhorar a IA para identificar melhor as nao conformidades das peças
# melhorar o aspeto da app
# identificar cavidades como os encaixes do 20C, 30C ou 40C e verificar se os pinos estão completamente retos e nao estao tortos
# usar o chat gpt-5.5 como modelo para treino de IA


import sys
import os
import sqlite3
import random
import numpy as np
from datetime import datetime, timedelta

os.environ["QT_API"] = "pyside6"
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                               QPushButton, QHeaderView, QFrame, QLineEdit,
                               QScrollArea, QMessageBox, QComboBox)
from PySide6.QtCore import Qt, QTimer, QByteArray, QBuffer, QIODevice
from PySide6.QtGui import QPixmap, QImage

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from falhas_manager import FalhasManager

# Caminho da BD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "COLOR_DETECTION/sistema_industrial.db")

# ==============================================================================
# CONFIGURAÇÕES DE ESTILO (COMPATÍVEL COM VISION PRO)
# ==============================================================================
STYLE_SHEET = """
    QMainWindow { background-color: #0F1115; }
    QWidget { background-color: #0F1115; color: #ABB2BF; font-family: 'Segoe UI'; }
    QFrame#Card { background-color: #1E2227; border-radius: 10px; border: 1px solid #333; }
    QFrame#Header { background-color: #16191E; border-radius: 12px; border: 1px solid #2C313C; padding: 15px; }
    
    QLineEdit {
        background-color: #0F1115;
        border: 1px solid #3E4451;
        color: white;
        padding: 8px;
        border-radius: 4px;
        font-size: 13px;
    }
    QLineEdit:focus { border: 1px solid #FF8C00; }
    
    QTableWidget {
        background-color: transparent;
        border: none;
        gridline-color: transparent;
        selection-background-color: #1A1A1A;
        font-size: 13px;
    }
    
    QTableWidget::item {
        border-bottom: 1px solid #2C313C;
        padding: 10px;
        color: #ABB2BF;
    }
    
    QHeaderView::section {
        background-color: transparent;
        color: #888;
        padding: 12px;
        border: none;
        font-weight: bold;
        font-size: 11px;
    }
    
    QPushButton#ControlBtn {
        background-color: #21252B;
        border: 1px solid #3E4451;
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 11px;
    }
    QPushButton#ControlBtn:hover { border-color: #FF8C00; background-color: #2C313C; }
    
    QPushButton#PrimaryBtn {
        background-color: #FF8C00;
        border: none;
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: bold;
    }
    QPushButton#PrimaryBtn:hover { background-color: #E67E00; }
    
    QPushButton#DangerBtn {
        background-color: #2A1B1B;
        border: 1px solid #4A2626;
        color: #FF6666;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: bold;
    }
    QPushButton#DangerBtn:hover { background-color: #3D2222; border-color: #FF5555; }
    
    QScrollBar:vertical {
        border: none;
        background: #0F1115;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background: #333;
        min-height: 20px;
        border-radius: 5px;
    }
"""


# ==============================================================================
# COMPONENTES PERSONALIZADOS
# ==============================================================================

class StatusBadge(QLabel):
    """Badge de status para OK/FAIL com cores da estética VISION PRO"""
    def __init__(self, status, parent=None):
        super().__init__(status, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(70, 28)
        color = "#2ECC71" if status == "OK" else "#E74C3C"
        self.setStyleSheet(f"""
            background-color: {color}22;
            color: {color};
            border: 1px solid {color};
            border-radius: 14px;
            font-weight: bold;
            font-size: 11px;
        """)


class DetailPanel(QFrame):
    """Painel lateral de detalhes - Compatível com estilo VISION PRO"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setFixedWidth(350)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.title = QLabel("DETALHES DA FALHA")
        self.title.setStyleSheet("color: #FF8C00; font-weight: bold; font-size: 18px;")
        layout.addWidget(self.title)

        # Caixa com informações
        info_card = QFrame()
        info_card.setObjectName("Card")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(15, 15, 15, 15)
        info_layout.setSpacing(10)

        self.img_preview = QLabel()
        self.img_preview.setFixedSize(300, 200)
        self.img_preview.setStyleSheet("background-color: #000; border-radius: 8px; border: 1px solid #333;")
        self.img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_preview.setText("Sem Imagem")
        info_layout.addWidget(self.img_preview)

        self.info_area = QLabel("Selecione uma falha na tabela para ver detalhes.")
        self.info_area.setWordWrap(True)
        self.info_area.setStyleSheet("font-size: 12px; line-height: 160%; color: #ABB2BF;")
        info_layout.addWidget(self.info_area)

        layout.addWidget(info_card)
        layout.addStretch()

        # Botões de ação
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_reinspect = QPushButton("🔍 REABRIR INSPEÇÃO")
        self.btn_reinspect.setObjectName("ControlBtn")
        btn_layout.addWidget(self.btn_reinspect)

        self.btn_analise = QPushButton("📊 ANÁLISE DETALHADA")
        self.btn_analise.setObjectName("ControlBtn")
        btn_layout.addWidget(self.btn_analise)

        layout.addLayout(btn_layout)

    def update_info(self, data):
        dt, status, motive, img_blob = data
        cor_status = "🟢 OK" if status == "OK" else "🔴 FALHA"
        self.info_area.setText(f"""
<b>📅 Data/Hora:</b>
{dt}

<b>📊 Status:</b>
{cor_status}

<b>⚠️ Motivo da Falha:</b>
{motive if motive != "---" else "Nenhum (Peça Aprovada)"}
        """)

        # Exibe a imagem do blob
        if img_blob:
            try:
                nparr = np.frombuffer(img_blob, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = rgb_img.shape[:2]
                    bytes_per_line = 3 * w
                    qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    pixmap = pixmap.scaledToWidth(300, Qt.TransformationMode.SmoothTransformation)
                    self.img_preview.setPixmap(pixmap)
                    return
            except Exception as e:
                print(f"Erro ao carregar imagem: {e}")

        self.img_preview.setText("📷 Sem Imagem")
        self.img_preview.setStyleSheet("background-color: #1A1A1A; border-radius: 8px; border: 1px solid #333; color: #666;")


# ==============================================================================
# WIDGET PRINCIPAL
# ==============================================================================

class HistoricoInterativo(QWidget):
    """Interface de histórico de falhas com estética VISION PRO"""
    
    def __init__(self, operador_nome="demo"):
        super().__init__()
        self.operador_nome = operador_nome
        self.falhas_manager = FalhasManager(DB_NAME)
        self.peca_ativa_id = None
        self.data_source = []
        self.setup_ui()
        self.carregar_pecas()
        self.caregar_dados()

    def setup_ui(self):
        self.setStyleSheet(STYLE_SHEET)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ── HEADER ───────────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("Header")
        header_lay = QVBoxLayout(header)
        
        title = QLabel("📋 HISTÓRICO DE FALHAS")
        title.setStyleSheet("color: #FF8C00; font-size: 24px; font-weight: bold; border: none;")
        header_lay.addWidget(title)
        
        subtitle = QLabel(f"Operador: {self.operador_nome} | Acompanhamento detalhado de todas as falhas detetadas")
        subtitle.setStyleSheet("color: #888; font-size: 12px; border: none;")
        header_lay.addWidget(subtitle)
        
        main_layout.addWidget(header)

        # ── CONTEÚDO PRINCIPAL ───────────────────────────────────────────────
        content_h = QHBoxLayout()
        content_h.setSpacing(20)

        # Coluna esquerda: Filtros e Tabela
        left_col = QVBoxLayout()
        left_col.setSpacing(15)

        # Seleção de peça
        peca_layout = QHBoxLayout()
        peca_layout.setSpacing(10)
        peca_lbl = QLabel("🔧 Peça:")
        peca_lbl.setStyleSheet("color: #ABB2BF;")
        peca_layout.addWidget(peca_lbl)

        self.combo_pecas = QComboBox()
        self.combo_pecas.setStyleSheet("""
            QComboBox {
                background-color: #0F1115;
                border: 1px solid #3E4451;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #1E2227; color: white; }
        """)
        self.combo_pecas.currentIndexChanged.connect(self.on_peca_changed)
        peca_layout.addWidget(self.combo_pecas, stretch=1)

        left_col.addLayout(peca_layout)

        # Toolbar de filtros
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        
        search_lbl = QLabel("🔍 Filtrar:")
        search_lbl.setStyleSheet("color: #ABB2BF;")
        toolbar.addWidget(search_lbl)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Motivo, data ou resultado...")
        self.search_input.textChanged.connect(self.filter_table)
        toolbar.addWidget(self.search_input)

        self.btn_export = QPushButton("💾 EXPORTAR CSV")
        self.btn_export.setObjectName("ControlBtn")
        toolbar.addWidget(self.btn_export)

        self.btn_refresh = QPushButton("🔄 RECARREGAR")
        self.btn_refresh.setObjectName("ControlBtn")
        self.btn_refresh.clicked.connect(self.caregar_dados)
        toolbar.addWidget(self.btn_refresh)

        left_col.addLayout(toolbar)

        # Tabela dentro de Card
        table_card = QFrame()
        table_card.setObjectName("Card")
        table_lay = QVBoxLayout(table_card)
        table_lay.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["📅 DATA/HORA", "📊 STATUS", "⚠️ MOTIVO", "👤 OPERADOR"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemClicked.connect(self.on_row_selected)

        table_lay.addWidget(self.table)
        left_col.addWidget(table_card)

        content_h.addLayout(left_col, stretch=2)

        # Coluna direita: Detalhes
        self.details = DetailPanel()
        content_h.addWidget(self.details)

        main_layout.addLayout(content_h)

    def carregar_pecas(self):
        """Carrega lista de peças da BD"""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT id, nome FROM pecas ORDER BY id")
            pecas = cursor.fetchall()
            conn.close()

            self.combo_pecas.blockSignals(True)
            self.combo_pecas.addItem("Todas as Peças", None)
            for pid, pnome in pecas:
                self.combo_pecas.addItem(f"#{pid} - {pnome}", pid)
            self.combo_pecas.blockSignals(False)
        except Exception as e:
            print(f"Erro ao carregar peças: {e}")

    def on_peca_changed(self):
        """Quando muda a seleção de peça"""
        self.peca_ativa_id = self.combo_pecas.currentData()
        self.caregar_dados()

    def caregar_dados(self):
        """Carrega dados do histórico de falhas"""
        self.data_source = []

        if self.peca_ativa_id is not None:
            # Busca histórico da peça específica
            historico = self.falhas_manager.obter_historico_peca(self.peca_ativa_id, limite=100)
            for falha_id, peca_id, operador, data_hora, status, motivo, timestamp in historico:
                img_blob = self.falhas_manager.obter_imagem_falha(falha_id)
                self.data_source.append((data_hora, status, motivo, img_blob, operador, falha_id))
        else:
            # Busca histórico de todas as peças
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''SELECT id, data_hora, status, motivo, imagem_blob, operador, peca_id
                    FROM historico_falhas 
                    ORDER BY timestamp DESC
                    LIMIT 100''')
                resultados = cursor.fetchall()
                conn.close()

                for falha_id, data_hora, status, motivo, img_blob, operador, peca_id in resultados:
                    self.data_source.append((data_hora, status, motivo, img_blob, operador, falha_id))
            except Exception as e:
                print(f"Erro ao carregar histórico: {e}")

        self.update_table(self.data_source)

    def update_table(self, data_list):
        """Atualiza a tabela com os dados fornecidos"""
        self.table.setRowCount(len(data_list))
        for row, (dt, status, mot, img_blob, operador, falha_id) in enumerate(data_list):
            # Coluna 1: Data/Hora
            self.table.setItem(row, 0, QTableWidgetItem(dt))

            # Coluna 2: Status Badge
            badge = StatusBadge(status)
            container = QWidget()
            l = QHBoxLayout(container)
            l.addWidget(badge)
            l.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 1, container)

            # Coluna 3: Motivo
            self.table.setItem(row, 2, QTableWidgetItem(mot if mot != "---" else "Nenhum"))

            # Coluna 4: Operador
            self.table.setItem(row, 3, QTableWidgetItem(operador))

            self.table.setRowHeight(row, 45)

    def filter_table(self, text):
        """Filtra a tabela conforme o texto inserido"""
        if not text:
            self.update_table(self.data_source)
            return
        filtered = [d for d in self.data_source if text.lower() in str(d).lower()]
        self.update_table(filtered)

    def on_row_selected(self, item):
        """Atualiza o painel de detalhes ao selecionar uma linha"""
        row = item.row()
        if row < len(self.data_source):
            dt, status, motive, img_blob, operador, falha_id = self.data_source[row]
            self.details.update_info((dt, status, motive, img_blob))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("VISION PRO | Histórico de Falhas")
    window.resize(1400, 800)

    # Se foi passado um argumento, usa como nome do operador
    operador = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    central = HistoricoInterativo(operador)
    window.setCentralWidget(central)

    window.show()
    sys.exit(app.exec())

# TODO: Implementar exportação CSV com filtros aplicados
# TODO: Implementar análise detalhada ao clicar em "ANÁLISE DETALHADA"
# TODO: Sincronizar automaticamente quando nova falha é registada

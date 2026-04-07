# spellchecker: disable
import cv2
import numpy as np
import sqlite3
import json
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog
import time
import os

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
DB_NAME = "sistema_industrial.db"
DIR_DATASET = "../dataset_ia"
COR_DESTAQUE_HEX = "#00FFFF"
COR_OPENCV_BGR = (255, 255, 0)
COR_FUNDO_FRAME = (15, 15, 15)

# Cores para Deteção (HSV) - Trazido do sistema_completo.py
CORES_DETECCAO = {
    "AZUL": (np.array([85, 100, 50]), np.array([140, 255, 255]), (255, 0, 0)),
    "PRETO": (np.array([0, 0, 0]), np.array([180, 255, 50]), (60, 60, 60))
}


# ==============================================================================
# CLASSE PRINCIPAL
# ==============================================================================
class VisionProSystem(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configuração Inicial ---
        if not os.path.exists(DIR_DATASET):
            os.makedirs(DIR_DATASET)

        self.inicializar_bd()

        # --- Estado do Sistema ---
        self.peca_ativa_id = 1
        self.peca_ativa_nome = "Modelo 01"
        self.inspecao_ativa = False
        self.status_inspecao = "OFF"
        self.aba_ativa = "INSPECAO"

        # --- Variáveis de Processamento ---
        self.lista_poligonos = []
        self.ponto_atual = []
        self.frame_fixo_treino = None
        self.foto_referencia_display = None  # Imagem para exibição
        self.foto_referencia_processamento = None  # Imagem bruta para CV
        self.modelo_aprendido = None  # Contorno da referência
        self.bordas_ref = 0
        self.medida_ideal = 0.0
        self.tolerancia = 0.5
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Estatísticas
        self.total_ok = 0
        self.total_fail = 0
        self.historico_inspecao = []
        self.ultima_gravacao = 0

        # Calibração (Padrões)
        self.valor_threshold = 100
        self.valor_sensibilidade = 800
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # --- Interface Gráfica (Layout) ---
        self.title("VISION PRO - VERSION 1.0")
        self.geometry("1400x900")
        self.configure(fg_color="#000000")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._setup_sidebar()
        self._setup_main_area()
        self._setup_dashboard()

        # --- Hardware ---
        self.cap = cv2.VideoCapture(0)

        # --- Carregar Dados Iniciais ---
        self.carregar_dados_peca(1)
        self.criar_lista_pecas()

        # --- Atalhos de Teclado ---
        self.bind("<KeyPress-i>", lambda _: self.toggle_inspecao())
        self.bind("<KeyPress-s>", lambda _: self.comando_salvar_s())
        self.bind("<KeyPress-d>", lambda _: self.reset_poligonos_db())
        self.bind("<KeyPress-p>", lambda _: self.capturar_para_ia())
        self.bind("<KeyPress-r>", lambda _: self.reset_contadores())
        self.bind("<KeyPress-n>", lambda _: self.renomear_peca())
        self.bind("<KeyPress-z>", lambda _: self.retroceder_ponto())

        # Iniciar Loop
        self.atualizar_loop()

    def _setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#0A0A0A")
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="VISION PRO", font=("Orbitron", 24, "bold"),
                     text_color=COR_DESTAQUE_HEX).pack(pady=20)

        # Botões de Abas Laterais
        abas = ["INSPECAO", "TREINAR", "REFERENCIAS", "IA / DATASET", "COMANDOS"]
        for aba in abas:
            ctk.CTkButton(self.sidebar, text=aba, font=("Roboto", 12, "bold"),
                          command=lambda a=aba: self.mudar_aba(a),
                          fg_color="#4F909F", hover_color="#333333").pack(pady=4, padx=20, fill="x")

        # Botão LIMPAR
        ctk.CTkButton(self.sidebar, text="LIMPAR DESENHOS", font=("Roboto", 12, "bold"),
                      command=self.reset_poligonos_db,
                      fg_color="#441111", hover_color="#662222").pack(pady=(20, 4), padx=20, fill="x")
        # Botâo RETROCEDER
        ctk.CTkButton(self.sidebar, text="RETROCEDER (Z)", font=("Roboto", 12, "bold"),
                      command=self.retroceder_ponto,
                      fg_color="#444411", hover_color="#666622").pack(pady=4, padx=20, fill="x")

        # Adicionar modelo de peça nova
        ctk.CTkButton(self.sidebar,
                      text="NOVO MODELO",
                      font=("Roboto", 12, "bold"),
                      command=self.adicionar_novo_modelo,
                      fg_color="#114411",  # Cor verde para indicar "adicionar"
                      hover_color="#226622").pack(pady=4, padx=20, fill="x")

        # Botão para eliminar peça atual
        ctk.CTkButton(self.sidebar,
                      text="ELIMINAR MODELO",
                      font=("Roboto", 12, "bold"),
                      command=self.eliminar_modelo_atual,
                      fg_color="#661111",  # Vermelho escuro
                      hover_color="#992222").pack(pady=4, padx=20, fill="x")

        # Controles de Ajuste
        ctk.CTkLabel(self.sidebar, text="LIMIAR (Threshold)", font=("Roboto", 11)).pack(pady=(20, 0))
        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0, to=255, command=self.set_thresh)
        self.slider_thresh.set(self.valor_threshold)
        self.slider_thresh.pack(pady=5, padx=20)

        ctk.CTkLabel(self.sidebar, text="SENSIBILIDADE (Inspecao)", font=("Roboto", 11)).pack(pady=(10, 0))
        self.slider_sens = ctk.CTkSlider(self.sidebar, from_=0, to=10000, command=self.set_sens)
        self.slider_sens.set(self.valor_sensibilidade)
        self.slider_sens.pack(pady=5, padx=20)

        # Lista de Peças
        ctk.CTkLabel(self.sidebar, text="MODELOS", font=("Roboto", 11, "bold")).pack(pady=(20, 5))
        self.scroll_ids = ctk.CTkScrollableFrame(self.sidebar, fg_color="#050505", height=300)
        self.scroll_ids.pack(fill="both", expand=True, padx=10, pady=10)

    def _setup_main_area(self):
        self.main_area = ctk.CTkFrame(self, corner_radius=15, fg_color="#000000")
        self.main_area.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        self.lbl_nome_peca = ctk.CTkLabel(self.main_area, text="...", font=("Roboto", 20, "bold"),
                                          text_color=COR_DESTAQUE_HEX)
        self.lbl_nome_peca.pack(pady=10)

        # Label para exibir o vídeo
        self.video_label = ctk.CTkLabel(self.main_area, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=5)

        # Eventos do rato no Vídeo
        self.video_label.bind("<Button-1>", self.gerir_clique_esquerdo)
        self.video_label.bind("<Button-3>", self.gerir_clique_direito)

    def _setup_dashboard(self):
        self.dash = ctk.CTkFrame(self, height=100, corner_radius=15, fg_color="#0A0A0A")
        self.dash.grid(row=1, column=1, padx=15, pady=(0, 15), sticky="ew")

        # Indicação de peças OK e NOK e do status da INSPEÇÂO
        self.lbl_ok = ctk.CTkLabel(self.dash, text="OK: 0", text_color="#00FF96", font=("Roboto", 24, "bold"))
        self.lbl_ok.pack(side="left", padx=40)

        self.lbl_fail = ctk.CTkLabel(self.dash, text="FAIL: 0", text_color="#FF4B4B", font=("Roboto", 24, "bold"))
        self.lbl_fail.pack(side="left", padx=40)

        self.lbl_status = ctk.CTkLabel(self.dash, text="STATUS: OFF", font=("Roboto", 24, "bold"))
        self.lbl_status.pack(side="right", padx=50)

    # ==========================================================================
    # BANCO DE DADOS (sistema_industrial.bd)
    # ==========================================================================
    def inicializar_bd(self):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Criação da tabela com todos os campos necessários
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

        # Migração: Verificar se colunas novas existem (para compatibilidade com versões antigas)
        cursor.execute("PRAGMA table_info(pecas)")
        colunas = [info[1] for info in cursor.fetchall()]

        novas_colunas = {
            "aprovadas": "INTEGER DEFAULT 0",
            "rejeitadas": "INTEGER DEFAULT 0",
            "bordas_ref": "INTEGER DEFAULT 0",
            "medida_ideal": "REAL",
            "tolerancia": "REAL"
        }

        for col, definition in novas_colunas.items():
            if col not in colunas:
                try:
                    cursor.execute(f"ALTER TABLE pecas ADD COLUMN {col} {definition}")
                except:
                    pass  # Coluna pode já existir ou erro de compatibilidade

        # Criar dados iniciais se vazio
        # 3 linhas abaixo ultimo numero aumenta numero de peças
        cursor.execute("SELECT COUNT(*) FROM pecas")
        if cursor.fetchone()[0] == 0:
            for i in range(1, 11):
                cursor.execute("INSERT INTO pecas (id, nome, medida_ideal, tolerancia) VALUES (?, ?, ?, ?)",
                               (i, f"Modelo {i:02d}", 5.0, 0.5))

        conn.commit()
        conn.close()

    def carregar_dados_peca(self, id_p):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pecas WHERE id=?", (id_p,))
        p = cursor.fetchone()
        conn.close()

        if p:
            # Assumindo a estrutura:
            # 0:id, 1:nome, 2:polys, 3:pts, 4:ideal, 5:tol, 6:escala, 7:foto, 8:ok, 9:fail, 10: bordas
            self.peca_ativa_id = p[0]
            self.peca_ativa_nome = p[1]
            self.lista_poligonos = json.loads(p[2]) if p[2] else []
            self.medida_ideal = p[4] if p[4] else 0.0
            self.tolerancia = p[5] if p[5] else 0.5
            self.total_ok = p[8] if len(p) > 8 else 0
            self.total_fail = p[9] if len(p) > 9 else 0
            self.bordas_ref = p[10] if len(p) > 10 else 0

            # Carregar Imagem de Referência
            if len(p) > 7 and p[7]:
                nparr = np.frombuffer(p[7], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.foto_referencia_processamento = img
                self.foto_referencia_display = img.copy()
                self.processar_referencia(img)
            else:
                self.foto_referencia_processamento = None
                self.foto_referencia_display = None
                self.modelo_aprendido = None

            self.atualizar_ui_labels()

    def atualizar_ui_labels(self):
        self.lbl_nome_peca.configure(text=f"ATIVO: {self.peca_ativa_nome}")
        self.lbl_ok.configure(text=f"OK: {self.total_ok}")
        self.lbl_fail.configure(text=f"FAIL: {self.total_fail}")

    def criar_lista_pecas(self):
        for w in self.scroll_ids.winfo_children(): w.destroy()
        conn = sqlite3.connect(DB_NAME)
        data = conn.execute("SELECT id, nome FROM pecas").fetchall()
        conn.close()

        for pid, nome in data:
            btn = ctk.CTkButton(self.scroll_ids, text=f"ID {pid:02} | {nome[:12]}",
                                fg_color="#4F909F", height=35, anchor="w",
                                command=lambda x=pid: self.carregar_dados_peca(x))
            btn.pack(fill="x", pady=2)

    # ==========================================================================
    # LÓGICA DE VISÃO COMPUTACIONAL
    # ==========================================================================
    def processar_referencia(self, img):
        """Gera o modelo matemático a partir da foto de referência"""
        if img is None: return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe_img = self.clahe.apply(blur)

        _, thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            self.modelo_aprendido = max(contornos, key=cv2.contourArea)

    def executar_inspecao(self, frame):
        """Lógica combinada de inspeção (Polígonos + Cores + Forma)"""
        h, w = frame.shape[:2]
        status_final = "OK"
        motivo_fail = ""

        # 1. Inspeção por Zonas (Polígonos) - Lógica de Geometria e Furos Adicionada
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filt = self.clahe.apply(gray)
        _, thresh = cv2.threshold(filt, self.valor_threshold, 255, cv2.THRESH_BINARY_INV)

        for poly in self.lista_poligonos:
            pts = np.array(poly)
            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, [pts], 255)

            # --- MELHORIA: ISOLAMENTO GEOMÉTRICO ---
            zona_isolada = cv2.bitwise_and(thresh, mask)
            cnts, _ = cv2.findContours(zona_isolada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            zona_ok = False
            if cnts:
                c_max = max(cnts, key=cv2.contourArea)
                area_real = cv2.contourArea(c_max)

                # Critério 1: Área mínima
                if area_real > self.valor_sensibilidade:
                    x_b, y_b, w_b, h_b = cv2.boundingRect(c_max)
                    ratio = float(w_b) / h_b

                    # Critério 2: Formato (Evita erros por braços ou janelas de UI)
                    if 0.4 < ratio < 2.5:
                        zona_ok = True
                        cv2.drawContours(frame, [c_max], -1, (0, 255, 0), 2)

                        # --- MELHORIA: VERIFICAÇÃO DE FUROS ---
                        zona_gray = cv2.bitwise_and(filt, mask)
                        circulos = cv2.HoughCircles(zona_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                                    param1=50, param2=30, minRadius=5, maxRadius=40)
                        if circulos is not None:
                            for i in np.uint16(np.around(circulos))[0, :]:
                                cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 255), 2)

            if not zona_ok:
                status_final = "FAIL"
                motivo_fail = "Zona Falha"
                cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

        # 2. Inspeção por Cor e Forma - Lógica original restaurada
        if status_final == "OK":
            roi_blur = cv2.bilateralFilter(frame, 9, 75, 75)
            hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

            objetos_candidatos = []
            for nome_cor, (low, high, cor_viz) in CORES_DETECCAO.items():
                mask_cor = cv2.inRange(hsv, low, high)
                contornos, _ = cv2.findContours(mask_cor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contornos:
                    area = cv2.contourArea(c)
                    if area > 3000:
                        objetos_candidatos.append((area, c, nome_cor))

            if objetos_candidatos:
                objetos_candidatos.sort(key=lambda x: x[0], reverse=True)
                area_prin, c_prin, cor_prin = objetos_candidatos[0]

                # Match Shape
                if self.modelo_aprendido is not None:
                    try:
                        match_val = cv2.matchShapes(self.modelo_aprendido, c_prin, 1, 0.0)
                        if match_val > 0.25:
                            status_final = "FAIL"
                            motivo_fail = f"Forma {match_val:.2f}"
                    except:
                        pass

                cor_rect = (0, 255, 0) if status_final == "OK" else (0, 0, 255)
                cv2.drawContours(frame, [c_prin], -1, cor_rect, 2)
                cv2.putText(frame, f"{cor_prin} {motivo_fail}", (c_prin[0][0][0], c_prin[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_rect, 2)

        return status_final, motivo_fail

    # ==========================================================================
    # LOOP PRINCIPAL E RENDERIZAÇÃO
    # ==========================================================================
    def atualizar_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display_frame = frame.copy()

            # --- Lógica por Aba ---
            if self.aba_ativa == "TREINAR":
                if self.frame_fixo_treino is not None:
                    display_frame = self.frame_fixo_treino.copy()
                    for poly in self.lista_poligonos:
                        cv2.polylines(display_frame, [np.array(poly)], True, (100, 100, 100), 1)
                    for pt in self.ponto_atual:
                        cv2.circle(display_frame, pt, 4, (0, 255, 255), -1)
                    if len(self.ponto_atual) > 1:
                        cv2.polylines(display_frame, [np.array(self.ponto_atual)], False, (0, 255, 255), 2)

                    cv2.putText(display_frame, "MODO: CONGELADO (Desenhe Agora)", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_frame, "[Botao Direito] Limpar | [S] Salvar", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                else:
                    cv2.putText(display_frame, "PRESSIONE [S] PARA CONGELAR E DESENHAR", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    for poly in self.lista_poligonos:
                        cv2.polylines(display_frame, [np.array(poly)], True, COR_OPENCV_BGR, 1)

            elif self.aba_ativa == "INSPECAO":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                filt = self.clahe.apply(gray)
                _, thresh = cv2.threshold(filt, self.valor_threshold, 255, cv2.THRESH_BINARY_INV)

                if self.inspecao_ativa:
                    status, motivo = self.executar_inspecao(display_frame)
                    self.status_inspecao = status
                    self.lbl_status.configure(text=f"STATUS: {status}",
                                              text_color="#00FF96" if status == "OK" else "#FF4B4B")

                    if time.time() - self.ultima_gravacao > 1.0:
                        if status == "OK":
                            self.total_ok += 1
                        else:
                            self.total_fail += 1
                        self.ultima_gravacao = time.time()
                        self.atualizar_ui_labels()
                else:
                    self.status_inspecao = "PAUSADO"
                    self.lbl_status.configure(text="STATUS: PAUSADO", text_color="white")
                    cv2.putText(display_frame, "INSPECAO PAUSADA (TECLA 'I')", (50, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    for poly in self.lista_poligonos:
                        cv2.polylines(display_frame, [np.array(poly)], True, (255, 255, 0), 2)

                # Desenhar mini-janela por ultimo para nao interferir na inspecao
                thresh_mini = cv2.resize(thresh, (w // 5, h // 5))
                mini_bgr = cv2.cvtColor(thresh_mini, cv2.COLOR_GRAY2BGR)
                display_frame[10:h // 5 + 10, 10:w // 5 + 10] = mini_bgr
                cv2.rectangle(display_frame, (10, 10), (w // 5 + 10, h // 5 + 10), (0, 255, 0), 1)

            elif self.aba_ativa == "REFERENCIAS":
                if self.foto_referencia_display is not None:
                    display_frame = self.foto_referencia_display.copy()
                    cv2.putText(display_frame, "IMAGEM DE REFERENCIA SALVA", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    display_frame = np.zeros_like(frame)
                    cv2.putText(display_frame, "SEM REFERENCIA DEFINIDA", (50, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

            elif self.aba_ativa == "IA / DATASET":
                imgs = [f for f in os.listdir(DIR_DATASET) if
                        f.startswith(f"peca_{self.peca_ativa_id}_") and f.endswith(".jpg")]
                cv2.putText(display_frame, f"Total Imagens: {len(imgs)}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if imgs:
                    last_img_path = os.path.join(DIR_DATASET, sorted(imgs)[-1])
                    try:
                        last_img = cv2.imread(last_img_path)
                        if last_img is not None:
                            last_img = cv2.resize(last_img, (320, 240))
                            display_frame[150:150 + 240, 30:30 + 320] = last_img
                    except:
                        pass


            elif self.aba_ativa == "COMANDOS":
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
                cmds = [
                    "ATALHOS DO SISTEMA:",
                    "[ S ]  : Salvar Imagem/Poligono (Treino)",
                    "[ I ]  : Iniciar/Parar Inspecao",
                    "[ P ]  : Capturar Foto para Dataset IA",
                    "[ R ]  : Reset Contadores",
                    "[ N ]  : Renomear Peca Atual",
                    "[ D ]  : Deletar Poligonos",
                    "[ Z ]  : Retroceder Ponto"  # Adicionámos o novo que criaste
                ]
                # --- LÓGICA DE CENTRALIZAÇÃO ---
                espacamento = 50  # Distância entre linhas
                altura_total_texto = len(cmds) * espacamento
                # Calcular o Y inicial para que o bloco fique no meio
                y_start = (h - altura_total_texto) // 2

                for i, txt in enumerate(cmds):
                    # Calcular a largura do texto para centrar horizontalmente também
                    tamanho_texto = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    x_center = (w - tamanho_texto[0]) // 2
                    # Desenhar cada linha
                    cv2.putText(display_frame, txt, (x_center, y_start + i * espacamento),

                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            self.exibir_frame(display_frame)

        self.after(20, self.atualizar_loop)

    def exibir_frame(self, frame):
        w_lbl = self.video_label.winfo_width()
        h_lbl = self.video_label.winfo_height()
        if w_lbl < 10 or h_lbl < 10: return

        h_img, w_img = frame.shape[:2]

        # Cálculo rigoroso da escala
        self.scale_factor = min(w_lbl / w_img, h_lbl / h_img)

        new_w = int(w_img * self.scale_factor)
        new_h = int(h_img * self.scale_factor)

        # Margens exatas (Offsets)
        self.offset_x = (w_lbl - new_w) / 2
        self.offset_y = (h_lbl - new_h) / 2

        resized = cv2.resize(frame, (new_w, new_h))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

    def gerir_clique_esquerdo(self, event):
        if self.aba_ativa == "TREINAR" and self.frame_fixo_treino is not None:
            # 1. Pegar o tamanho real do widget
            w_lbl = self.video_label.winfo_width()
            h_lbl = self.video_label.winfo_height()

            # Altura e largura da imagem original
            h_img, w_img = self.frame_fixo_treino.shape[:2]

            # 2. CÁLCULO DE ESCALA E OFFSETS (PREVISÃO DECIMAL)
            # Mudamos de // para / para manter a precisão dos meios-pixels
            scale = min(w_lbl // w_img, h_lbl // h_img)

            # O offset deve ser float para não "empurrar" a imagem 1 pixel para o lado errado
            off_x = (w_lbl - (w_img * scale)) // 3
            off_y = (h_lbl - (h_img * scale)) // 2.80

            # 3. AJUSTE DE COORDENADAS COM ARREDONDAMENTO (ROUND)
            # O int() apenas corta os decimais. O round() aproxima para o pixel mais próximo.
            # Isto elimina o desvio de "quase 1 pixel" que estás a sentir.
            real_x = int(round((event.x - off_x) // scale))
            real_y = int(round((event.y - off_y) // scale))

            # 4. SEGURANÇA DE LIMITES
            # Garante que o arredondamento não atira o ponto para fora da matriz (ex: pixel 640 numa imagem de 640)
            real_x = max(0, min(real_x, w_img - 1))
            real_y = max(0, min(real_y, h_img - 1))

            self.ponto_atual.append((real_x, real_y))

            # Debug opcional para veres no terminal se o clique está a bater certo
            # print(f"Rato: {event.x} | OffsetX: {off_x:.2f} | RealX: {real_x}"))

    # Para limpar o desenho que estás a fazer AGORA (Botão Direito)
    def gerir_clique_direito(self, event):
        self.ponto_atual = []  # Limpa os pontos amarelos atuais
        # Opcional: print("Desenho atual cancelado")

    def reset_poligonos_db(self, event=None):
        pergunta = f"Deseja apagar os desenhos do modelo: {self.peca_ativa_nome}?"
        if messagebox.askyesno("Limpar Zonas", pergunta):
            # Limpa apenas os dados da peça ATIVA na memória
            self.lista_poligonos = []
            self.ponto_atual = []

            # Atualiza a Base de Dados apenas para esta peça
            try:
                conn = sqlite3.connect(DB_NAME)
                # json.dumps([]) guarda uma lista vazia no formato correto para o teu sistema
                conn.execute("UPDATE pecas SET poligonos = ? WHERE id = ?",
                             (json.dumps([]), self.peca_ativa_id))
                conn.commit()
                conn.close()

                # Reset do frame de treino para "limpar" a vista do utilizador
                self.frame_fixo_treino = None
                messagebox.showinfo("Sucesso", f"Zonas de {self.peca_ativa_nome} apagadas.")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao aceder à DB: {e}")

    def mudar_aba(self, nova_aba):
        self.aba_ativa = nova_aba
        self.frame_fixo_treino = None
        self.ponto_atual = []

    def set_thresh(self, val):
        self.valor_threshold = int(float(val))

    def set_sens(self, val):
        self.valor_sensibilidade = int(float(val))

    def toggle_inspecao(self):
        self.inspecao_ativa = not self.inspecao_ativa
        if not self.inspecao_ativa:
            self.status_inspecao = "OFF"
            self.lbl_status.configure(text="STATUS: OFF", text_color="white")

    def comando_salvar_s(self):
        if self.aba_ativa == "TREINAR":
            if self.frame_fixo_treino is None:
                ret, f = self.cap.read()
                if ret: self.frame_fixo_treino = cv2.flip(f, 1)
            else:
                salvo = False
                if len(self.ponto_atual) > 2:
                    self.lista_poligonos.append(list(self.ponto_atual))
                    self.ponto_atual = []
                    salvo = True
                if not salvo and messagebox.askyesno("Atualizar Referência", "Salvar esta imagem como referência?"):
                    conn = sqlite3.connect(DB_NAME)
                    _, buf = cv2.imencode(".jpg", self.frame_fixo_treino)
                    conn.execute("UPDATE pecas SET foto_ref=? WHERE id=?", (buf.tobytes(), self.peca_ativa_id))
                    conn.commit();
                    conn.close()
                    self.foto_referencia_display = self.frame_fixo_treino.copy()
                    self.processar_referencia(self.frame_fixo_treino)
                    salvo = True
                if salvo:
                    self.salvar_progresso_bd()
                    self.frame_fixo_treino = None

    def salvar_progresso_bd(self):
        conn = sqlite3.connect(DB_NAME)
        conn.execute("""UPDATE pecas
                        SET poligonos=?,
                            aprovadas=?,
                            rejeitadas=?,
                            bordas_ref=?
                        WHERE id = ?""",
                     (json.dumps(self.lista_poligonos), self.total_ok, self.total_fail, self.bordas_ref,
                      self.peca_ativa_id))
        conn.commit();
        conn.close()

    def capturar_para_ia(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            ts = int(time.time() * 1000)
            nome_arq = f"peca_{self.peca_ativa_id}_{ts}"
            cv2.imwrite(os.path.join(DIR_DATASET, f"{nome_arq}.jpg"), frame)
            if self.lista_poligonos:
                pts = np.array(self.lista_poligonos[0])
                x, y, w, h = cv2.boundingRect(pts)
                img_h, img_w = frame.shape[:2]
                with open(os.path.join(DIR_DATASET, f"{nome_arq}.txt"), "w") as f:
                    f.write(
                        f"{self.peca_ativa_id - 1} {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")

    def reset_contadores(self):
        self.total_ok = 0;
        self.total_fail = 0
        self.salvar_progresso_bd();
        self.atualizar_ui_labels()

    def renomear_peca(self):
        novo = simpledialog.askstring("Renomear", f"Novo nome:")
        if novo:
            conn = sqlite3.connect(DB_NAME);
            conn.execute("UPDATE pecas SET nome=? WHERE id=?", (novo, self.peca_ativa_id))
            conn.commit();
            conn.close();
            self.peca_ativa_nome = novo;
            self.criar_lista_pecas();
            self.atualizar_ui_labels()

    def on_close(self):
        self.cap.release();
        self.destroy()

    def retroceder_ponto(self):
        if self.ponto_atual:
            self.ponto_atual.pop()  # Remove o último item da lista
        elif self.lista_poligonos:
            # Se não houver pontos soltos, remove o último polígono completo
            if messagebox.askyesno("Retroceder", "Apagar o último polígono completo?"):
                self.lista_poligonos.pop()

    def adicionar_novo_modelo(self): # adiciona uma peça a BD
        # 1. Conectar à Base de Dados
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # 2. Descobrir qual é o próximo ID disponível
        cursor.execute("SELECT MAX(id) FROM pecas")
        resultado = cursor.fetchone()[0]
        proximo_id = (resultado + 1) if resultado else 1

        # 3. Inserir o novo modelo na tabela
        # Usamos f-strings para o nome, ex: "Modelo 05"
        nome_padrao = f"Modelo {proximo_id:02d}"
        cursor.execute("INSERT INTO pecas (id, nome, poligonos) VALUES (?, ?, ?)",
                       (proximo_id, nome_padrao, json.dumps([])))

        # 4. Gravar e fechar
        conn.commit()
        conn.close()

        # 5. Atualizar a interface para o novo botão aparecer na lista
        self.criar_lista_pecas()
        print(f"Sucesso: {nome_padrao} adicionado à base de dados.")

    def eliminar_modelo_atual(self): #Elemina a peça da BD
        # 1. Confirmar com o utilizador para evitar acidentes
        pergunta = f"Tens a certeza que queres eliminar o modelo: {self.peca_ativa_nome}?"
        if messagebox.askyesno("Eliminar Modelo", pergunta):
            try:
                # 2. Conectar e remover da Base de Dados
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM pecas WHERE id = ?", (self.peca_ativa_id,))
                conn.commit()
                conn.close()

                # 3. Informar o utilizador
                messagebox.showinfo("Sucesso", f"Modelo {self.peca_ativa_id} removido.")

                # 4. Carregar o modelo ID 1 (ou outro) para o sistema não ficar vazio
                self.carregar_dados_peca(1)

                # 5. Atualizar a lista visual (reconstruir os botões laterais)
                self.criar_lista_pecas()

            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível eliminar: {e}")

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark");
    app = VisionProSystem();
    app.mainloop()



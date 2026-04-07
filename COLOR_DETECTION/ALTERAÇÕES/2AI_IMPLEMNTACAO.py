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

# --- CONFIGURAÇÕES ---
DB_NAME = "sistema_industrial.db"
COR_DESTAQUE_HEX = "#00FFFF"
COR_OPENCV_BGR = (255, 255, 0)
DIR_DATASET = "dataset_ia"


class VisionProSystem(ctk.CTk):
    def __init__(self):
        super().__init__()

        if not os.path.exists(DIR_DATASET):
            os.makedirs(DIR_DATASET)

        # Variáveis de Inspeção e Calibração
        self.peca_ativa_id = 1
        self.peca_ativa_nome = "Modelo 01"
        self.inspecao_ativa = False
        self.total_ok = 0
        self.total_fail = 0
        self.status_inspecao = "OFF"
        self.lista_poligonos = []
        self.ponto_atual = []
        self.aba_ativa = "INSPECAO"
        self.frame_fixo_treino = None
        self.foto_referencia_display = None

        # Parâmetros de Processamento (Ajustáveis via Slider)
        self.valor_threshold = 100
        self.valor_sensibilidade = 800
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.title("VISION PRO - INDUSTRIAL INSPECTOR v2.0")
        self.geometry("1400x900")
        self.configure(fg_color="#000000")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- BARRA LATERAL ---
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#0A0A0A")
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="VISION PRO", font=("Orbitron", 24, "bold"),
                     text_color=COR_DESTAQUE_HEX).pack(pady=20)

        # Menu de Abas
        for aba in ["INSPECAO", "TREINAR", "REFERENCIAS", "COMANDOS"]:
            ctk.CTkButton(self.sidebar, text=aba, font=("Roboto", 14, "bold"),
                          command=lambda a=aba: self.mudar_aba(a)).pack(pady=5, padx=20, fill="x")

        # Sliders de Calibração
        ctk.CTkLabel(self.sidebar, text="LIMIAR DE LUZ (Thresh)", font=("Roboto", 11)).pack(pady=(20, 0))
        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0, to=255, command=self.set_thresh)
        self.slider_thresh.set(self.valor_threshold)
        self.slider_thresh.pack(pady=5, padx=20)

        ctk.CTkLabel(self.sidebar, text="SENSIBILIDADE (Área)", font=("Roboto", 11)).pack(pady=(10, 0))
        self.slider_sens = ctk.CTkSlider(self.sidebar, from_=0, to=10000, command=self.set_sens)
        self.slider_sens.set(self.valor_sensibilidade)
        self.slider_sens.pack(pady=5, padx=20)

        # Lista de Peças (Scroll)
        ctk.CTkLabel(self.sidebar, text="LISTA DE MODELOS", font=("Roboto", 11, "bold")).pack(pady=(20, 5))
        self.scroll_ids = ctk.CTkScrollableFrame(self.sidebar, fg_color="#050505", height=300)
        self.scroll_ids.pack(fill="both", expand=True, padx=10, pady=10)

        # --- ÁREA PRINCIPAL ---
        self.main_area = ctk.CTkFrame(self, corner_radius=15, fg_color="#000000")
        self.main_area.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        self.lbl_nome_peca = ctk.CTkLabel(self.main_area, text="", font=("Roboto", 20, "bold"),
                                          text_color=COR_DESTAQUE_HEX)
        self.lbl_nome_peca.pack(pady=10)

        self.video_label = ctk.CTkLabel(self.main_area, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=5)
        self.video_label.bind("<Button-1>", self.gerir_clique_esquerdo)
        self.video_label.bind("<Button-3>", self.gerir_clique_direito)

        # --- DASHBOARD INFERIOR ---
        self.dash = ctk.CTkFrame(self, height=100, corner_radius=15, fg_color="#0A0A0A")
        self.dash.grid(row=1, column=1, padx=15, pady=(0, 15), sticky="ew")

        self.lbl_ok = ctk.CTkLabel(self.dash, text="OK: 0", text_color="#00FF96", font=("Roboto", 24, "bold"))
        self.lbl_ok.pack(side="left", padx=40)

        self.lbl_fail = ctk.CTkLabel(self.dash, text="FAIL: 0", text_color="#FF4B4B", font=("Roboto", 24, "bold"))
        self.lbl_fail.pack(side="left", padx=40)

        self.lbl_status = ctk.CTkLabel(self.dash, text="STATUS: OFF", font=("Roboto", 24, "bold"))
        self.lbl_status.pack(side="right", padx=50)

        # Inicialização
        self.inicializar_bd()
        self.carregar_dados_peca(1)
        self.criar_lista_pecas()
        self.cap = cv2.VideoCapture(0)

        # Atalhos de Teclado
        self.bind("<KeyPress-i>", lambda _: self.toggle_inspecao())
        self.bind("<KeyPress-s>", lambda _: self.comando_salvar_s())
        self.bind("<KeyPress-d>", lambda _: self.reset_poligonos_db())
        self.bind("<KeyPress-p>", lambda _: self.capturar_para_ia())
        self.bind("<KeyPress-r>", lambda _: self.reset_contadores())
        self.bind("<KeyPress-n>", lambda _: self.renomear_peca())

        self.atualizar_loop()

    # --- MÉTODOS DE APOIO ---
    def set_thresh(self, v):
        self.valor_threshold = int(float(v))

    def set_sens(self, v):
        self.valor_sensibilidade = int(float(v))

    @staticmethod
    def inicializar_bd():
        conn = sqlite3.connect(DB_NAME)
        conn.execute('''CREATE TABLE IF NOT EXISTS pecas
                        (
                            id         INTEGER PRIMARY KEY,
                            nome       TEXT,
                            poligonos  TEXT,
                            foto_ref   BLOB,
                            aprovadas  INTEGER DEFAULT 0,
                            rejeitadas INTEGER DEFAULT 0
                        )''')
        conn.commit()
        conn.close()

    def carregar_dados_peca(self, id_p):
        conn = sqlite3.connect(DB_NAME)
        p = conn.execute("SELECT id, nome, poligonos, foto_ref, aprovadas, rejeitadas FROM pecas WHERE id=?",
                         (id_p,)).fetchone()
        conn.close()
        if p:
            self.peca_ativa_id, self.peca_ativa_nome = p[0], p[1]
            self.lista_poligonos = json.loads(p[2]) if p[2] else []
            self.total_ok, self.total_fail = p[4], p[5]
            self.lbl_nome_peca.configure(text=f"MODELO ATIVO: {self.peca_ativa_nome}")
            self.lbl_ok.configure(text=f"OK: {self.total_ok}")
            self.lbl_fail.configure(text=f"FAIL: {self.total_fail}")

            if p[3]:
                nparr = np.frombuffer(p[3], np.uint8)
                self.foto_referencia_display = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                self.foto_referencia_display = None

    # --- LOOP DE VÍDEO E PROCESSAMENTO ---
    def atualizar_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display_frame = frame.copy()

            # Lógica por Aba
            if self.aba_ativa == "TREINAR":
                if self.frame_fixo_treino is not None:
                    display_frame = self.frame_fixo_treino.copy()
                    for poly in self.lista_poligonos:
                        cv2.polylines(display_frame, [np.array(poly)], True, (100, 100, 100), 1)
                    for pt in self.ponto_atual:
                        cv2.circle(display_frame, pt, 4, (0, 255, 255), -1)
                    if len(self.ponto_atual) > 1:
                        cv2.polylines(display_frame, [np.array(self.ponto_atual)], False, (0, 255, 255), 2)
                else:
                    for poly in self.lista_poligonos:
                        cv2.polylines(display_frame, [np.array(poly)], True, COR_OPENCV_BGR, 1)

            elif self.aba_ativa == "REFERENCIAS":
                if self.foto_referencia_display is not None:
                    display_frame = cv2.resize(self.foto_referencia_display, (w, h))
                else:
                    display_frame = np.zeros_like(frame)

            elif self.aba_ativa == "INSPECAO":
                # PROCESSAMENTO DE IMAGEM (Visão de Máquina)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                filt = self.clahe.apply(gray)
                _, thresh = cv2.threshold(filt, self.valor_threshold, 255, cv2.THRESH_BINARY_INV)

                # Mini monitor do Threshold (Canto Superior Esquerdo)
                thresh_mini = cv2.resize(thresh, (w // 5, h // 5))
                mini_bgr = cv2.cvtColor(thresh_mini, cv2.COLOR_GRAY2BGR)
                display_frame[10:h // 5 + 10, 10:w // 5 + 10] = mini_bgr
                cv2.rectangle(display_frame, (10, 10), (w // 5 + 10, h // 5 + 10), (0, 255, 0), 1)

                if self.inspecao_ativa:
                    self.status_inspecao = "OK"
                    for poly in self.lista_poligonos:
                        pts = np.array(poly)
                        mask = np.zeros_like(thresh)
                        cv2.fillPoly(mask, [pts], 255)

                        pixeis_alvo = cv2.countNonZero(cv2.bitwise_and(thresh, mask))

                        # Critério de Falha
                        if pixeis_alvo < self.valor_sensibilidade:
                            self.status_inspecao = "FAIL"
                            cv2.polylines(display_frame, [pts], True, (0, 0, 255), 3)
                            cv2.putText(display_frame, f"ERR:{pixeis_alvo}", (pts[0][0], pts[0][1] - 10), 1, 1,
                                        (0, 0, 255), 2)
                        else:
                            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                else:
                    self.status_inspecao = "OFF"
                    for poly in self.lista_poligonos:
                        cv2.polylines(display_frame, [np.array(poly)], True, COR_OPENCV_BGR, 1)

            # Renderização Final
            w_ui, h_ui = max(self.video_label.winfo_width(), 100), max(self.video_label.winfo_height(), 100)
            img_tk = ImageTk.PhotoImage(
                Image.fromarray(cv2.resize(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), (w_ui, h_ui))))
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)
            self.lbl_status.configure(text=f"STATUS: {self.status_inspecao}",
                                      text_color="#00FF96" if self.status_inspecao == "OK" else "#FF4B4B" if self.status_inspecao == "FAIL" else "white")

        self.after(15, self.atualizar_loop)

    # --- EVENTOS ---
    def mudar_aba(self, nome):
        self.aba_ativa = nome
        self.frame_fixo_treino = None
        self.ponto_atual = []

    def gerir_clique_esquerdo(self, e):
        if self.aba_ativa == "TREINAR" and self.frame_fixo_treino is not None:
            fw, fh = self.cap.get(3), self.cap.get(4)
            px = int(e.x * (fw / self.video_label.winfo_width()))
            py = int(e.y * (fh / self.video_label.winfo_height()))
            self.ponto_atual.append((px, py))

    def gerir_clique_direito(self, _):
        self.ponto_atual = []

    def comando_salvar_s(self):
        if self.aba_ativa == "TREINAR":
            if self.frame_fixo_treino is None:
                ret, f = self.cap.read()
                if ret: self.frame_fixo_treino = cv2.flip(f, 1)
            else:
                if len(self.ponto_atual) > 2:
                    self.lista_poligonos.append(list(self.ponto_atual))
                _, buffer = cv2.imencode('.jpg', self.frame_fixo_treino)
                conn = sqlite3.connect(DB_NAME)
                conn.execute("UPDATE pecas SET poligonos=?, foto_ref=? WHERE id=?",
                             (json.dumps(self.lista_poligonos), buffer.tobytes(), self.peca_ativa_id))
                conn.commit();
                conn.close()
                self.frame_fixo_treino = None
                self.ponto_atual = []
                self.carregar_dados_peca(self.peca_ativa_id)

    def reset_poligonos_db(self):
        if messagebox.askyesno("VisionPro", "Limpar desenhos deste modelo?"):
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET poligonos='[]' WHERE id=?", (self.peca_ativa_id,))
            conn.commit();
            conn.close()
            self.carregar_dados_peca(self.peca_ativa_id)

    def capturar_para_ia(self):
        ret, frame = self.cap.read()
        if ret and self.lista_poligonos:
            frame = cv2.flip(frame, 1)
            ts = int(time.time() * 1000)
            img_path = os.path.join(DIR_DATASET, f"peca_{self.peca_ativa_id}_{ts}.jpg")
            txt_path = os.path.join(DIR_DATASET, f"peca_{self.peca_ativa_id}_{ts}.txt")
            cv2.imwrite(img_path, frame)

            pts = np.array(self.lista_poligonos[0])
            x, y, w, h = cv2.boundingRect(pts)
            img_h, img_w = frame.shape[:2]
            cx, cy, nw, nh = (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h

            with open(txt_path, "w") as f:
                f.write(f"{self.peca_ativa_id - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            print(f"Dataset: Imagem salva para ID {self.peca_ativa_id - 1}")

    def criar_lista_pecas(self):
        for w in self.scroll_ids.winfo_children(): w.destroy()
        conn = sqlite3.connect(DB_NAME)
        data = conn.execute("SELECT id, nome FROM pecas").fetchall();
        conn.close()
        for pid, nome in data:
            ctk.CTkButton(self.scroll_ids, text=f"ID {pid:02} | {nome[:10]}", fg_color="#1a1a1a",
                          command=lambda x=pid: self.carregar_dados_peca(x)).pack(fill="x", pady=2)

    def toggle_inspecao(self):
        self.inspecao_ativa = not self.inspecao_ativa

    def reset_contadores(self):
        self.total_ok = self.total_fail = 0
        conn = sqlite3.connect(DB_NAME)
        conn.execute("UPDATE pecas SET aprovadas=0, rejeitadas=0 WHERE id=?", (self.peca_ativa_id,))
        conn.commit();
        conn.close()
        self.carregar_dados_peca(self.peca_ativa_id)

    def renomear_peca(self):
        n = simpledialog.askstring("VisionPro", "Novo Nome:")
        if n:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("UPDATE pecas SET nome=? WHERE id=?", (n, self.peca_ativa_id))
            conn.commit();
            conn.close()
            self.carregar_dados_peca(self.peca_ativa_id);
            self.criar_lista_pecas()


if __name__ == "__main__":
    app = VisionProSystem()
    app.mainloop()

# ==============================================================================
# MANUAL DE OPERAÇÃO PARA MÚLTIPLAS PEÇAS (IA)
# ==============================================================================
#
# 1. PASSO: TREINO GEOMÉTRICO (O QUE JÁ FAZES)
#    - Seleciona a PEÇA 01 na barra lateral.
#    - Vai à aba TREINAR e desenha o polígono (Tecla S).
#    - Repete isto para a PEÇA 02, 03, etc.
#
# 2. PASSO: COLETA DE DADOS PARA IA (SISTEMA HÍBRIDO)
#    - Com a PEÇA 01 selecionada, vai à aba INSPEÇÃO.
#    - Coloca a peça na frente da câmara em várias posições e carrega em [ P ]
#      pelo menos 50 vezes. O sistema salva automaticamente o ID da classe (0).
#    - Seleciona a PEÇA 02 na lista lateral. Repete a tecla [ P ].
#      O sistema salva automaticamente o ID da classe (1).
#
# 3. COMANDOS ADICIONADOS:
#    - [ P ] : Captura foto limpa e gera ficheiro .txt (Auto-Labeling).
#    - [ N ] : Renomeia o modelo selecionado para facilitar a organização.
#    - Pasta 'dataset_ia': Fica criada na raiz do projeto com todos os teus dados.
#
# 4. PRÓXIMO NÍVEL:
#    - Quando tiveres centenas de fotos, usaremos um script de treino para gerar o
#      ficheiro 'best.pt' que fará a deteção automática de qual peça está no ecrã.
# ==============================================================================
#
# ERROS CODIGO E COISAS A FAZER
#
# ==============================================================================
#
# - menu de treinar - nao consigo desenhar as zonas de inspeção
# - menu de referencias - nao esta a funcionar e nao aparece nada quando seleciono a peça
# - colocar uma aba de IA para ver as referencias qeu tirou para acertar todas as peças (4 ID's +/- 100 fotos por peça em diferentes fundos)
# - dentro da aba de IA colocar secçoes com as pastas das referencias, sendo uma forma de ver
# - ou abrindo um menu no centro do ecra, ou uma janela completamente nova apenas para ver as referencias, podendo escolher qual ver


# tentar medir o diametro de uma peça e verificar se a peça tem os furos abertos ou fechados

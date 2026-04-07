# spellchecker: disable
import cv2
import numpy as np
import sqlite3
import json
import winsound
import tkinter as tk
from tkinter import simpledialog
import time
import os
import csv
from datetime import datetime
import glob

# ==============================================================================
# 0. CONFIGURAÇÃO DO SISTEMA E CONSTANTES
# ==============================================================================
DB_NAME = "sistema_industrial.db"
CAMERA_ID = 0
PIXEL_PARA_MM = 0.12

# Cores e Estilos
COR_FUNDO = (15, 15, 15)
COR_MENU_BG = (25, 25, 25)
COR_DESTAQUE = (0, 255, 150)
COR_TEXTO = (220, 220, 220)
COR_TEXTO_INATIVO = (110, 110, 110)

# Cores para Detecção (HSV) - Trazido do main2.py
CORES_DETECCAO = {
    "AZUL": (np.array([85, 100, 50]), np.array([140, 255, 255]), (255, 0, 0)),
    "PRETO": (np.array([0, 0, 0]), np.array([180, 255, 50]), (60, 60, 60))
}

# ==============================================================================
# 1. GERENCIAMENTO DE ESTADO (GLOBAIS)
# ==============================================================================
# UI States
menu_expandido = True
menu_largura_full = 260
aba_selecionada = "INSPECAO"
mostrar_comandos = False
inspecao_ativa = False

# Model States (Carregados do BD)
peca_ativa_id = 1
peca_ativa_nome = "Peca 1"
cache_nomes = {}  # Cache para nomes das peças na UI
lista_poligonos = []
foto_referencia = None  # Imagem bruta armazenada
modelo_aprendido = None  # Contorno processado para MatchShapes
bordas_ref = 0
centro_ref = (0, 0)
angulo_ref = 0
total_ok = 0
total_fail = 0

# Runtime States
ponto_atual = []
status_inspecao = "OFF"
historico_inspecao = []  # Lista de dicts {'hora':, 'status':, 'cor':}
ultima_gravacao = 0
tolerancia_mm = 0.5
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Layout config
Y_OFFSET_BOTOES = 65
ALTURA_BOTAO = 45
Y_LISTA_MODELOS = 320
ALTURA_ITEM_MODELO = 32


# ==============================================================================
# 2. BANCO DE DADOS E PERSISTÊNCIA
# ==============================================================================
# noinspection PyDuplication
def inicializar_bd():
    global cache_nomes
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS pecas
                      (
                          id
                              INTEGER
                              PRIMARY
                                  KEY,
                          nome
                              TEXT,
                          poligonos
                              TEXT,
                          pontos_medicao
                              TEXT,
                          medida_ideal
                              REAL,
                          tolerancia
                              REAL,
                          escala
                              REAL,
                          foto_ref
                              BLOB,
                          aprovadas
                              INTEGER
                              DEFAULT
                                  0,
                          rejeitadas
                              INTEGER
                              DEFAULT
                                  0,
                          bordas_ref
                              INTEGER
                              DEFAULT
                                  0
                      )''')

    # Migração segura: verificar colunas existentes
    cursor.execute("PRAGMA table_info(pecas)")
    colunas_existentes = [info[1] for info in cursor.fetchall()]

    novas_colunas = {
        "aprovadas": "INTEGER DEFAULT 0",
        "rejeitadas": "INTEGER DEFAULT 0",
        "bordas_ref": "INTEGER DEFAULT 0"
    }

    for col, tipo in novas_colunas.items():
        if col not in colunas_existentes:
            cursor.execute(f"ALTER TABLE pecas ADD COLUMN {col} {tipo}")

    # Criar dados iniciais se vazio
    cursor.execute("SELECT COUNT(*) FROM pecas")
    if cursor.fetchone()[0] == 0:
        for i in range(1, 11):
            cursor.execute("INSERT INTO pecas (id, nome, medida_ideal, tolerancia, escala) VALUES (?, ?, ?, ?, ?)",
                           (i, f"Peca {i}", 5.0, 0.5, 0.12))
            cache_nomes[i] = f"Peca {i}"
    else:
        # Carregar cache de nomes
        cursor.execute("SELECT id, nome FROM pecas")
        for pid, pnome in cursor.fetchall():
            cache_nomes[pid] = pnome

    conn.commit()
    conn.close()


def carregar_dados_peca(id_peca):
    global lista_poligonos, foto_referencia, peca_ativa_id, peca_ativa_nome
    global total_ok, total_fail, bordas_ref, modelo_aprendido, tolerancia_mm

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pecas WHERE id=?", (id_peca,))
    p = cursor.fetchone()
    conn.close()

    if p:
        # Mapeamento baseado na ordem de criação/alteração.
        # ID=0, Nome=1, Polys=2, Pts=3, Ideal=4, Tol=5, Escala=6, Foto=7, OK=8, Fail=9, Bordas=10
        peca_ativa_id = p[0]
        peca_ativa_nome = p[1]
        lista_poligonos = json.loads(p[2]) if p[2] else []
        tolerancia_mm = p[5] if p[5] else 0.5

        # Carregar contadores (indices 8 e 9 assumindo ordem de criação)
        # Para segurança, vamos pegar pelo nome da coluna em query futura se necessario,
        # mas aqui assumiremos a ordem padrão do SQLite + Alters.
        try:
            total_ok = p[8]
            total_fail = p[9]
            bordas_ref = p[10] if len(p) > 10 else 0
        except IndexError:
            total_ok = 0
            total_fail = 0

        # Processar Foto de Referência para extrair Modelo
        if p[7]:
            nparr = np.frombuffer(p[7], np.uint8)
            foto_referencia = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            processar_referencia(foto_referencia)
        else:
            foto_referencia = None
            modelo_aprendido = None


def processar_referencia(img):
    """Gera o modelo matemático (contorno) a partir da foto salva para comparação"""
    global modelo_aprendido, centro_ref, angulo_ref

    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe_img = clahe.apply(blur)

    _, thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        modelo_aprendido = max(contornos, key=cv2.contourArea)
        rect = cv2.minAreaRect(modelo_aprendido)
        # Ajustar centro para coordenadas relativas se necessário, mas aqui usaremos absoluto da ref
        centro_ref = rect[0]
        angulo_ref = rect[2]


def salvar_progresso():
    """Salva estado atual (poligonos, foto, contadores) no BD"""
    if foto_referencia is None: return

    conn = sqlite3.connect(DB_NAME)
    _, buf = cv2.imencode('.jpg', foto_referencia)

    conn.execute("""
                 UPDATE pecas
                 SET poligonos=?,
                     foto_ref=?,
                     aprovadas=?,
                     rejeitadas=?,
                     bordas_ref=?
                 WHERE id = ?
                 """, (json.dumps(lista_poligonos), buf.tobytes(), total_ok, total_fail, bordas_ref, peca_ativa_id))

    conn.commit()
    conn.close()


def renomear_peca():
    global peca_ativa_nome
    root = tk.Tk()
    root.withdraw()
    novo = simpledialog.askstring("Renomear", f"Novo nome para ID {peca_ativa_id}:")
    root.destroy()

    if novo:
        conn = sqlite3.connect(DB_NAME)
        conn.execute("UPDATE pecas SET nome=? WHERE id=?", (novo, peca_ativa_id))
        conn.commit()
        conn.close()

        peca_ativa_nome = novo
        cache_nomes[peca_ativa_id] = novo
        carregar_dados_peca(peca_ativa_id)


# ==============================================================================
# 3. LÓGICA DE VISÃO COMPUTACIONAL
# ==============================================================================
def executar_inspecao(frame, roi_rect):
    global status_inspecao, total_ok, total_fail, ultima_gravacao

    rx1, ry1, rx2, ry2 = roi_rect
    roi = frame[ry1:ry2, rx1:rx2]

    # Pre-processamento
    roi_blur = cv2.bilateralFilter(roi, 9, 75, 75)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)
    roi_clahe = clahe.apply(gray)

    # Bordas
    bordas_detectadas = np.sum(cv2.Canny(roi_clahe, 30, 90) > 0)

    status_frame = "IDLE"
    motivo_fail = ""

    # 1. Coletar todos os objetos validos na cena
    objetos_candidatos = []

    for nome_cor, (low, high, cor_viz) in CORES_DETECCAO.items():
        mask = cv2.inRange(hsv, low, high)
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            area = cv2.contourArea(c)
            if area > 3000:
                objetos_candidatos.append((area, c, nome_cor))

    # 2. Processar apenas o maior objeto (Peça Principal)
    if objetos_candidatos:
        # Ordenar pelo maior area
        objetos_candidatos.sort(key=lambda x: x[0], reverse=True)
        area_principal, c_principal, cor_principal = objetos_candidatos[0]

        status_temp = "OK"

        # 1. Verificar Forma (MatchShapes)
        if modelo_aprendido is not None:
            match_val = cv2.matchShapes(modelo_aprendido, c_principal, 1, 0.0)
            if match_val > 0.20:  # Limite de similaridade
                status_temp = "FAIL"
                motivo_fail = f"Forma ({match_val:.2f})"

        # 2. Verificar Detalhes Internos (Bordas/Vedante)
        if cor_principal == "PRETO" and bordas_ref > 0:
            if bordas_detectadas < (bordas_ref * 0.5):
                status_temp = "FAIL"
                motivo_fail = "Vedante Ausente"

        # Desenhar resultado na ROI
        cor_rect = (0, 255, 0) if status_temp == "OK" else (0, 0, 255)
        cv2.drawContours(frame, [c_principal + [rx1, ry1]], -1, cor_rect, 2)

        # Nome da cor e status
        cv2.putText(frame, f"{cor_principal}", (rx1 + c_principal[0][0][0], ry1 + c_principal[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_rect, 2)

        # Atualizar Status Global
        status_frame = status_temp

        # Registrar Log (com delay para não flodar)
        if time.time() - ultima_gravacao > 2.0:
            agora = datetime.now().strftime("%H:%M:%S")
            historico_inspecao.insert(0, {'hora': agora, 'status': status_temp, 'cor': cor_principal})
            if len(historico_inspecao) > 10: historico_inspecao.pop()

            if status_temp == "OK":
                total_ok += 1
            else:
                total_fail += 1
                salvar_foto_erro(roi)

            ultima_gravacao = time.time()
            salvar_progresso()  # Salvar contadores

    return status_frame, motivo_fail


def salvar_foto_erro(roi):
    path = "deteccoes"
    if not os.path.exists(path): os.makedirs(path)
    filename = f"{path}/FAIL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, roi)


# ==============================================================================
# 4. INTERFACE GRÁFICA
# ==============================================================================
def desenhar_ui(frame):
    h, w = frame.shape[:2]
    largura_menu = menu_expandido and menu_largura_full or 60

    # --- MENU LATERAL ---
    cv2.rectangle(frame, (0, 0), (largura_menu, h), COR_FUNDO, -1)
    cv2.line(frame, (largura_menu, 0), (largura_menu, h), (40, 40, 40), 1)

    # Título da aplicação
    if menu_expandido:
        cv2.putText(frame, "VISION PRO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, "SYSTEM V1.0", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    # Botões
    botoes = ["INSPECAO", "TREINAR", "REFERENCIAS", "COMANDOS"]
    for i, nome in enumerate(botoes):
        y = Y_OFFSET_BOTOES + i * (ALTURA_BOTAO + 10)
        ativo = (aba_selecionada == nome)

        # Fundo do botão
        cor_btn = (35, 30, 25) if ativo else COR_MENU_BG
        cv2.rectangle(frame, (10, y), (largura_menu - 10, y + ALTURA_BOTAO), cor_btn, -1)

        # Indicador de seleção
        if ativo:
            cv2.rectangle(frame, (largura_menu - 14, y), (largura_menu - 10, y + ALTURA_BOTAO), COR_DESTAQUE, -1)

        # Texto
        texto = nome if menu_expandido else nome[:3]
        cv2.putText(frame, texto, (25, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COR_TEXTO if ativo else COR_TEXTO_INATIVO, 1)

    # Lista de Modelos (Aparece em TREINAR ou REFERENCIAS)
    if menu_expandido and aba_selecionada in ["TREINAR", "REFERENCIAS"]:
        cv2.putText(frame, "MODELOS DISPONIVEIS", (20, Y_LISTA_MODELOS - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 100, 100), 1)

        for i in range(1, 11):
            y_item = Y_LISTA_MODELOS + (i - 1) * (ALTURA_ITEM_MODELO + 5)
            if y_item > h - 40: break

            sel = (peca_ativa_id == i)
            cor_bg = (50, 50, 40) if sel else (20, 20, 20)

            cv2.rectangle(frame, (15, y_item), (largura_menu - 15, y_item + ALTURA_ITEM_MODELO), cor_bg, -1)
            if sel:
                cv2.rectangle(frame, (15, y_item), (20, y_item + ALTURA_ITEM_MODELO), COR_DESTAQUE, -1)

            nome_display = f"ID {i:02d} {cache_nomes.get(i, '??')[:12]}"
            cv2.putText(frame, nome_display, (30, y_item + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COR_TEXTO if sel else COR_TEXTO_INATIVO, 1)

    # --- ÁREA PRINCIPAL ---
    # Dashboard Superior
    cv2.rectangle(frame, (largura_menu, 0), (w, 50), COR_MENU_BG, -1)

    # Info de Status (Movido para baixo)
    cor_status = (0, 255, 0) if status_inspecao == "OK" else (
        (0, 0, 255) if status_inspecao == "FAIL" else (255, 255, 0))

    # Desenhar Status no canto inferior direito
    cv2.putText(frame, f"STATUS: {status_inspecao}", (w - 340, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2)

    stats = f"APROVADAS: {total_ok} | REJEITADAS: {total_fail} | TOL: +/-{tolerancia_mm}mm"
    cv2.putText(frame, stats, (largura_menu + 20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Painel Lateral Direito (Histórico) - Apenas na Inspeção
    if aba_selecionada == "INSPECAO":
        # Ajustado para dar mais espaço à zona de inspeção (antes era w-200)
        x_hist = w - 160
        cv2.rectangle(frame, (x_hist, 50), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "HISTORICO", (x_hist + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        for i, log in enumerate(historico_inspecao):
            y_log = 110 + i * 40
            cor_log = (0, 255, 100) if log['status'] == "OK" else (50, 50, 255)
            cv2.putText(frame, f"{log['hora']}", (x_hist + 10, y_log), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1)
            cv2.putText(frame, f"{log['status']} - {log['cor']}", (x_hist + 10, y_log + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        cor_log, 1)

    # Comandos Overlay
    if aba_selecionada == "COMANDOS" or mostrar_comandos:
        overlay = frame.copy()
        # Ajustado box de cmd para ocupar o reto do ecra
        cv2.rectangle(overlay, (largura_menu, 50), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.05, 0, frame)  # cor de fundo da aba de comandos
        # ajuste largura aba de comandos
        cv2.putText(frame, "COMANDOS DO SISTEMA", (largura_menu + 60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_DESTAQUE,
                    2)
        cv2.line(frame, (largura_menu + 40, 130), (largura_menu + 300, 130), COR_DESTAQUE, 1)

        cmds = [
            "[TREINAR] Mouse Esq : Desenhar Poligono",
            "[TREINAR] Mouse Dir : Limpar Desenho",
            "[TREINAR] Tecla 'S' : Salvar Referencia",
            "[INSPECAO] Tecla 'I' : Iniciar/Parar Inspecao",
            "[GERAL]   Tecla 'R' : Resetar Contadores",
            "[GERAL]   Tecla 'N' : Renomear Peca",
        ]

        for i, cmd in enumerate(cmds):
            # Ajustado posicionamento e espaçamento da lista de comandos (D e E, C e B), tamanho e grossura da letra
            cv2.putText(frame, cmd, (largura_menu + 30, 160 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_TEXTO, 1)


# ==============================================================================
# 5. INPUTS E EVENTOS
# ==============================================================================
def gerir_mouse(event, x, y, flags, param):
    global aba_selecionada, peca_ativa_id, ponto_atual, lista_poligonos, mostrar_comandos

    largura_menu = menu_expandido and menu_largura_full or 60

    if event == cv2.EVENT_LBUTTONDOWN:
        # Clique no Menu
        if x < largura_menu:
            # Botões Principais
            for i, nome in enumerate(["INSPECAO", "TREINAR", "REFERENCIAS", "COMANDOS"]):
                y_btn = Y_OFFSET_BOTOES + i * (ALTURA_BOTAO + 10)
                if y_btn <= y <= y_btn + ALTURA_BOTAO:
                    aba_selecionada = nome
                    return

            # Lista de Modelos (Se visível)
            if aba_selecionada in ["TREINAR", "REFERENCIAS"]:
                for i in range(1, 11):
                    y_item = Y_LISTA_MODELOS + (i - 1) * (ALTURA_ITEM_MODELO + 5)
                    if y_item <= y <= y_item + ALTURA_ITEM_MODELO:
                        carregar_dados_peca(i)
                        return

        # Clique na Área Principal
        else:
            if aba_selecionada == "TREINAR":
                ponto_atual.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if aba_selecionada == "TREINAR":
            ponto_atual = []
            lista_poligonos = []


def main():
    global ponto_atual, lista_poligonos, foto_referencia, bordas_ref, status_inspecao, total_ok, total_fail
    global inspecao_ativa, historico_inspecao

    inicializar_bd()
    carregar_dados_peca(1)

    cap = cv2.VideoCapture(CAMERA_ID)
    cv2.namedWindow("SISTEMA INTEGRADO", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SISTEMA INTEGRADO", gerir_mouse)

    msg_feedback = ""
    tempo_feedback = 0
    frame_fixo_treino = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Ajuste de Frame (Invertido conforme solicitado)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- NOVA LÓGICA DE EXIBIÇÃO (CONGELAMENTO) ---
        if aba_selecionada == "TREINAR":
            if frame_fixo_treino is not None:
                frame_ui = frame_fixo_treino.copy()
            else:
                frame_ui = frame.copy()
        else:
            # Fora do modo treinar, limpamos a foto fixa para a próxima sessão
            frame_fixo_treino = None
            frame_ui = frame.copy()

        # Definição Dinâmica da Área de Inspeção/Trabalho
        rx1, ry1 = int(menu_largura_full + 20), 80
        rx2, ry2 = int(w - 170), int(h - 50)

        # --- LÓGICA POR ABA ---
        if aba_selecionada == "INSPECAO":
            cv2.rectangle(frame_ui, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)

            if inspecao_ativa:
                # Executar CV
                status_temp, motivo = executar_inspecao(frame, (rx1, ry1, rx2, ry2))

                # Atualizar Status Visual
                if status_temp != "IDLE":
                    status_inspecao = status_temp

                # Desenhar sobreposição de erro se houver
                if motivo:
                    cv2.putText(frame_ui, f"ERRO: {motivo}", (rx1 + 10, ry1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
            else:
                status_inspecao = "OFF"
                cv2.putText(frame_ui, "INSPECAO PAUSADA (TECLA 'I')", (rx1 + 10, ry1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Desenhar Polígonos de Referência (Visualização)
            if lista_poligonos:
                for poly in lista_poligonos:
                    cv2.polylines(frame_ui, [np.array(poly)], True, (255, 200, 0), 1)

        elif aba_selecionada == "TREINAR":
            cv2.rectangle(frame_ui, (rx1, ry1), (rx2, ry2), (50, 50, 50), 1)
            txt_help = "PRIME 'S' PARA FIXAR IMAGEM" if frame_fixo_treino is None else "IMAGEM FIXA - DESENHA O POLIGONO"
            cv2.putText(frame_ui, txt_help, (rx1, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Desenhar Polígonos Salvos
            for poly in lista_poligonos:
                cv2.polylines(frame_ui, [np.array(poly)], True, (0, 255, 0), 2)

            # Desenhar Ponto Atual
            if len(ponto_atual) > 0:
                cv2.polylines(frame_ui, [np.array(ponto_atual)], False, (0, 255, 255), 2)

            if len(ponto_atual) > 1:
                cv2.line(frame_ui, ponto_atual[-1], ponto_atual[0], (0, 255, 255), 1)

        elif aba_selecionada == "REFERENCIAS":
            if foto_referencia is not None:
                # Mostrar thumbnail da referência salva
                h_ref, w_ref = foto_referencia.shape[:2]

                # Calcular escala para caber na área disponível
                largura_disp = rx2 - rx1 - 40
                altura_disp = ry2 - ry1 - 40

                if largura_disp > 0 and altura_disp > 0:
                    escala = min(altura_disp / h_ref, largura_disp / w_ref, 1.0)
                    novo_h, novo_w = int(h_ref * escala), int(w_ref * escala)

                    try:
                        thumb = cv2.resize(foto_referencia, (novo_w, novo_h))

                        # Centralizar
                        y_offset = ry1 + (altura_disp - novo_h) // 2
                        x_offset = rx1 + (largura_disp - novo_w) // 2

                        frame_ui[y_offset:y_offset + novo_h, x_offset:x_offset + novo_w] = thumb
                        cv2.putText(frame_ui, f"REFERENCIA: {peca_ativa_nome}", (x_offset, y_offset - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    except Exception as e:
                        cv2.putText(frame_ui, "ERRO AO MOSTRAR IMAGEM", (rx1 + 50, ry1 + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame_ui, "SEM FOTO DE REFERENCIA SALVA", (rx1 + 50, ry1 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # Feedback Visual Temporário (ex: "SALVO!")
        if time.time() < tempo_feedback:
            cv2.putText(frame_ui, msg_feedback, (int(w / 2) - 100, int(h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # --- FINALIZAÇÃO UI ---
        desenhar_ui(frame_ui)
        cv2.imshow("SISTEMA INTEGRADO", frame_ui)

        # --- TECLAS DE ATALHO ---
        k = cv2.waitKey(1)
        if k in (ord('i'), ord('I')):
            inspecao_ativa = not inspecao_ativa
            status_inspecao = "IDLE" if inspecao_ativa else "OFF"
            msg_feedback = "INSPECAO INICIADA" if inspecao_ativa else "INSPECAO PARADA"
            tempo_feedback = time.time() + 1.2
        elif k == ord('s') and aba_selecionada == "TREINAR":
            # 1. Se não houver imagem fixa, tira a foto agora
            if frame_fixo_treino is None:
                frame_fixo_treino = frame.copy()
                msg_feedback = "IMAGEM FIXADA!"
                tempo_feedback = time.time() + 1.2

            # 2. Se houver um polígono a ser desenhado, guarda-o
            if len(ponto_atual) > 2:
                lista_poligonos.append(ponto_atual)
                ponto_atual = []

            # 3. Salva a referência no BD e calcula bordas (evitando erro de bytes)
            # Captura a referência atual usando a ROI correta
            # Validar limites
            c_y1, c_y2 = max(0, ry1), min(h, ry2)
            c_x1, c_x2 = max(0, rx1), min(w, rx2)

            if c_x2 > c_x1 and c_y2 > c_y1:
                foto_referencia = frame[c_y1:c_y2, c_x1:c_x2].copy()

                # Calcula bordas para referência
                gray = cv2.cvtColor(foto_referencia, cv2.COLOR_BGR2GRAY)
                # Garantimos que bordas_ref é um número inteiro
                bordas_ref = int(np.sum(cv2.Canny(gray, 30, 90) > 0))

                salvar_progresso()
                processar_referencia(foto_referencia)
                if msg_feedback == "": msg_feedback = "GUARDADO!"
                tempo_feedback = time.time() + 1.5
            else:
                msg_feedback = "ERRO: AREA INVALIDA"
                tempo_feedback = time.time() + 2.0

        elif k == ord('n'):
            renomear_peca()
        elif k == ord('r'):
            total_ok = 0
            total_fail = 0
            historico_inspecao = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# verificar o inicio de inspeção tecla i nao esta configurada
# verificar os erros de OK e NOK
# diminuir a barra do historico
# verificar a parte das peças aprovadas, esta muito de fora linha 118
# verificar que na parte de inspeção quando meto a cor preta a frente da camara dá erro
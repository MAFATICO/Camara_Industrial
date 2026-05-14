"""
Script de teste para demonstrar a integração do sistema de falhas
Simula 10 inspeções (5 OK, 5 FAIL) e carrega dados no histórico
"""

import os
import sys
import sqlite3
import cv2
import numpy as np
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from falhas_manager import FalhasManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "COLOR_DETECTION/sistema_industrial.db")

def gerar_frame_teste():
    """Gera um frame de teste com dimensões realistas"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 30
    h, w = frame.shape[:2]

    # Adiciona alguns elementos de teste
    cv2.rectangle(frame, (50, 50), (w-50, h-50), (100, 200, 100), 2)
    cv2.putText(frame, "FRAME DE TESTE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def inserir_dados_teste():
    """Insere dados de teste na tabela de falhas"""

    # Cria tabela de pecas se não existir
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Cria peças de teste
        cursor.execute("INSERT OR IGNORE INTO pecas (id, nome, poligonos) VALUES (?, ?, ?)",
                      (1, "Peça A - Modelo 01", "[]"))
        cursor.execute("INSERT OR IGNORE INTO pecas (id, nome, poligonos) VALUES (?, ?, ?)",
                      (2, "Peça B - Modelo 02", "[]"))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao criar peças: {e}")

    # Inicializa o gerenciador de falhas
    falhas_manager = FalhasManager(DB_NAME)

    # Dados de teste
    motivos_fail = [
        "ROI 1: Baixo preenchimento",
        "Racha detectada",
        "Fora de tolerância",
        "Contorno incompleto",
        "Forma Errada"
    ]

    operadores = ["admin", "joao", "maria", "pedro"]

    print("📝 Inserindo dados de teste no histórico de falhas...")
    print("=" * 60)

    now = datetime.now()

    # Insere 10 registos de teste
    for i in range(10):
        peca_id = (i % 2) + 1  # Alterna entre peça 1 e 2
        operador = operadores[i % len(operadores)]
        status = "OK" if i < 5 else "FAIL"  # Primeiros 5 são OK, resto FAIL
        motivo = "---" if status == "OK" else motivos_fail[i % len(motivos_fail)]

        # Gera frame de teste
        frame = gerar_frame_teste()

        # Adiciona texto ao frame para identificar
        cv2.putText(frame, f"ID: {i+1} | Status: {status} | Op: {operador}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Timestamp com diferença de minutos
        data_hora = (now - timedelta(minutes=i*10)).strftime("%d/%m/%Y %H:%M:%S")

        # Registra a falha
        sucesso = falhas_manager.registar_falha(
            peca_id=peca_id,
            operador=operador,
            status=status,
            motivo=motivo,
            frame=frame
        )

        status_icon = "✅" if sucesso else "❌"
        print(f"{status_icon} Registo {i+1}: Peça {peca_id} | {operador:6} | {status:4} | {motivo[:30]:30}")

    print("=" * 60)
    print("\n✅ Dados de teste inseridos com sucesso!")
    print("\nAgora você pode:")
    print("  1. Executar: python \"menu falhas.py\" admin")
    print("  2. Selecionar as peças no ComboBox")
    print("  3. Ver o histórico de falhas com imagens")

if __name__ == "__main__":
    inserir_dados_teste()


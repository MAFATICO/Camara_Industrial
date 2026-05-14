"""
Gerenciador de Falhas - Integração entre sistema_industrial_final.py e menu falhas.py
Responsável por guardar e recuperar histórico de falhas com imagens
"""

import os
import sqlite3
import cv2
import json
from datetime import datetime

class FalhasManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.inicializar_tabela_falhas()

    def inicializar_tabela_falhas(self):
        """Cria a tabela de histórico de falhas se não existir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''CREATE TABLE IF NOT EXISTS historico_falhas (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                peca_id         INTEGER,
                operador        TEXT,
                data_hora       TEXT,
                status          TEXT,
                motivo          TEXT,
                imagem_blob     BLOB,
                imagem_path     TEXT,
                timestamp       REAL,
                FOREIGN KEY(peca_id) REFERENCES pecas(id)
            )''')

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"ERRO ao criar tabela de falhas: {e}")

    def registar_falha(self, peca_id, operador, status, motivo, frame, imagem_path=""):
        """
        Registra uma falha ou OK no histórico

        Args:
            peca_id: ID da peça sendo inspecionada
            operador: Username/ID do operador autenticado
            status: "OK" ou "FAIL"
            motivo: Descrição do motivo da falha (ou "---" para OK)
            frame: Frame OpenCV a ser guardado
            imagem_path: Caminho opcional da imagem
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Converte o frame para JPEG
            imagem_blob = None
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                imagem_blob = buffer.tobytes()

            data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            timestamp = datetime.now().timestamp()

            cursor.execute('''INSERT INTO historico_falhas 
                (peca_id, operador, data_hora, status, motivo, imagem_blob, imagem_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (peca_id, operador, data_hora, status, motivo, imagem_blob, imagem_path, timestamp))

            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"ERRO ao registar falha: {e}")
            return False

    def obter_historico_peca(self, peca_id, limite=50):
        """
        Recupera o histórico de falhas de uma peça

        Returns:
            Lista de tuples: (id, peca_id, operador, data_hora, status, motivo, timestamp)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''SELECT id, peca_id, operador, data_hora, status, motivo, timestamp
                FROM historico_falhas 
                WHERE peca_id = ?
                ORDER BY timestamp DESC
                LIMIT ?''', (peca_id, limite))

            resultados = cursor.fetchall()
            conn.close()
            return resultados
        except sqlite3.Error as e:
            print(f"ERRO ao recuperar histórico: {e}")
            return []

    def obter_imagem_falha(self, falha_id):
        """
        Recupera a imagem BLOB de uma falha

        Returns:
            bytes da imagem JPEG ou None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT imagem_blob FROM historico_falhas WHERE id = ?', (falha_id,))
            resultado = cursor.fetchone()
            conn.close()

            return resultado[0] if resultado else None
        except sqlite3.Error as e:
            print(f"ERRO ao recuperar imagem: {e}")
            return None

    def obter_falhas_por_motivo(self, peca_id):
        """
        Retorna um resumo de falhas agrupadas por motivo

        Returns:
            Dict: {motivo: contagem, ...}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''SELECT motivo, COUNT(*) as contagem 
                FROM historico_falhas 
                WHERE peca_id = ? AND status = 'FAIL'
                GROUP BY motivo
                ORDER BY contagem DESC''', (peca_id,))

            resultados = cursor.fetchall()
            conn.close()

            return {motivo: contagem for motivo, contagem in resultados}
        except sqlite3.Error as e:
            print(f"ERRO ao obter resumo de falhas: {e}")
            return {}

    def limpar_historico_peca(self, peca_id):
        """Limpa todo o histórico de uma peça"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM historico_falhas WHERE peca_id = ?', (peca_id,))
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"ERRO ao limpar histórico: {e}")
            return False


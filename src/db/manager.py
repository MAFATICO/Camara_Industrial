import sqlite3
import json
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_path="data/inspecao.db"):
        self.db_path = db_path
        self._create_tables()

    @contextmanager
    def connection(self):
        """Abre e fecha a ligação automaticamente e com segurança."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _create_tables(self):
        with self.connection() as conn:
            # Tabela para configurações (como o teu polígono)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS configs (
                    chave TEXT PRIMARY KEY,
                    valor TEXT
                )
            """)
            # Tabela para histórico de inspeções
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historico (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resultado TEXT,
                    motivo TEXT
                )
            """)
            # Tabela de Utilizadores
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usuarios (
                    username TEXT PRIMARY KEY,
                    password TEXT
                )
            """)
            # Criar admin padrão se necSessário
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM usuarios")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", ("admin", "1234"))


    def salvar_poligono(self, pontos):
        """Guarda a lista de pontos do polígono como uma string JSON."""
        with self.connection() as conn:
            conn.execute("INSERT OR REPLACE INTO configs (chave, valor) VALUES (?, ?)",
                         ("roi_polygon", json.dumps(pontos)))

    def carregar_poligono(self):
        """Recupera o polígono guardado."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT valor FROM configs WHERE chave = ?", ("roi_polygon",))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else []
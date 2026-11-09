"""Persist and display inspection failures in the Vision Pro application."""

import sqlite3

from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class FailureHistory:
    """SQLite storage for inspection failure events."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.initialize()

    def initialize(self):
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS falhas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    model_id INTEGER,
                    model_name TEXT,
                    operator TEXT,
                    reason TEXT NOT NULL
                )
                """
            )

    def record(self, model_id, model_name, operator, reason):
        """Record one failed-piece event."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO falhas (model_id, model_name, operator, reason)
                VALUES (?, ?, ?, ?)
                """,
                (
                    model_id,
                    model_name,
                    operator,
                    reason or "Falha sem motivo especificado",
                ),
            )

    def latest(self, limit=500):
        """Return the most recent failure events."""
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                """
                SELECT detected_at, model_id, model_name, operator, reason
                FROM falhas
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()


class FailureHistoryDialog(QDialog):
    """Read-only window showing when and why parts failed inspection."""

    HEADERS = ("Data/hora", "ID modelo", "Modelo", "Operador", "Motivo")

    def __init__(self, history, parent=None):
        super().__init__(parent)
        self.history = history
        self.setWindowTitle("Histórico de falhas")
        self.resize(900, 500)

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self):
        rows = self.history.latest()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self.table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem("" if value is None else str(value)),
                )

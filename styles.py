"""
Функции для настройки палитры, шрифтов, и чтения CSS файлов
"""

from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtWidgets import QApplication, QToolTip

def set_dark_theme(app: QApplication):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(15, 15, 15))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)


def set_fonts(app: QApplication):
    app.setFont(QFont("Segoe UI", 12))
    QToolTip.setFont(QFont('SansSerif', 10))


def load_stylesheet(filepath: str):
    """Загружает стили из CSS файла."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Файл {filepath} не найден.")
        return ""
    
def set_light_theme(app):
    palette = QPalette()
    # Основные цвета
    palette.setColor(QPalette.Window, QColor(240, 240, 240))  # Светлый фон
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))  # Черный текст
    palette.setColor(QPalette.Base, QColor(255, 255, 255))  # Фон для ввода текста
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    # Цвета текста
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))  # Яркий текст (ошибки, предупреждения)
    # Цвет выделения
    palette.setColor(QPalette.Highlight, QColor(76, 163, 224))  # Синий цвет выделения
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # Белый текст на выделении
    
    app.setPalette(palette)
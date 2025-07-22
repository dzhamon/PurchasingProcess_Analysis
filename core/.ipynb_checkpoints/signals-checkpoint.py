from PyQt5.QtCore import QObject, pyqtSignal

class PlotEmitter(QObject):
    """Класс для передачи данных графика между потоками"""
    plot_ready = pyqtSignal(str, object)  # str: название графика, object: данные
    save_plot = pyqtSignal(str, str, object)  # путь, название, данные
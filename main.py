import json
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QFont, QCursor
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QAction,
    QLabel,
    QStatusBar,
    QProgressBar,
    QTabWidget,
    QVBoxLayout,
    QMessageBox,
    QWidget,
    QFileDialog,
)
from PyQt5.QtWidgets import QToolTip

# Импорт AlternativeSuppliersAnalyzer
from models_analyses.find_alternative_suppliers_enhanced import (
    AlternativeSuppliersAnalyzer, export_alternative_suppliers_to_excel
)

from selection_dialog import SelectionDialog
from styles import set_light_theme, set_fonts, load_stylesheet
from utils.clean_datas import clean_database
from utils.data_model import DataModel
from utils.visualizer import KPIVisualizer
from widgets.module_tab1 import Tab1Widget
from widgets.module_tab2 import Tab2Widget
from widgets.module_tab3 import Tab3Widget
from widgets.module_tab4 import Tab4Widget
from widgets.module_tab5 import Tab5Widget
from widgets.module_tab6 import Tab6Widget


# Загрузка подсказок из JSON-файла
def load_menu_hints():
    with open("menu_hints.json", "r", encoding="utf-8") as file:
        return json.load(file)


def clicked_connect(self):
    """эта функция вызывает метод open_file из класса Data_model модуля data_model.py"""
    DataModel(self).open_file_dialog()


class AnalysisThread(QThread):
    update_progress = pyqtSignal(int)  # сигнал для обновления прогресса

    def __init__(self, analysis_method, create_plot=False, *args, **kwargs):
        super().__init__()
        self.analysis_method = analysis_method
        self.create_plot = create_plot
        self.args = args
        self.kwargs = kwargs

    def run(self):
        # Передаем сигнал update_progress в метод анализа
        self.analysis_method(
            update_progress=self.update_progress,
            create_plot=self.create_plot,
            *self.args,
            **self.kwargs,
        )


class MyTabWidget(QWidget):
    def __init__(self):
        super().__init__()
        # self.df_kpi_normalized = None
        self.notebook = QTabWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.notebook)
        self.setLayout(layout)

        # Инициализация QLabel для отображения подсказок
        self.tooltip_label = QLabel(self)
        self.tooltip_label.setStyleSheet(
            "background-color: yellow; color: black; font-size: 12px; padding: 5px; border: 1px solid black;"
        )
        self.tooltip_label.hide()  # Скрываем по умолчанию

        # Вызов метода setup_tabs с параметрами
        self.setup_tabs()

    def showTooltip(self, text, x=20, y=20):
        try:
            # Устанавливаем текст подсказки и показываем ее в фиксированной позиции для отладки
            self.tooltip_label.setText(text)
            cursor_pos = QCursor.pos()
            self.tooltip_label.move(self.mapFromGlobal(cursor_pos + QPoint(x, y)))
            self.tooltip_label.adjustSize()
            self.tooltip_label.show()
        except Exception as error:
            print("Ошибка при отображении подсказки:", error)

    def hideTooltip(self):
        # Скрываем виджет подсказки
        self.tooltip_label.hide()

    def handle_analysis_data(self, df):
        """Слот для получения данных из Tab2"""
        self._current_filtered_df = df.copy()
        print(f"Данные получены. Размер: {df.shape}")

    def setup_tabs(self):
        # создание отдельных вкладок
        tab1 = Tab1Widget()
        params_for_tab2 = [
            "lot_number",
            "project_name",
            "discipline",
            "actor_name",
            "winner_name",
            "currency",
            "good_name",
        ]
        tab2 = Tab2Widget(params_for_tab2)
        tab2.data_ready_for_analysis.connect(self.handle_analysis_data)
        tab3 = Tab3Widget()
        params_for_tab4 = [
            "lot_number",
            "discipline",
            "project_name",
            "executor_dak",
            "counterparty_name",
            "product_name",
            "contract_currency",
        ]
        tab4 = Tab4Widget(params_for_tab4)
        tab4.data_ready_for_analysis.connect(self.handle_analysis_data)

        self._current_filtered_df = None

        tab5 = Tab5Widget()

        params_for_tab6 = [
            "warehouse",
            "nomenclature",
            "currency",
            "stock_category",
            "department",
            "project_name",
            "date_column",
        ]
        tab6 = Tab6Widget(params_for_tab6)

        self.notebook.addTab(tab1, "Данные по Лотам")
        self.notebook.addTab(tab2, "Параметры загруженных Лотов")
        self.notebook.addTab(tab3, "Данные по Контрактам")
        self.notebook.addTab(tab4, "Параметры загруженных Контрактов")
        self.notebook.addTab(tab5, "Остатки складов")
        self.notebook.addTab(tab6, "Параметры складов")

        # подключаем сигнал для взаимодействия между вкладками
        tab1.filtered_data_changed.connect(tab2.update_data)
        tab3.filtered_data_changed.connect(tab4.update_data)
        tab5.filtered_data_changed.connect(tab6.update_data)
        print("MyTabWidget: Вкладки успешно созданы")  # Отладочный принт


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_df = None
        self.contract_df = None
        self.filtered_df = None
        self._hhi_error_message = None
        self._hhi_success_message = None
        self._hhi_results = None
        
        # Загрузка подсказок
        self.menu_hints = load_menu_hints()

        # Создание и установка вкладок
        self.tab_widget = MyTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Подключение сигнала для обновления данных между вкладками
        tab1_widget = self.tab_widget.notebook.widget(0)  # Получаем первый виджет вкладки
        if isinstance(tab1_widget, Tab1Widget):
            tab1_widget.filtered_data_changed.connect(self.update_tab2_data)

        # Подключение сигнала для получения отфильтрованных данных
        tab2_widget = self.tab_widget.notebook.widget(
            1
        )  # Получаем второй виджет вкладки (Tab2)
        if isinstance(tab2_widget, Tab2Widget):
            tab2_widget.data_ready_for_analysis.connect(self.set_filtered_data)

        # Подключение сигнала для обновления данных между вкладками
        tab3_widget = self.tab_widget.notebook.widget(2)  # получаем Tab3Widget
        # if isinstance(tab3_widget, Tab3Widget):
        # 	tab3_widget.filtered_data_changed.connect(self.update_tab3_data())
        tab4_widget = self.tab_widget.notebook.widget(3)
        # if isinstance(tab3_widget, Tab3Widget) and isinstance(tab4_widget, UniversalTabWidget):
        # 	tab3_widget.filtered_data_changed.connect(tab4_widget.on_filtered_contracts_received)
        if isinstance(tab4_widget, Tab4Widget):
            tab4_widget.data_ready_for_analysis.connect(self.set_filtered_data)

        tab5_widget = self.tab_widget.notebook.widget(4)
        if isinstance(tab5_widget, Tab5Widget):
            tab5_widget.filtered_data_changed.connect(self.set_filtered_data)

        # --- ИНИЦИАЛИЗАЦИЯ AlternativeSuppliersAnalyzer И ПОДКЛЮЧЕНИЕ СИГНАЛА ---
        self.alternative_suppliers_analyzer = AlternativeSuppliersAnalyzer()
        if isinstance(tab3_widget, Tab3Widget):
            # Подключаем сигнал my_custom_signal из Tab3Widget к слоту анализатора
            tab3_widget.my_custom_signal.connect(
                self.alternative_suppliers_analyzer.receive_contract_data
            )
            print(
                "Window: Соединение Tab3Widget.my_custom_signal -> AlternativeSuppliersAnalyzer.receive_contract_data установлено."
            )
        # --- КОНЕЦ ИНИЦИАЛИЗАЦИИ И ПОДКЛЮЧЕНИЯ ---

        # Настройка главного окна
        self.setFont(QFont("Arial", 12))
        self.setWindowTitle("Анализ закупочных процессов")
        self.resize(1200, 600)
        self.setFont(QFont("Arial", 11))

        # Создание меню и действий
        self._createActions()
        self._createMenuBar()
        self._connectActions()

        # Создание статусной строки и прогрессбара
        self.status_bar = QStatusBar(self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)

    def update_tab2_data(self, filtered_df):
        self.tab_widget.notebook.widget(1).update_data(filtered_df)

    def update_tab3_data(self, filtered_df):
        # Логика обновления данных на вкладке 3
        self.tab_widget.notebook.widget(2).update_data(filtered_df)
        print("Данные для вкладки 3 обновлены")

    def _createMenuBar(self):
        menuBar = self.menuBar()

        # Меню Файл
        fileMenu = menuBar.addMenu("Ввод основной информации и выход из программы")
        fileMenu.addAction(self.ContrAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.ExitAction)

        # Меню Анализ по Лотам
        analysisMenu = menuBar.addMenu("Анализ данных по Лотам")
        analysisMenu.addAction(self.analyzeMonthlyExpensesAction)
        analysisMenu.addAction(self.analyzeTopSuppliersAction)
        analysisMenu.addAction(self.analyzeClasterAction)
        analysisMenu.addAction(self.suppliersfriquencyAction)
        analysisMenu.addAction(self.networkanalyseAction)
        analysisMenu.addAction(self.analyzeKPIAction)
        analysisMenu.addAction(self.efficiency_analyses_action)
        analysisMenu.addAction(self.suppliers_by_unit_price_action)
        analysisMenu.addAction(self.find_cross_discipline_lotsAction)
        analysisMenu.addAction(self.lotcount_peryearAction)

        # Меню Анализ по Контрактам
        analysisMenuContract = menuBar.addMenu("Анализ данных по Контрактам")
        analysisMenuContract.addAction(self.analyzeNoneEquilSums)
        analysisMenuContract.addAction(self.trend_analyses_action)
        analysisMenuContract.addAction(self.prophet_arima_action)
        analysisMenuContract.addAction(self.contracts_less_dates_action)
        analysisMenuContract.addAction(self.herfind_hirshman_action)
        # Отдельный пункт для анализа альтернативных поставщиков
        analysisMenuContract.addAction(self.run_alternative_suppliers_action)
        
        # Меню Анализ данных по Складам
        analysisMenuWarehouses = menuBar.addMenu("Анализ данных по Складам")
        analysisMenuWarehouses.addAction(self.warehouseStatistics)

    def setActionTooltip(self, action, group, hint_key, x=0, y=0):
        hint_text = self.menu_hints.get(group, {}).get(
            hint_key, "Нет инструкции для этого пункта"
        )
        action.hovered.connect(lambda: self.tab_widget.showTooltip(hint_text))
        action.triggered.connect(self.tab_widget.hideTooltip)

    def leaveEvent(self, event):
        self.tab_widget.hideTooltip()
        super().leaveEvent(event)

    def _createActions(self):
        # Действия для меню Файл
        self.ContrAction = QAction("Загрузить данные из Отчетов", self)
        # self.CleanDatas = QAction("Очистить данные в БД", self)
        # self.GetBasData = QAction("Получить основные данные", self)
        self.ExitAction = QAction("Выход", self)

        self.statusBar().showMessage("Все ОК")

        # Действия для меню Анализ данных по Лотам
        self.analyzeMonthlyExpensesAction = QAction("Анализ месячных затрат", self)
        self.setActionTooltip(
            self.analyzeMonthlyExpensesAction,
            "Анализ данных по Лотам",
            "menu_item_1",
            x=20,
            y=20,
        )

        self.analyzeTopSuppliersAction = QAction("Анализ топ-10 поставщиков", self)
        self.setActionTooltip(
            self.analyzeTopSuppliersAction,
            "Анализ данных по Лотам",
            "menu_item_2",
            x=20,
            y=20,
        )

        self.analyzeClasterAction = QAction("Кластерный анализ", self)
        self.setActionTooltip(
            self.analyzeClasterAction,
            "Анализ данных по Лотам",
            "menu_item_3",
            x=20,
            y=20,
        )

        self.suppliersfriquencyAction = QAction("Анализ частоты поставщиков", self)
        self.setActionTooltip(
            self.suppliersfriquencyAction,
            "Анализ данных по Лотам",
            "menu_item_4",
            x=20,
            y=20,
        )

        self.networkanalyseAction = QAction("Сетевой анализ проектов", self)
        self.setActionTooltip(
            self.networkanalyseAction,
            "Анализ данных по Лотам",
            "menu_item_5",
            x=20,
            y=20,
        )

        self.analyzeKPIAction = QAction("Анализ KPI", self)
        self.setActionTooltip(
            self.analyzeKPIAction, "Анализ данных по Лотам", "menu_item_6", x=20, y=20
        )

        self.efficiency_analyses_action = QAction(
            "Анализ эффективности исполнителей и поиск аномалий", self
        )
        self.setActionTooltip(
            self.efficiency_analyses_action,
            "Анализ данных по Лотам",
            "menu_item_7",
            x=20,
            y=20,
        )
        self.suppliers_by_unit_price_action = QAction(
            "Ранжирование Поставщиков по цене за единицу товара", self
        )
        self.setActionTooltip(
            self.suppliers_by_unit_price_action,
            "Анализ данных по Лотам",
            "menu_item_8",
            x=20,
            y=20,
        )
        self.find_cross_discipline_lotsAction = QAction(
            "Поиск и анализ лотов общих для разных дисциплин", self
        )
        self.setActionTooltip(
            self.find_cross_discipline_lotsAction,
            "Анализ данных по Лотам",
            "menu_item_9",
            x=20,
            y=20,
        )

        self.lotcount_peryearAction = QAction(
            "Количество лотов по дисциплинам по-квартально", self
        )
        self.setActionTooltip(
            self.lotcount_peryearAction,
            "Анализ данных по Лотам",
            "menu_item_10",
            x=20,
            y=20,
        )
        # ================================================
        # Действия для меню Анализ данных по Контрактам
        self.analyzeNoneEquilSums = QAction(
            "Поиск и анализ несоответствий в суммах Лотов и Контрактов", self
        )
        self.setActionTooltip(
            self.analyzeNoneEquilSums,
            "Анализ данных по Контрактам",
            "menu_item_1",
            x=450,
            y=20,
        )

        self.trend_analyses_action = QAction("Тренд - анализ", self)
        self.setActionTooltip(
            self.trend_analyses_action,
            "Анализ данных по Контрактам",
            "menu_item_2",
            x=450,
            y=20,
        )

        self.prophet_arima_action = QAction("Моделирование и прогнозирование", self)
        self.setActionTooltip(
            self.prophet_arima_action,
            "Анализ данных по Контрактам",
            "menu_item_3",
            x=450,
            y=20,
        )

        self.contracts_less_dates_action = QAction(
            "Поиск и анализ контрактов с инвалидными датами ", self
        )
        self.setActionTooltip(
            self.contracts_less_dates_action,
            "Анализ данных по Контрактам",
            "menu_item_4",
            x=450,
            y=20,
        )
        self.herfind_hirshman_action = QAction("Метод Херфиндаля-Хиршмана", self)
        self.setActionTooltip(
            self.herfind_hirshman_action,
            "Анализ данных по Контрактам",
            "menu_item_5",
            x=450,
            y=20,
        )
        
        # Определение действия для запуска анализа альтернативных поставщиков
        self.run_alternative_suppliers_action = QAction(
            "Анализ альтернативных поставщиков", self
        )
        self.setActionTooltip(
            self.run_alternative_suppliers_action,
            "Анализ альтернативных поставщиков",
            "menu_item_6",
            x=450,
            y=20,
        )
        # ===================================================

        # Действия для меню Анализ данных по Складам
        self.warehouseStatistics = QAction(
            "Расчет остатков сум на складах по валютам", self
        )

    def _connectActions(self):
        # Подключение сигналов к действиям
        self.ContrAction.triggered.connect(self.load_sql_data)
        # self.CleanDatas.triggered.connect(self.run_clean_data)
        self.ExitAction.triggered.connect(self.close)

        # Подключение сигналов к методам Анализа данных по Лотам
        self.analyzeMonthlyExpensesAction.triggered.connect(
            self.run_analyze_monthly_cost
        )
        self.analyzeTopSuppliersAction.triggered.connect(self.run_analyze_top_suppliers)
        self.analyzeClasterAction.triggered.connect(self.run_ClusterAnalyze)
        self.suppliersfriquencyAction.triggered.connect(
            self.run_analyze_supplier_friquency
        )
        self.networkanalyseAction.triggered.connect(self.run_network_analysis)
        self.analyzeKPIAction.triggered.connect(self.run_kpi_analysis)
        self.efficiency_analyses_action.triggered.connect(self.run_efficiency_analyses)
        self.suppliers_by_unit_price_action.triggered.connect(
            self.run_analyze_by_unit_price
        )
        self.find_cross_discipline_lotsAction.triggered.connect(
            self.run_find_cross_discipline_lots
        )
        self.lotcount_peryearAction.triggered.connect(self.run_lotcount_peryear)

        # Подключение сигналов к методам Анализа данных по Контрактам
        self.analyzeNoneEquilSums.triggered.connect(self.run_analyzeNonEquilSums)
        self.trend_analyses_action.triggered.connect(self.run_trend_analyses)
        self.prophet_arima_action.triggered.connect(self.run_prophet_and_arima)
        self.contracts_less_dates_action.triggered.connect(
            self.run_contracts_less_dates
        )
        # Подключение метода Херфиндаля-Хиршмана к его запуску
        self.herfind_hirshman_action.triggered.connect(
            self.run_herfind_hirshman_analysis
        )
        # Подключение метода анализа альтернативных поставщиков
        self.run_alternative_suppliers_action.triggered.connect(
            self.run_alternative_suppliers_for_major_suppliers
        )

        # Подключение сигналов к методам Анализа данных по Складам
        self.warehouseStatistics.triggered.connect(self.run_warehouseStatistics)

    def set_filtered_data(self, df):
        """Устанавливает отфильтрованный DataFrame для анализа."""
        self._current_filtered_df = df.copy()
        print(f"Данные сохранены. Размер: {self._current_filtered_df.shape}")
        QMessageBox.information(
            self, "Информация", "Данные для анализа успешно обновлены."
        )

    def run_clean_data(self):
        clean_database()

    def start_analysis(self, analysis_task, on_finished_callback, create_plot=False, *args, **kwargs):
        """
        Унифицированный метод для запуска анализа в отдельном потоке
        :param analysis_task: Метод, который будет выполнять анализ.
        :param on_finished_callback: Метод, который будет вызван после завершения анализа.
        :param create_plot: флаг для создания графика (передается в AnalysisThread
        :param args: Позиционные аргументы для analysis_task
        :param kwargs: Именованные аргументы для analysis_task
        """
        # Сброс прогресс-бара
        self.progress_bar.setValue(0)
        self.progress_bar.show()  # Показываем прогресс-бар

        # Создание и настройка потока с передачей всех аргументов
        self.analysis_thread = AnalysisThread(
            analysis_task, create_plot, *args, **kwargs)
        self.analysis_thread.update_progress.connect(
            self.progress_bar.setValue)  # Обновление прогресс-бара
        self.analysis_thread.finished.connect(
            on_finished_callback)  # Уведомление о завершении

        # Запуск потока
        self.analysis_thread.start()

    def on_analysis_finished(self):
        # Эта функция будет вызвана по завершении анализа
        self.progress_bar.setValue(100)
        self.progress_bar.hide()  # Скрываем прогресс-бар
        QMessageBox.information(self, "Завершено", "Анализ завершен!")

    def run_kpi_analysis(self):
        """Запуск анализа KPI с использованием отфильтрованных данных."""
        if self._current_filtered_df is not None:
            self.progress_bar.setValue(0)
            from models_analyses.MyLotAnalyzeKPI import LotAnalyzeKPI

            # Передаем отфильтрованные данные в KPI анализатор
            kpi_analyzer = LotAnalyzeKPI(self._current_filtered_df)
            self.df_kpi_normalized = kpi_analyzer.calculate_kpi(
                self._current_filtered_df
            )

            # Визуализация KPI
            self.visualize_kpi()
            QMessageBox.information(self, "KPI Анализ", "KPI анализ успешно завершен.")
        else:
            QMessageBox.warning(
                self, "Ошибка", "Нет отфильтрованных данных для анализа KPI."
            )

    def visualize_kpi(self):
        """Вызов визуализации KPI."""
        if hasattr(self, "df_kpi_normalized") and self.df_kpi_normalized is not None:
            visualizer = KPIVisualizer(self.df_kpi_normalized)

            # Создаем диалог для выбора типа визуализации
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Выбор типа визуализации")
            dialog.setText("Выберите тип визуализации KPI:")
            bar_btn = dialog.addButton("Бар-чарт", QMessageBox.ActionRole)
            pie_btn = dialog.addButton("Круговая диаграмма", QMessageBox.ActionRole)
            heatmap_btn = dialog.addButton("Тепловая карта", QMessageBox.ActionRole)
            line_btn = dialog.addButton("Линейный график", QMessageBox.ActionRole)
            dialog.exec_()

            clicked_button = dialog.clickedButton()

            if clicked_button == bar_btn:
                visualizer.plot_bar_chart()
            elif clicked_button == pie_btn:
                visualizer.plot_pie_chart()
            elif clicked_button == heatmap_btn:
                visualizer.plot_heatmap()
            elif clicked_button == line_btn:
                visualizer.plot_line_chart()
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных KPI для визуализации.")

    def run_analyze_monthly_cost(self):
        # метод для анализа месячных затрат
        if self._current_filtered_df is not None:
            print("Данные для анализа (месячные затраты):")
            # Используем минимальную и максимальную даты из отфильтрованных данных
            start_date = self._current_filtered_df["close_date"].min()
            end_date = self._current_filtered_df["close_date"].max()
            from models_analyses.analysis import analyze_monthly_cost

            analyze_monthly_cost(self, self._current_filtered_df, start_date, end_date)
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def run_analyze_top_suppliers(self):
        #  метод анализа поставщиков с высокими и низкими ценами за единицу товара
        if self._current_filtered_df is not None:
            """Используем минимальную и максимальную даты из отфильтрованных данных"""
            start_date = self._current_filtered_df["close_date"].min()
            end_date = self._current_filtered_df["close_date"].max()
            uniq_project_name = self._current_filtered_df["project_name"].unique()
            # Проверяем, что уникальное значение только одно
            if len(uniq_project_name) == 1:
                project_name = (
                    uniq_project_name.item()
                )  # Извлекаем значение из массива как строку
            else:
                raise ValueError(
                    f"Ожидалось одно уникальное значение project_name, но найдено: {uniq_project_name}"
                )

            from models_analyses.analysis import analyze_top_suppliers

            # здесь логика для анализа данных
            analyze_top_suppliers(
                self, self._current_filtered_df, start_date, end_date, project_name
            )
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def run_ClusterAnalyze(self):
        # Метод для классификации исполнителей с обучением методом SeedKMeans
        if self._current_filtered_df is not None:
            from models_analyses.MyLotAnalyzeKPI import LotAnalyzeKPI
            from models_analyses.SeedKMeans_clustering import (
                SeedKMeansClustering,
                export_to_excel,
            )
            from models_analyses.SeedKMeans_clustering import analysis_df_clusters
            import logging
            import os

            output_dir = "D:\Analysis-Results\clussification_analysis"
            os.makedirs(output_dir, exist_ok=True)

            # Создаем объект KPI-анализатора
            kpi_analyzer = LotAnalyzeKPI(self._current_filtered_df)

            # Создаем объект для кластеризации, передавая KPI-анализатор
            clustering_module = SeedKMeansClustering(kpi_analyzer)
            df_clusters, kmeans_model = clustering_module.perform_clustering()

            if df_clusters is not None:
                # Сохранение гистограммы
                histogram_path = os.path.join(output_dir, "cluster_distribution.png")
                clustering_module.plot_cluster_distribution(df_clusters, histogram_path)

                # Сохранение данных в Excel
                excel_path = os.path.join(output_dir, "cluster_analysis_report.xlsx")
                export_to_excel(df_clusters, excel_path)
                analysis_df_clusters(df_clusters)

                logging.info(
                    f"Кластерный анализ завершен. Результаты сохранены в {output_dir}"
                )
            else:
                logging.error("Кластерный анализ завершился ошибкой.")
        else:
            print("Нет данных для анализа")

    def run_analyze_supplier_friquency(self):
        # Метод для анализа частоты выбора поставщиков
        if self._current_filtered_df is not None:
            from models_analyses.analysis import analyze_supplier_frequency

            analyze_supplier_frequency(self._current_filtered_df)
            self.on_analysis_finished()
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def run_network_analysis(self):
        # Метод для сетевого анализа
        if self._current_filtered_df is not None:
            print("Запуск сетевого анализа")
            from models_analyses.analysis import network_analysis

            network_analysis(self, self._current_filtered_df)
            self.on_analysis_finished()
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def load_sql_data(self):
        print("Загрузка данных в базу SQL...")
        DataModel(self).open_file_dialog()

    # Логика работы с Анализом данных по Контрактам

    def run_analyzeNonEquilSums(self):
        # Метод для поиска несоответствий в суммах Лотов и Контрактов
        print("Запуск поиска несоответствий сумм")
        from models_analyses.analyze_contracts import analyzeNonEquilSums

        analyzeNonEquilSums(self, self._current_filtered_df)
        self.on_analysis_finished()

    def run_trend_analyses(self):
        if not hasattr(self, "_current_filtered_df") or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа!")
            return
        else:
            from models_analyses.analyze_contracts import (
                data_preprocessing_and_analysis,
            )

            # Удалим в датафрейме self._current_filtered_df строки-дубликаты если они есть
            self._current_filtered_df = self._current_filtered_df.drop_duplicates()
            df_merged = data_preprocessing_and_analysis(self._current_filtered_df)

            dialog = SelectionDialog(df_merged=df_merged, parent=self)
            dialog.exec_()

    # построение множественной регресии и корреляционный анализ
    def run_prophet_and_arima(self):
        # загружаем контракты с заданным диапазоном дат
        if not hasattr(self, "_current_filtered_df") or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа!")
            return
        else:
            from models_analyses.regression_analyses import (
                regression_analysis_month_by_month,
            )

            # в метод регрессионного анализа отправляем отфильтрованный _current_filtered_df
            regression_analysis_month_by_month(
                self._current_filtered_df
            )  # --------- contract_df подгрузить-------------

    # анализ контрактов без соответствующих лотов
    def run_contracts_less_dates(self):
        from models_analyses.contracts_without_lots import check_contracts_less_dates

        # метод поиска контрактов без лотов
        check_contracts_less_dates(self.contract_df) # необходимо убедиться в наличии данных

    def run_efficiency_analyses(self):
        from models_analyses.efficiency_analyses import main_method

        main_method(self.filtered_df, self.data_df)
        # Использует self.filtered_df и self.data_df, убедиться, что они заполнены

    def run_analyze_by_unit_price(self):
        """
        Запуск Ранжирование поставщиков по цене товара за единицу
        """
        if (
            self._current_filtered_df is not None
            and not self._current_filtered_df.empty
        ):
            self.start_analysis(
                analysis_task=self._analyze_by_unit_price_task,
                on_finished_callback=self.on_analysis_finished,  # Передаем ссылку на метод, а не результат его вызова
                create_plot=False,  # Или True, если нужно
                current_project_data=self._current_filtered_df,  # Передаем данные как kwargs
            )
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def handle_plot(self, title, figure):
        """Слот для отображения графика в главном потоке"""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        canvas = FigureCanvasQTAgg(figure)
        self.setCentralWidget(canvas)
        self.show()

    def _analyze_by_unit_price_task(self, update_progress, create_plot, **kwargs):
        # Получаем данные из kwargs, переданных через AnalysisThread
        current_project_data = kwargs.get("current_project_data")
        from models_analyses.efficiency_analyses import analyze_suppliers_by_unit_price

        update_progress.emit(10)
        # Убедиться, что analyze_suppliers_by_unit_price принимает df, update_progress и create_plot
        analyze_suppliers_by_unit_price(
            self, current_project_data, update_progress, create_plot
        ) # self.filtered_df
        update_progress.emit(100)

    def run_find_cross_discipline_lots(self):
        """
        Поиск поставщиков общих для разных дисциплин
        """
        if self._current_filtered_df is None or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
            return
        
        # Шаг 1. Находим общих поставщиков для дисциплин
        from models_analyses.analysis import find_common_suppliers_between_disciplines

        common_suppliers_df = find_common_suppliers_between_disciplines(
            self.filtered_df
        ) # при ошибке используем self._current_filtered_df
        if not common_suppliers_df.empty:
            from models_analyses.analysis import compare_materials_and_prices

            # Шаг 2. Сравниваем цены за единицу продукции
            comparison_results = compare_materials_and_prices(
                self.filtered_df, common_suppliers_df
            ) # при ошибке используем self._current_filtered_df

            if not comparison_results.empty:
                # Шаг 3. Визуализация
                from utils.vizualization_tools import (
                    visualize_price_differences,
                    heatmap_common_suppliers,
                )

                visualize_price_differences(comparison_results)
                heatmap_common_suppliers(common_suppliers_df)

                # Шаг 4. Статистика
                from models_analyses.analysis import matches_results_stat

                matches_results_stat(comparison_results)
                QMessageBox.information(
                    self,
                    "Кросс-дисциплинарный анализ",
                    "Кросс-дисциплинарный анализ завершен.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Кросс-дисциплинарный анализ",
                    "Нет данных для сравнения материалов и цен.",
                )
        else:
            QMessageBox.information(
                self,
                "Кросс-дисциплинарный анализ",
                "Общие поставщики между дисциплинами не найдены.",
            )
    def run_lotcount_peryear(self):
        # self.progress_bar.setValue(0)  # сброс прогресс-бара
        # self.analysis_thread = AnalysisThread(self._lotcount_peryear_task)
        # self.analysis_thread.update_progress.connect(
        #     self.progress_bar.setValue
        # )  # Обновляем прогресс-бар
        # self.analysis_thread.finished.connect(
        #     self.on_analysis_finished
        # )  # Уведомление о завершении
        # self.analysis_thread.start()
        """Запускает подсчет и визуализацию лотов по годам в отдельном потоке."""
        
        if self._current_filtered_df is None or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
            return
        
        # Исправленный вызов start_analysis
        self.start_analysis(
            analysis_task=self._lotcount_peryear_task,
            on_finished_callback=self.on_analysis_finished,
            create_plot=False,  # Или True, если нужно
            current_project_data=self._current_filtered_df,  # Передаем данные как kwargs
        )

    # def _lotcount_peryear_task(self):
    #     # метод посчета и визуализации количества лотов по дисциплинам по-квартально за год по проекту
    #     from widgets.analysis import lotcount_peryear
    #
    #     lotcount_peryear(self.filtered_df)
    #     self.analysis_thread.update_progress.emit(100)
    def _lotcount_peryear_task(self, update_progress, create_plot, **kwargs):
        # Получаем данные из kwargs
        current_project_data = kwargs.get("current_project_data")

        from widgets.analysis import lotcount_peryear

        update_progress.emit(10)
        lotcount_peryear(current_project_data)  # Используем данные из kwargs
        update_progress.emit(100)
    
    def run_herfind_hirshman_analysis(self):
        """ Запускает анализ Херфиндаля-Хиршмана в отдельном потоке """
        if self._current_filtered_df is None or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа HHI.")
            return
        
        print("Запуск анализа HHI...")
        print(f"Размер данных: {self._current_filtered_df.shape}")

        # Запускаем _herfind_hirshman_analysis_task через унифицированный метод start_analysis
        self.start_analysis(
            analysis_task=self._herfind_hirshman_analysis_task,
            on_finished_callback=self.on_analysis_finished,  # Общая функция завершения
            create_plot=False,  # Если HHI не строит графиков через этот флаг
            current_project_data=self._current_filtered_df,  # Передаем данные для анализа
        )
        
    def _herfind_hirshman_analysis_task(
            self, update_progress, create_plot=False, **kwargs):
        """Задача для расчета индекса Герфиндаля-Хиршмана (выполняется в отдельном потоке)."""
        update_progress.emit(10)
        current_project_data = kwargs.get("current_project_data")
        
        # создаем переменные для передачи в основной поток
        self._hhi_error_message = None
        self._hhi_success_message = None
        self._hhi_results = None
        
        if current_project_data is None or current_project_data.empty:
            self._hhi_error_message = "Данные для HHI анализа не предоставлены."
            update_progress.emit(100)
            return
        
        try:
            print(f"HHI анализ: Размер данных {current_project_data.shape}")
            print(f"Столбцы в данных: {list(current_project_data.columns)}")
            
            # Импортируем функции
            from widgets.analysisWidget import calculate_herfind_hirshman
            from utils.vizualization_tools import save_herfind_hirshman_results
            # from utils.functions import load_contracts
            
            update_progress.emit(30)
            
            # Выполняем расчет HHI
            returned_df, supplier_stats, hhi = calculate_herfind_hirshman(current_project_data)
            update_progress.emit(70)
            
            # Сохраняем результаты
            success = save_herfind_hirshman_results(supplier_stats, hhi)
            self._hhi_results = (returned_df, supplier_stats, hhi, success)
            
            if success:
                self._hhi_success_message = "Анализ HHI завершен успешно. Результаты сохранены в папку Analysis-Results."
            else:
                self._hhi_error_message = "Анализ выполнен, но возникли проблемы при сохранении результатов."
        
        except ImportError as e:
            error_msg = f"Ошибка импорта: {e}. Проверьте наличие модуля analysisWidget."
            print(f"Ошибка в _herfind_hirshman_analysis_task: {error_msg}")
            self._hhi_error_message = error_msg
        except Exception as e:
            error_msg = f"Произошла ошибка при анализе Герфиндаля-Хиршмана: {e}"
            print(f"Ошибка в _herfind_hirshman_analysis_task: {e}")
            self._hhi_error_message = error_msg
        finally:
            update_progress.emit(100)
            
    def on_hhi_analysis_finished(self):
        """Обработчик завершения анализа HHI - вызывается в основном потоке"""

        if hasattr(self, "_hhi_error_message") and self._hhi_error_message:
            QMessageBox.critical(self, "Ошибка HHI", self._hhi_error_message)
            self._hhi_error_message = None
        elif hasattr(self, "_hhi_success_message") and self._hhi_success_message:
            QMessageBox.information(
                self, "Анализ Герфиндаля-Хиршмана", self._hhi_success_message
            )
            self._hhi_success_message = None

        # Очищаем результаты
        if hasattr(self, "_hhi_results"):
            self._hhi_results = None

    # запуск унифицированного анализа альтернативных поставщиков

    # def run_alternative_suppliers_analysis(self):
    #     """
    #     Запускает анализ альтернативных поставщиков
    #     Использует _current_filtered_df как current_project_data
    #     и данные, полученные от Tab3Widget, как all_contracts_data.
    #     """
    #     if self.alternative_suppliers_analyzer.all_contracts_data is None:
    #         QMessageBox.warning(
    #             self,
    #             "Ошибка",
    #             "Полные данные контрактов не загружены. Пожалуйста, загрузите их через 'Данные по Контрактам'.",
    #         )
    #         return
    #
    #     if self._current_filtered_df is None or self._current_filtered_df.empty:
    #         QMessageBox.warning(
    #             self,
    #             "Ошибка",
    #             "Нет отфильтрованных данных (из лотов/контрактов) для текущего анализа. Пожалуйста, отфильтруйте данные на вкладках 'Параметры загруженных Лотов' или 'Параметры загруженных Контрактов'.",
    #         )
    #         return
    #
    #     print(f"Запуск анализа альтернативных поставщиков.")
    #
    #     # Запускаем _alternative_suppliers_analysis_task через унифицированный метод start_analysis
    #     # Передаем None в target_disciplines, чтобы Analyzer сам проанализировал все дисциплины
    #     self.start_analysis(
    #         analysis_task=self._alternative_suppliers_analysis_task,
    #         on_finished_callback=self._on_alternative_suppliers_analysis_finished,
    #         create_plot=False,  # Если этот анализ не строит графиков через этот флаг
    #         current_project_data=self._current_filtered_df,  # Отфильтрованные данные
    #         target_disciplines=None,  # Указываем анализатору обработать все дисциплины
    #     )
        
        # ============== изменения/дополнения - вызов записанного файла all_major_suppliers
        
    def run_alternative_suppliers_for_major_suppliers(self):
        """
        Читает файл all_major_suppliers.xlsx и запускает анализ альтернатив
        для каждого указанного там поставщика.
        """
        import pandas as pd
        if self.alternative_suppliers_analyzer.all_contracts_data is None:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Полные данные контрактов не загружены. Пожалуйста, загрузите их через 'Данные по Контрактам'.",
            )
            return
        try:
            file_path = r'D:\Analysis-Results\hirshman_results\all_major_suppliers.xlsx'
            major_suppliers_df = pd.read_excel(file_path)
        except FileNotFoundError:
            QMessageBox.warning(self,
                                "Ошибка",
                                "Файл 'all_major_suppliers.xlsx' не найден. Убедитесь, что он существует.",
                                )
            return
        
        if major_suppliers_df.empty:
            QMessageBox.information(
                self,
                "Информация",
                "Файл 'all_major_suppliers.xlsx' пуст. Не для кого искать альтернативы.",
            )
            return
        
        results_aggregator = {}
        
        for _, row in major_suppliers_df.iterrows():
            discipline = row.get('discipline')
            major_supplier = row.get('counterparty_name')
            
            if not discipline or not major_supplier:
                print("Предупреждение: Пропущена строка в файле")
                continue
            
            print(
                f"Запуск анализа альтернатив для дисциплины: '{discipline}', поставщика: '{major_supplier}'"
            )
            self.start_analysis(
                analysis_task=self._alternative_suppliers_analysis_task,
                on_finished_callback=self._on_alternative_suppliers_analysis_finished,
                create_plot = False,
                current_project_data = self._current_filtered_df,
                target_disciplines = [discipline],
                target_supplier = major_supplier  # Новый параметр!
            )
        
        # ============== конец изменений/дополнений
    
    def _alternative_suppliers_analysis_task(
     self, update_progress, create_plot=False, **kwargs
 ):
     """Задача для анализа альтернативных поставщиков (выполняется в отдельном потоке)."""
     update_progress.emit(10)
 
     current_project_data = kwargs.get("current_project_data")
     target_disciplines = kwargs.get("target_disciplines")
     target_supplier = kwargs.get("target_supplier")
 
     # Вызов метода анализатора
     results = self.alternative_suppliers_analyzer.run_analysis(
         current_project_data,
         target_disciplines=target_disciplines,
         target_supplier=target_supplier
     )
     update_progress.emit(80)
 
     # Здесь теперь обрабатываются результаты для отображения в QMessageBox
     # и для экспорта.
 
     # --- Часть для отображения в QMessageBox (адаптируем, чтобы показывать по дисциплинам) ---
     if results:
         full_report_text = "<h3>Результаты анализа альтернативных поставщиков:</h3>"
         for disc, products_data in results.items():
             full_report_text += f"<h4>Дисциплина: {disc}</h4>"
             if products_data:
                 for product, info in products_data.items():
                     full_report_text += f"<h5>Продукт: {product}</h5>"
                     full_report_text += f"Текущие поставщики: {', '.join(info['current_suppliers'])}<br>"
                     full_report_text += (
                         f"Найдено альтернатив: {info['alternatives_found']}<br>"
                     )
                     full_report_text += f"Рекомендация: {info['recommendation']}<br>"
                     full_report_text += "<b>Топ-альтернативы:</b><br>"
                     if info["alternative_suppliers"]:
                         # Показываем топ-3 для QMessageBox
                         top_alternatives_for_display = sorted(
                             info["alternative_suppliers"],
                             key=lambda x: x["recommendation_score"],
                             reverse=True,
                         )[:3]
                         for alt in top_alternatives_for_display:
                             full_report_text += f"- {alt['supplier_name']} (Рейтинг: {alt['recommendation_score']:.2f}, Ср. цена: {alt['avg_price']:.2f})<br>"
                     else:
                         full_report_text += "  Нет доступных альтернатив.<br>"
                     full_report_text += "<br>"
             else:
                 full_report_text += "  Нет данных по продуктам для этой дисциплины.<br>"
             full_report_text += "<hr>"  # Разделитель между дисциплинами
         QMessageBox.information(self, "Результаты Анализа", full_report_text)
     else:
         QMessageBox.information(
             self,
             "Результаты Анализа",
             "Анализ не дал результатов или не удалось найти альтернативных поставщиков.",
         )
 
     # --- Часть для экспорта в Excel ---
     if results:
         # Запрашиваем у пользователя путь для сохранения файла
         options = QFileDialog.Options()
         file_name, _ = QFileDialog.getSaveFileName(
             self,
             "Сохранить результаты анализа альтернативных поставщиков",
             "alternative_suppliers_analysis.xlsx",
             "Excel Files (*.xlsx);;All Files (*)",
             options=options,
         )
         if file_name:
             export_success = export_alternative_suppliers_to_excel(results, file_name)
             if export_success:
                 QMessageBox.information(
                     self,
                     "Экспорт завершен",
                     "Данные успешно экспортированы в Excel.",
                 )
             else:
                 QMessageBox.warning(
                     self,
                     "Ошибка экспорта",
                     "Произошла ошибка при экспорте данных в Excel.",
                 )
     else:
         print(
             "Нет результатов для экспорта."
         )  # Уже было сообщение об отсутствии результатов анализа
 
     update_progress.emit(100)
    
    
    def _on_alternative_suppliers_analysis_finished(self):
        # эта функция завершения анализа альтернативных поставщиков
        QMessageBox.information(
            self, "Завершено", "Анализ альтернативных поставщиков завершен!"
        )

    # Расчет сумм остатков по складам по валютам поставок
    def run_warehouseStatistics(self):
        print("Входим в метод Статистики по Складам")
        from utils.analyzeWarehouseStatistics import calculate_statistics

        calculate_statistics(self.filtered_df)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Устанавливаем шрифты и тему
    # set_dark_theme(app)
    set_fonts(app)
    # Загружаем стили из CSS-файла
    stylesheet = load_stylesheet("styles_black.qss")
    app.setStyleSheet(stylesheet)
    # Установим шрифт для отображения подсказок
    QToolTip.setFont(QFont("SansSerif", 10))
    # Установим стиль приложения
    app.setStyle("Fusion")
    set_light_theme(app)

    app.setAttribute(
        Qt.AA_EnableHighDpiScaling, True
    )  # Включить поддержку высокого разрешения
    app.setAttribute(
        Qt.AA_UseHighDpiPixmaps, True
    )  # Включить использование DPI картинок
    app.setAttribute(
        Qt.AA_DisableWindowContextHelpButton, False
    )  # Активировать tooltips

    window = Window()
    window.show()
    sys.exit(app.exec_())

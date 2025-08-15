import matplotlib

matplotlib.use("Agg")  # Устанавливаем безопасный бэкенд

import json
import sys
import os
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import Qt, QPoint
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
    AlternativeSuppliersAnalyzer,
    export_alternative_suppliers_to_excel,
)

from selection_dialog import SelectionDialog
from styles import set_light_theme, set_fonts, load_stylesheet
from utils.clean_datas import clean_database
from utils.data_model import DataModel
from utils.functions import CurrencyConverter
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


class MyTabWidget(QWidget):
    def __init__(self):
        super().__init__()
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
        tab1_widget = self.tab_widget.notebook.widget(
            0
        )  # Получаем первый виджет вкладки
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
        tab4_widget = self.tab_widget.notebook.widget(3)
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

    # УДАЛЕНЫ ВСЕ МЕТОДЫ ДЛЯ РАБОТЫ С ПОТОКАМИ:
    # - start_analysis_thread
    # - on_thread_finished
    # - closeEvent с остановкой потоков
    # - AnalysisThread класс полностью убран

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
        analysisMenu.addAction(self.suppliersfriquencyAction)
        analysisMenu.addAction(self.networkanalyseAction)
        analysisMenu.addAction(self.analyzeKPIAction)
        analysisMenu.addAction(self.efficiency_analyses_action)
        analysisMenu.addAction(self.suppliers_by_unit_price_action)
        analysisMenu.addAction(self.find_cross_discipline_lotsAction)
        analysisMenu.addAction(self.lotcount_peryearAction)

        # Меню Анализ по Контрактам
        analysisMenuContract = menuBar.addMenu("Анализ данных по Контрактам")
        analysisMenuContract.addAction(self.analyzeClasterAction)
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

        # Действия для меню Анализ данных по Контрактам
        self.analyzeClasterAction = QAction("Кластерный анализ", self)
        self.setActionTooltip(
            self.analyzeClasterAction,
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

        # Действия для меню Анализ данных по Складам
        self.warehouseStatistics = QAction(
            "Расчет остатков сум на складах по валютам", self
        )

    def _connectActions(self):
        # Подключение сигналов к действиям
        self.ContrAction.triggered.connect(self.load_sql_data)
        self.ExitAction.triggered.connect(self.close)

        # Подключение сигналов к методам Анализа данных по Лотам
        self.analyzeMonthlyExpensesAction.triggered.connect(
            self.run_analyze_monthly_cost
        )
        self.analyzeTopSuppliersAction.triggered.connect(self.run_analyze_top_suppliers)
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
        self.analyzeClasterAction.triggered.connect(self.run_ClusterAnalyze)
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

    # УБРАН СЛОЖНЫЙ start_analysis МЕТОД - ВСЕ АНАЛИЗЫ ТЕПЕРЬ СИНХРОННЫЕ
    def show_progress(self, value):
        """Простое обновление прогресс-бара для синхронных операций"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Обновляем интерфейс

    def hide_progress(self):
        """Скрытие прогресс-бара после завершения операции"""
        self.progress_bar.hide()

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
            self.progress_bar.show()
            self.show_progress(10)

            print("Данные для анализа (месячные затраты):")
            # Используем минимальную и максимальную даты из отфильтрованных данных
            start_date = self._current_filtered_df["close_date"].min()
            end_date = self._current_filtered_df["close_date"].max()

            self.show_progress(30)
            from models_analyses.analysis import analyze_monthly_cost

            analyze_monthly_cost(self, self._current_filtered_df, start_date, end_date)
            self.show_progress(100)
            self.hide_progress()
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def run_analyze_top_suppliers(self):
        #  метод анализа поставщиков с высокими и низкими ценами за единицу товара
        if self._current_filtered_df is not None:
            self.progress_bar.show()
            self.show_progress(10)

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

            self.show_progress(30)
            from models_analyses.analysis import analyze_top_suppliers

            # здесь логика для анализа данных
            analyze_top_suppliers(
                self, self._current_filtered_df, start_date, end_date, project_name
            )
            self.show_progress(100)
            self.hide_progress()
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def run_ClusterAnalyze(self):
        # Метод для классификации поставщиков с обучением методом SeedKMeans
        from models_analyses.clusterAnalysis_suppliers import run_supplier_clustering
        
        if self._current_filtered_df is not None:
            self.progress_bar.show()
            self.show_progress(10)

            output_dir = r"D:\Analysis-Results\suppliers_cluster_analysis"
            os.makedirs(output_dir, exist_ok=True)

            self.show_progress(30)
            
            # Конвертируем цены за единицу и суммы контрактов в единую валюту EUR
            converter = CurrencyConverter()
            columns_info = [
                ("total_contract_amount", "contract_currency", "total_contract_amount_eur"),
                ("unit_price", "contract_currency", "unit_price_eur"),
            ]
            contracts_data = converter.convert_multiple_columns(self._current_filtered_df, columns_info)
            
            supplier_clusters, analyzer = run_supplier_clustering(contracts_data)


    def run_analyze_supplier_friquency(self):
        # Метод для анализа частоты выбора поставщиков
        if self._current_filtered_df is not None:
            self.progress_bar.show()
            self.show_progress(10)

            from models_analyses.analysis import analyze_supplier_frequency

            analyze_supplier_frequency(self._current_filtered_df)
            self.show_progress(100)
            self.hide_progress()

            QMessageBox.information(
                self, "Завершено", "Анализ частоты поставщиков завершен!"
            )
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def run_network_analysis(self):
        # Метод для сетевого анализа
        if self._current_filtered_df is not None:
            self.progress_bar.show()
            self.show_progress(10)

            print("Запуск сетевого анализа")
            from models_analyses.analysis import network_analysis

            network_analysis(self, self._current_filtered_df)
            self.show_progress(100)
            self.hide_progress()

            QMessageBox.information(self, "Завершено", "Сетевой анализ завершен!")
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def load_sql_data(self):
        print("Загрузка данных в базу SQL...")
        DataModel(self).open_file_dialog()

    # Логика работы с Анализом данных по Контрактам

    def run_analyzeNonEquilSums(self):
        # Метод для поиска несоответствий в суммах Лотов и Контрактов
        self.progress_bar.show()
        self.show_progress(10)

        print("Запуск поиска несоответствий сумм")
        from models_analyses.analyze_contracts import analyzeNonEquilSums

        analyzeNonEquilSums(self, self._current_filtered_df)
        self.show_progress(100)
        self.hide_progress()

        QMessageBox.information(self, "Завершено", "Анализ несоответствий завершен!")

    def run_trend_analyses(self):
        if not hasattr(self, "_current_filtered_df") or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа!")
            return
        else:
            self.progress_bar.show()
            self.show_progress(10)

            from models_analyses.analyze_contracts import (
                data_preprocessing_and_analysis,
            )

            # Удалим в датафрейме self._current_filtered_df строки-дубликаты если они есть
            self._current_filtered_df = self._current_filtered_df.drop_duplicates()
            self.show_progress(30)

            df_merged = data_preprocessing_and_analysis(self._current_filtered_df)
            self.show_progress(70)

            dialog = SelectionDialog(df_merged=df_merged, parent=self)
            self.show_progress(100)
            self.hide_progress()

            dialog.exec_()

    # построение множественной регресии и корреляционный анализ
    def run_prophet_and_arima(self):
        # загружаем контракты с заданным диапазоном дат
        if not hasattr(self, "_current_filtered_df") or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа!")
            return
        else:
            self.progress_bar.show()
            self.show_progress(10)

            from models_analyses.regression_analyses import (
                regression_analysis_month_by_month,
            )

            # в метод регрессионного анализа отправляем отфильтрованный _current_filtered_df
            regression_analysis_month_by_month(self._current_filtered_df)
            self.show_progress(100)
            self.hide_progress()

    # анализ контрактов без соответствующих лотов
    def run_contracts_less_dates(self):
        self.progress_bar.show()
        self.show_progress(10)

        from models_analyses.contracts_without_lots import check_contracts_less_dates

        # метод поиска контрактов без лотов
        check_contracts_less_dates(
            self.contract_df
        )  # необходимо убедиться в наличии данных
        self.show_progress(100)
        self.hide_progress()

    def run_efficiency_analyses(self):
        self.progress_bar.show()
        self.show_progress(10)

        from models_analyses.efficiency_analyses import main_method

        main_method(self.filtered_df, self.data_df)
        # Использует self.filtered_df и self.data_df, убедиться, что они заполнены
        self.show_progress(100)
        self.hide_progress()

    def run_analyze_by_unit_price(self):
        """
        Запуск Ранжирование поставщиков по цене товара за единицу
        """
        if (
            self._current_filtered_df is not None
            and not self._current_filtered_df.empty
        ):
            self.progress_bar.show()
            self.show_progress(10)

            from models_analyses.efficiency_analyses import (
                analyze_suppliers_by_unit_price,
            )

            analyze_suppliers_by_unit_price(
                self, self._current_filtered_df, lambda x: self.show_progress(x), False
            )
            self.show_progress(100)
            self.hide_progress()

            QMessageBox.information(
                self, "Завершено", "Анализ по цене за единицу завершен!"
            )
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")

    def run_find_cross_discipline_lots(self):
        """
        Поиск поставщиков общих для разных дисциплин
        """
        if self._current_filtered_df is None or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
            return

        self.progress_bar.show()
        self.show_progress(10)

        # Шаг 1. Находим общих поставщиков для дисциплин
        from models_analyses.analysis import find_common_suppliers_between_disciplines

        common_suppliers_df = find_common_suppliers_between_disciplines(
            self.filtered_df
        )  # при ошибке используем self._current_filtered_df
        self.show_progress(40)

        if not common_suppliers_df.empty:
            from models_analyses.analysis import compare_materials_and_prices

            # Шаг 2. Сравниваем цены за единицу продукции
            comparison_results = compare_materials_and_prices(
                self.filtered_df, common_suppliers_df
            )  # при ошибке используем self._current_filtered_df
            self.show_progress(70)

            if not comparison_results.empty:
                # Шаг 3. Визуализация
                from utils.vizualization_tools import (
                    visualize_price_differences,
                    heatmap_common_suppliers,
                )

                visualize_price_differences(comparison_results)
                heatmap_common_suppliers(common_suppliers_df)
                self.show_progress(90)

                # Шаг 4. Статистика
                from models_analyses.analysis import matches_results_stat

                matches_results_stat(comparison_results)
                self.show_progress(100)
                self.hide_progress()

                QMessageBox.information(
                    self,
                    "Кросс-дисциплинарный анализ",
                    "Кросс-дисциплинарный анализ завершен.",
                )
            else:
                self.hide_progress()
                QMessageBox.information(
                    self,
                    "Кросс-дисциплинарный анализ",
                    "Нет данных для сравнения материалов и цен.",
                )
        else:
            self.hide_progress()
            QMessageBox.information(
                self,
                "Кросс-дисциплинарный анализ",
                "Общие поставщики между дисциплинами не найдены.",
            )

    def run_lotcount_peryear(self):
        """Запускает подсчет и визуализацию лотов по годам."""

        if self._current_filtered_df is None or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа.")
            return

        self.progress_bar.show()
        self.show_progress(10)

        from widgets.analysis import lotcount_peryear

        lotcount_peryear(self._current_filtered_df)
        self.show_progress(100)
        self.hide_progress()

        QMessageBox.information(self, "Завершено", "Анализ количества лотов завершен!")

    def run_herfind_hirshman_analysis(self):
        """Запускает анализ Херфиндаля-Хиршмана"""
        if self._current_filtered_df is None or self._current_filtered_df.empty:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа HHI.")
            return

        self.progress_bar.show()
        self.show_progress(10)

        print("Запуск анализа HHI...")
        print(f"Размер данных: {self._current_filtered_df.shape}")

        try:
            print(f"HHI анализ: Размер данных {self._current_filtered_df.shape}")
            print(f"Столбцы в данных: {list(self._current_filtered_df.columns)}")

            # Импортируем функции
            from widgets.analysisWidget import calculate_herfind_hirshman
            from utils.vizualization_tools import save_herfind_hirshman_results

            self.show_progress(30)

            # Выполняем расчет HHI
            returned_df, supplier_stats, hhi = calculate_herfind_hirshman(
                self._current_filtered_df
            )
            self.show_progress(70)

            # Сохраняем результаты
            success = save_herfind_hirshman_results(supplier_stats, hhi)
            self.show_progress(100)

            if success:
                self.hide_progress()
                QMessageBox.information(
                    self,
                    "Анализ Герфиндаля-Хиршмана",
                    "Анализ HHI завершен успешно. Результаты сохранены в папку Analysis-Results.",
                )
            else:
                self.hide_progress()
                QMessageBox.warning(
                    self,
                    "Предупреждение",
                    "Анализ выполнен, но возникли проблемы при сохранении результатов.",
                )

        except ImportError as e:
            error_msg = f"Ошибка импорта: {e}. Проверьте наличие модуля analysisWidget."
            print(f"Ошибка в run_herfind_hirshman_analysis: {error_msg}")
            self.hide_progress()
            QMessageBox.critical(self, "Ошибка HHI", error_msg)
        except Exception as e:
            error_msg = f"Произошла ошибка при анализе Герфиндаля-Хиршмана: {e}"
            print(f"Ошибка в run_herfind_hirshman_analysis: {e}")
            self.hide_progress()
            QMessageBox.critical(self, "Ошибка HHI", error_msg)

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
            file_path = r"D:\Analysis-Results\hirshman_results\all_major_suppliers.xlsx"
            major_suppliers_df = pd.read_excel(file_path)
        except FileNotFoundError:
            QMessageBox.warning(
                self,
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

        self.progress_bar.show()
        self.show_progress(10)

        results_aggregator = {}
        total_suppliers = len(major_suppliers_df)

        for idx, row in major_suppliers_df.iterrows():
            discipline = row.get("discipline")
            major_supplier = row.get("counterparty_name")

            if not discipline or not major_supplier:
                print("Предупреждение: Пропущена строка в файле")
                continue

            # Обновляем прогресс
            progress = 10 + int((idx / total_suppliers) * 80)
            self.show_progress(progress)

            print(
                f"Запуск анализа альтернатив для дисциплины: '{discipline}', поставщика: '{major_supplier}'"
            )

            # Запускаем анализ синхронно
            converter = CurrencyConverter()
            # Конвертируем и сохраняем нужный столбец
            columns_info = [
                ("total_contract_amount", "contract_currency", "total_contract_amount_eur"),
                ("unit_price", "contract_currency", "unit_price_eur"),
            ]
            current_project_data = converter.convert_multiple_columns(self._current_filtered_df, columns_info)

            # запускаем непосредственно анализ
            results = self.alternative_suppliers_analyzer.run_analysis(
                current_project_data,
                target_disciplines=[discipline],
                target_supplier=major_supplier,
            )

            if results:
                results_aggregator.update(results)

        self.show_progress(90)

        # Обрабатываем результаты для отображения в QMessageBox
        if results_aggregator:
            # Подсчитываем статистику
            total_disciplines = len(results_aggregator)
            total_products = sum(len(products_data) for products_data in results_aggregator.values() if products_data)
            total_alternatives = sum(
                info.get("alternatives_found", 0)
                for products_data in results_aggregator.values()
                if products_data
                for info in products_data.values()
            )
            products_with_alternatives = sum(
                1 for products_data in results_aggregator.values()
                if products_data
                for info in products_data.values()
                if info.get("alternatives_found", 0) > 0
            )
            
            full_report_text = f"""Анализ альтернативных поставщиков завершен!

            Результаты:
            • Дисциплин: {total_disciplines}
            • Продуктов: {total_products}
            • Найдено альтернатив: {total_alternatives}
            • Продуктов с альтернативами: {products_with_alternatives}
    
            Подробные результаты сохранены в Excel файл."""
        
        else:
            full_report_text = "Анализ завершен, но результаты не получены."
            
            
        # Экспорт в Excel
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты анализа альтернативных поставщиков",
            "alternative_suppliers_analysis.xlsx",
            "Excel Files (*.xlsx);;All Files (*)",
            options=options,
        )
        if file_name:
            export_success = export_alternative_suppliers_to_excel(
                results_aggregator, file_name
            )
        
            if export_success:
                full_report_text += "\n\n✅ Данные успешно экспортированы в Excel."
            else:
                full_report_text += "\n\n❌ Ошибка при экспорте в Excel."
        
        self.show_progress(100)
        self.hide_progress()
        
        QMessageBox.information(self, "Результаты анализа", full_report_text)

    # Расчет сумм остатков по складам по валютам поставок
    def run_warehouseStatistics(self):
        self.progress_bar.show()
        self.show_progress(10)

        print("Входим в метод Статистики по Складам")
        from utils.analyzeWarehouseStatistics import calculate_statistics

        calculate_statistics(self.filtered_df)
        self.show_progress(100)
        self.hide_progress()


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Устанавливаем шрифты и тему
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

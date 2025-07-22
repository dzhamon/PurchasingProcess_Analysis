"""
	Модуль загрузки и обработки Лотов
"""
from PyQt5.QtCore import pyqtSignal, QDate
from PyQt5.QtWidgets import (QWidget, QGridLayout, QDateEdit, QLabel, QPushButton,
                             QTableView, QFrame, QMessageBox)
import pandas as pd
from utils.config import SQL_PATH
import sqlite3
from utils.date_range_selector_new import DateRangeSelector  # Импортируем класс из отдельного модуля

from utils.functions import cleanDataDF
from utils.PandasModel_previous import PandasModel


class Tab1Widget(QWidget):
	# Определяем сигнал, который будет испускаться при изменении фильтрованных данных
	filtered_data_changed = pyqtSignal(pd.DataFrame)
	
	def __init__(self):
		super().__init__()
		self.data_df = pd.DataFrame()
		self.init_ui()  # инициализация пользовательского интерфейса
	
	def init_ui(self):
		# создаем макет и виджеты
		self.layout = QGridLayout(self)
		
		# Выбор диапазона дат
		self.start_date_edit = QDateEdit(self)
		self.start_date_edit.setCalendarPopup(True)
		self.start_date_edit.setDate(QDate.currentDate().addMonths(-1))
		
		self.end_date_edit = QDateEdit(self)
		self.end_date_edit.setCalendarPopup(True)
		self.end_date_edit.setDate(QDate.currentDate())
		
		self.layout.addWidget(QLabel("Начальная дата:"), 0, 0)
		self.layout.addWidget(self.start_date_edit, 0, 1)
		self.layout.addWidget(QLabel("Конечная дата:"), 1, 0)
		self.layout.addWidget(self.end_date_edit, 1, 1)
		
		# Кнопка применения диапазона дат
		self.apply_button = QPushButton("Загрузить Лоты из выбранного диапазона дат")
		self.apply_button.clicked.connect(self.apply_date_filter)
		self.layout.addWidget(self.apply_button, 2, 0, 1, 2)
		
		# Создаем QTableView для отображения данных
		self.table_widget = QTableView()
		self.layout.addWidget(self.table_widget, 3, 0, 1, 3)
		
		# Второй фрейм с информацией о данных
		frame2 = QFrame()
		frame2.setFrameShape(QFrame.Box)
		self.layout.addWidget(frame2, 4, 0, 1, 3)
	
	def apply_date_filter(self):
		# Получаем выбранные даты
		start_date = self.start_date_edit.date().toPyDate()
		end_date = self.end_date_edit.date().toPyDate()
		
		# # Преобразуем даты в datetime64[ns]
		# start_date = pd.to_datetime(start_date)
		# end_date = pd.to_datetime(end_date)
		
		# Проверяем корректность диапазона дат
		if start_date > end_date:
			QMessageBox.warning(self, "Предупреждение",
			                    "Начальная дата позже конечной даты. Проверьте корректность значений")
			return
		
		# Загружаем данные из warehouses_remnants по выбранным датам
		db_path = SQL_PATH
		conn = sqlite3.connect(db_path)
		cursor = conn.cursor()
		
		# Проверяем, есть ли в таблице data_kp записи с такими датами
		cursor.execute("""
		       SELECT COUNT(*) FROM data_kp
		       WHERE DATE(close_date) BETWEEN DATE(?) AND DATE(?);
		   """, (start_date, end_date))
		
		count = cursor.fetchone()[0]
		
		if count == 0:
			QMessageBox.warning(self, "Ошибка",
			                    "В выбранном диапазоне нет данных. Измените даты.")
			conn.close()
			# Сброс дат на значения по умолчанию
			self.start_date_edit.setDate(QDate.currentDate().addMonths(-1))
			self.end_date_edit.setDate(QDate.currentDate())
			return
		# Если данные есть - загружаем их
		query = f"""
		SELECT * FROM data_kp
		WHERE DATE(close_date) BETWEEN DATE(?) AND DATE(?);
			"""
		# Загрузка данных в датафрейм
		self.data_df = pd.read_sql_query(query, conn, params=(start_date, end_date))
		# закрыть соединение с базой данных
		conn.close()
		
		self.data_df = cleanDataDF(self.data_df)  # очистка данных полученного df
		
		# Сигнал и отображение
		self.filtered_data_changed.emit(self.data_df)
		
		self.display_data(self.data_df)
	
	def display_data(self, data_df):
		if data_df.empty:
			QMessageBox.warning(self, "Ошибка", "DataFrame пустой, нечего отображать")
			return
		try:
			model = PandasModel(data_df)
			self.table_widget.setModel(model)
			self.table_widget.setSortingEnabled(True)
		except Exception as e:
			QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить данные: {str(e)}")

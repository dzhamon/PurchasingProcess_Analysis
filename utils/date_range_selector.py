from PyQt5.QtWidgets import QWidget, QGridLayout, QDateEdit, QLabel, QMessageBox
from PyQt5.QtCore import QDate
import pandas as pd


class DateRangeSelector(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.init_ui()
	
	def init_ui(self):
		self.layout = QGridLayout(self)
		
		# Метки и виджеты для выбора дат
		self.start_date_label = QLabel("Начальная дата:")
		self.start_date_edit = QDateEdit(self)
		self.start_date_edit.setCalendarPopup(True)
		self.start_date_edit.setDate(QDate.currentDate().addMonths(-1))
		
		self.end_date_label = QLabel("Конечная дата:")
		self.end_date_edit = QDateEdit(self)
		self.end_date_edit.setCalendarPopup(True)
		self.end_date_edit.setDate(QDate.currentDate())
		
		# Добавляем виджеты на макет
		self.layout.addWidget(self.start_date_label, 0, 0)
		self.layout.addWidget(self.start_date_edit, 0, 1)
		self.layout.addWidget(self.end_date_label, 1, 0)
		self.layout.addWidget(self.end_date_edit, 1, 1)
	
	def get_date_range(self):
		start_date = self.start_date_edit.date().toPyDate()
		end_date = self.end_date_edit.date().toPyDate()
		
		if start_date > end_date:
			QMessageBox.warning(self, "Предупреждение",
			                    "Начальная дата позже конечной даты. Проверьте корректность значений")
			return None, None
		
		# Приводим к datetime64[ns]
		return pd.to_datetime(start_date), pd.to_datetime(end_date)

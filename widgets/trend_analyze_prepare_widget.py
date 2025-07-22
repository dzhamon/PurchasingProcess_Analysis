from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QWidget, QListWidget, QMessageBox, QLineEdit,
                             QVBoxLayout, QPushButton, QTableView, QDialog, QAbstractItemView)
from PyQt5.QtCore import pyqtSignal, QTimer
from utils.PandasModel_previous import PandasModel
import pandas as pd


class TrendAnalyzeWidget(QWidget):
	data_ready_for_analysis = pyqtSignal(pd.DataFrame)
	
	def on_filtered_contracts_received(self, filtered_df):
		""" Слот для обновления данных на основе сигнала от Tab3 """
		print("Получены фильтрованные данные из Tab3")
		print("on_filtered_contracts_received вызван")
		if not isinstance(filtered_df, pd.DataFrame):
			print("Ошибка: Данные не являются DataFrame")
			return
		print(f"Получен DataFrame с {len(filtered_df)} строками и колонками: {filtered_df.columns.tolist()}")
		self.update_data(filtered_df)
	
	def __init__(self, params_list, parent=None):
		super().__init__(parent)
		self.params = params_list
		self.setup_ui()
	
	# self.receive_data_slot = receive_data_slot  # Параметр для слота получения данных
	
	def setup_ui(self):
		# Создание основного макета
		self.layout = QVBoxLayout(self)
		# Словарь для хранения QListWidget и их значений
		self.list_widgets = {}
		self.filtered_df = pd.DataFrame()  # Пустой DataFrame для хранения отфильтрованных данных
		
		# Создание макета для списков (горизонтальный макет)
		list_layout = QHBoxLayout()
		# Инициализация списков и кнопок "Очистить"
		self.init_list_widgets(list_layout)
		
		# Создание кнопки для выполнения запроса
		self.execute_button = QPushButton('Выполнить запрос')
		self.execute_button.clicked.connect(self.show_filtered_data)
		
		# Добавление макета списков в основной макет
		self.layout.addLayout(list_layout)
		self.layout.addWidget(self.execute_button)
		self.setLayout(self.layout)  # Устанавливаем основной макет
	
	def filter_listbox(self, text, list_widget):
		"""
		Фильтрует элементы в списке на основе текста, сохраняя полный список.
		"""
		# Получаем полный список элементов
		if not hasattr(list_widget, 'full_list'):
			list_widget.full_list = [list_widget.item(row).text() for row in range(list_widget.count())]
		
		# Фильтруем элементы
		filtered_items = [item for item in list_widget.full_list if text.lower() in item.lower()]
		
		# Очищаем текущий виджет
		list_widget.clear()
		list_widget.addItems(filtered_items)
	
	def update_data(self, filtered_df):
		"""
		Обновляет данные в QListWidget и включает их для выбора.
		"""
		print("Данные получены: ", len(filtered_df))
		
		if filtered_df.empty:
			print("Получен пустой DataFrame.")
			self.clear_list_widgets()
			return
		
		self.filtered_df = filtered_df  # Сохраняем отфильтрованный DataFrame
		
		# Очищаем текущий layout
		if hasattr(self, 'list_layout'):
			self.clear_layout(self.list_layout)
		
		# Создаем новый макет для списков
		self.list_layout = QHBoxLayout()
		self.init_list_widgets(self.list_layout)  # Передаем управление на инициализацию виджетов
		
		# Обновляем основной layout
		self.layout.insertLayout(0, self.list_layout)
		
		# Заполняем QListWidget уникальными значениями из DataFrame
		for col, (list_widget, search_entry) in self.list_widgets.items():
			list_widget.clear()
			unique_values = sorted(filtered_df[col].dropna().unique())
			list_widget.addItems([str(value) for value in unique_values])
	
	# # Сбрасываем full_list для обновленных данных
	# list_widget.full_list = [str(value) for value in unique_values]
	
	def clear_layout(self, layout):
		while layout.count():
			child = layout.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
	
	def send_data_to_analysis(self):
		"""
		Передает отфильтрованные данные в модуль анализа.
		"""
		if hasattr(self, 'filtered_df_to_analyze') and not self.filtered_df_to_analyze.empty:
			self.data_ready_for_analysis.emit(self.filtered_df_to_analyze)
			QMessageBox.information(self, "Успех", "Данные успешно переданы на анализ")
		else:
			QMessageBox.warning(self, "Ошибка", "Нет данных для анализа")
	
	def init_list_widgets(self, layout):
		"""
		Инициализирует QListWidget и кнопки "Очистить", добавляет их в горизонтальный макет.
		"""
		self.list_widgets = {}
		
		for name in self.params:
			widget_layout = QVBoxLayout()  # Используем вертикальный макет для метки, списка и кнопки "Очистить"
			label = QLabel(f"Выберите {name}:")
			search_entry = QLineEdit(self)
			search_entry.setPlaceholderText(f"Поиск по {name}...")
			search_entry.show()
			
			list_widget = QListWidget()
			list_widget.setObjectName(name)
			list_widget.setSelectionMode(QListWidget.MultiSelection)  # Поддержка множественного выбора
			list_widget.show()
			self.list_widgets[name] = (list_widget, search_entry)
			
			# Привязываем поиск к строке поиска
			search_entry.textChanged.connect(lambda text, lw=list_widget: self.filter_listbox(text, lw))
			
			# Кнопка "Очистить"
			clear_button = QPushButton('Очистить')
			clear_button.clicked.connect(lambda _, lw=list_widget: self.clear_selection(lw))
			
			# Добавление виджетов в макет
			widget_layout.addWidget(label)
			widget_layout.addWidget(search_entry)
			widget_layout.addWidget(list_widget)
			widget_layout.addWidget(clear_button)
			layout.addLayout(widget_layout)
			
			# Сохраняем ссылки на виджеты
			self.list_widgets[name] = (list_widget, search_entry)
	
	def clear_list_widgets(self):
		"""
		Очищает все QListWidget.
		"""
		for list_widget in self.list_widgets.values():
			list_widget.clear()
	
	def clear_selection(self, list_widget):
		"""
		Снимает выделение со всех элементов в указанном QListWidget.
		"""
		list_widget.clearSelection()
		
		# Восстанавливаем полный список элементов
		if hasattr(list_widget, 'full_list'):
			list_widget.clear()
			list_widget.addItems(list_widget.full_list)
	
	def show_filtered_data(self):
		"""
		Отображает отфильтрованные данные во всплывающем окне.
		"""
		if self.filtered_df.empty:
			QMessageBox.warning(self, "Ошибка", "Нет данных для отображения.")
			return
		
		# Применяем фильтры на основе выбранных элементов в QListWidget
		filtered_df = self.filtered_df.copy()
		for col, (list_widget, _) in self.list_widgets.items():
			selected_items = [item.text() for item in list_widget.selectedItems()]
			if selected_items:
				filtered_df = filtered_df[filtered_df[col].isin(selected_items)]
		
		if filtered_df.empty:
			QMessageBox.warning(self, "Ошибка", "Нет данных, соответствующих выбранным критериям.")
			return
		
		# Сохраняем отфильтрованные данные в атрибут класса для передачи в analysis.py
		self.filtered_df_to_analyze = filtered_df
		
		# Создание всплывающего окна для отображения отфильтрованных данных
		dialog = QDialog(self)
		dialog.setWindowTitle("Отфильтрованные данные")
		layout = QVBoxLayout(dialog)
		
		# Создаем виджет таблицы для отображения данных
		table_view = QTableView()
		model = PandasModel(filtered_df)
		table_view.setModel(model)
		table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
		layout.addWidget(table_view)
		
		# Создание кнопки "Analyze Data"
		analyze_button = QPushButton('Передать данные на анализ')
		analyze_button.clicked.connect(self.send_data_to_analysis)
		layout.addWidget(analyze_button)
		
		# Создание кнопки "Развернуть"
		expand_button = QPushButton('Развернуть')
		expand_button.setCheckable(True)  # Кнопка будет переключаемой
		expand_button.toggled.connect(lambda checked: self.toggle_fullscreen(dialog, checked))
		layout.addWidget(expand_button)
		
		dialog.setLayout(layout)
		dialog.resize(800, 600)
		dialog.exec_()
	
	def toggle_fullscreen(self, dialog, checked):
		"""
		Разворачивает или возвращает окно к предыдущему размеру в зависимости от состояния кнопки.
		"""
		if checked:
			dialog.showMaximized()
		else:
			dialog.showNormal()
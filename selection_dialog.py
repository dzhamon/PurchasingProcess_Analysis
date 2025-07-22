from PyQt5.QtWidgets import (
	QApplication, QWidget, QVBoxLayout, QPushButton, QBoxLayout,
	QDialog, QTableWidget, QTableWidgetItem, QComboBox,
	QLabel, QHBoxLayout, QMessageBox, QHeaderView, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt

from models_analyses.trend_analysis_module import perform_actual_trend_analysis, analyze_multiple_disciplines_in_project
import os  # для создания безопасных имен папок


class SelectionDialog(QDialog):
	"""
	Модальный диалог для выбора параметров тренд-анализа.
	"""
	
	def __init__(self, df_merged, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Выбор проекта и дисциплин для тренд-анализа")
		self.setMinimumSize(600, 650)  # x, y, width, height
		
		self.df_merged = df_merged
		self.project_selection_group = None
		
		self.init_ui()
	
	def init_ui(self):
		main_layout = QVBoxLayout(self)
		
		# 1. Таблица с уникальными комбинациями
		table_label = QLabel("Уникальные комбинации Проект / Дисциплина :")
		main_layout.addWidget(table_label)
		
		self.unique_data_df = self.df_merged[
			['project_name', 'discipline']].drop_duplicates().reset_index(drop=True)
		self.table_widget = QTableWidget()
		self.table_widget.setColumnCount(len(self.unique_data_df.columns))
		self.table_widget.setHorizontalHeaderLabels(self.unique_data_df.columns)
		self.table_widget.setRowCount(len(self.unique_data_df))
		
		# Заполняем таблицу данными
		for row_idx, row_data in self.unique_data_df.iterrows():
			for col_idx, item in enumerate(row_data):
				self.table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(item)))
		
		# Автоматически растягиваем столбцы по содержимому
		self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		main_layout.addWidget(self.table_widget)
		
		# Раздел выбора проекта
		project_selection_group = QVBoxLayout()
		project_selection_group.addWidget(QLabel("--- Выбор проекта для анализа ---"))
		project_select_layout = QHBoxLayout()
		
		project_select_layout.addWidget(QLabel("Выберите проект"))
		self.project_combobox = QComboBox()
		
		# Получаем уникальные проекты из всего DataFrame
		unique_projects = self.df_merged['project_name'].dropna().unique().tolist()
		unique_projects.sort()
		self.project_combobox.addItems([str(p) for p in unique_projects])
		
		# При изменении выбранного проекта, обновляем список дисциплин
		self.project_combobox.currentIndexChanged.connect(self.update_discipline_list)
		project_select_layout.addWidget((self.project_combobox))
		
		project_selection_group.addLayout(project_select_layout)
		main_layout.addLayout(project_selection_group)
		
		# --- Раздел выбора ДИСЦИПЛИН (для выбранного проекта) ---
		discipline_selection_group = QVBoxLayout()
		discipline_selection_group.addWidget(QLabel("--- Выбор дисциплин в рамках проекта (опционально) ---"))
		
		self.discipline_list_label = QLabel("Доступные дисциплины для выбранного проекта:")
		discipline_selection_group.addWidget(self.discipline_list_label)
		
		self.discipline_list_widget = QListWidget()
		# Позволяем выбирать несколько элементов
		self.discipline_list_widget.setSelectionMode(QListWidget.MultiSelection)
		discipline_selection_group.addWidget(self.discipline_list_widget)
		
		main_layout.addLayout(discipline_selection_group)
		
		# инициализируем список дисциплин при старте
		
		# Кнопки для запуска анализа
		button_layout = QHBoxLayout()
		
		"""--- Анализ выбранного проекта ---"""
		# Кнопка для анализа ТОЛЬКО выбранного проекта
		self.analyze_project_button = QPushButton("Анализировать ВЕСЬ проект")
		self.analyze_project_button.clicked.connect(self.run_project_analysis)
		button_layout.addWidget(self.analyze_project_button)
		
		# Кнопка для анализа ВЫБРАННЫХ дисциплин ВНУТРИ проекта
		self.analyze_disciplines_in_project_button = QPushButton("Анализировать ВЫБРАННЫЕ дисциплины")
		self.analyze_disciplines_in_project_button.clicked.connect(self.run_selected_disciplines_analysis)
		button_layout.addWidget(self.analyze_disciplines_in_project_button)
		
		# Кнопка "Закрыть"
		self.close_button = QPushButton("Закрыть")
		self.close_button.clicked.connect(self.reject)
		button_layout.addWidget(self.close_button)
		
		main_layout.addLayout(button_layout)
		
		self.update_discipline_list()
	
	def update_discipline_list(self):
		"""Обновляет список дисциплин в QListWidget в зависимости от выбранного проекта."""
		self.selected_project = self.project_combobox.currentText()
		self.discipline_list_widget.clear()  # Очищаем список перед заполнением
		
		if self.selected_project:
			# Фильтруем DataFrame по выбранному проекту
			disciplines_in_project_df = self.df_merged[self.df_merged['project_name'] == self.selected_project]
			
			# Получаем уникальные дисциплины из отфильтрованного DataFrame
			unique_disciplines = disciplines_in_project_df['discipline'].dropna().unique().tolist()
			unique_disciplines.sort()
			
			if not unique_disciplines:
				self.discipline_list_label.setText(f"Нет дисциплин в проекте {self.selected_project}")
				self.analyze_disciplines_in_project_button.setEnabled(False)  # Отключаем кнопку, если нет дисциплин
			else:
				self.discipline_list_label.setText(f"Доступные дисциплины для '{self.selected_project}':")
				self.analyze_disciplines_in_project_button.setEnabled(True)  # Включаем кнопку
				for discipline in unique_disciplines:
					self.discipline_list_widget.addItem(str(discipline))
		else:
			self.discipline_list_label.setText("Выберите проект для отображения дисциплин.")
			self.analyze_disciplines_in_project_button.setEnabled(False)  # Отключаем кнопку, если проект не выбран
	
	def get_safe_project_folder_name(self, project_name):
		""" Создает безопасное имя папки из имени проекта """
		return project_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?',
		                                                                                                     '_').replace(
			'"', '_').replace('<', '_').replace('>', '_').replace('|', '_').strip()  # Добавлен strip())
	
	def run_project_analysis(self):
		"""Запускает анализ для всего выбранного проекта."""
		self.selected_project = self.project_combobox.currentText()
		if not self.selected_project:
			QMessageBox.warning(self, "Предупреждение", "Пожалуйста, выберите проект для анализа.")
			return
		# Создаем имя подпапки для проекта
		project_folder_name = self.get_safe_project_folder_name(self.selected_project)
		
		QMessageBox.information(self, "Анализ запущен",
		                        f"Запускаем тренд-анализ для всего проекта: '{self.selected_project}'\n"
		                        f"Результаты будут сохранены в папку: 'Analysis-Results/Trend_Analysis/{project_folder_name}'")
		
		# Фильтруем DataFrame для передачи в функцию анализа
		df_for_analysis = self.df_merged[self.df_merged['project_name'] == self.selected_project].copy()
		
		# Вызываем perform_actual_trend_analysis для всего проекта
		perform_actual_trend_analysis(
			df_for_analysis,  # Передаем DataFrame, отфильтрованный по проекту
			'project_name',  # Указываем, что анализируем по project_name
			self.selected_project,  # Передаем само имя проекта
			output_subfolder=project_folder_name,
			parent_widget=self
		)
	
	# Диалог остается открытым
	
	def run_selected_disciplines_analysis(self):
		"""Запускает анализ для выбранных дисциплин в рамках проекта."""
		self.selected_project = self.project_combobox.currentText()
		selected_disciplines = [item.text() for item in self.discipline_list_widget.selectedItems()]
		
		if not self.selected_project:
			QMessageBox.warning(self, "Предупреждение", "Пожалуйста, сначала выберите проект.")
			return
		
		if not selected_disciplines:
			QMessageBox.warning(self, "Предупреждение", "Пожалуйста, выберите хотя бы одну дисциплину для анализа.")
			return
		# создаем имя подпапки для проекта
		project_folder_name = self.get_safe_project_folder_name(self.selected_project)
		
		QMessageBox.information(self, "Анализ запущен",
		                        f"Запускаем тренд-анализ для дисциплин {', '.join(selected_disciplines)} "
		                        f"в проекте '{self.selected_project}'\n"
		                        f"Результаты будут сохранены в папку: 'Analysis-Results/Trend_Analysis/{project_folder_name}'")
		
		# Сначала фильтруем основной DataFrame по выбранному проекту
		df_for_disciplines_analysis = self.df_merged[self.df_merged['project_name'] == self.selected_project].copy()
		
		analyze_multiple_disciplines_in_project(
			df_for_disciplines_analysis,
			selected_disciplines,
			output_subfolder=project_folder_name,  # <-- ЭТА СТРОКА ДОЛЖНА БЫТЬ
			parent_widget=self
		)
		
		# Вызываем новую функцию для анализа нескольких дисциплин в проекте
		analyze_multiple_disciplines_in_project(
			df_for_disciplines_analysis,  # DataFrame, уже отфильтрованный по проекту
			selected_disciplines,  # Список выбранных дисциплин
			parent_widget=self
		)
	

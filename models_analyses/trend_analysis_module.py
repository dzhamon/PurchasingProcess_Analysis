from PyQt5.QtWidgets import QMessageBox, QWidget
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # Для работы с путями к файлам
from utils.config import BASE_DIR


# Определяем корневую директорию для всех тренд-анализов
TREND_ANALYSIS_ROOT_DIR = os.path.join(BASE_DIR, 'Analysis-Results', 'Trend-Analysis')

# -- 1. Универсальная функция для анализа одного элемента

def analyze_time_series_data(df_data: pd.DataFrame, group_column: str, group_value: str,
                             output_subfolder: str = None, parent_widget: QWidget = None):
	print(f"DEBUG: Внутри analyze_time_series_data для {group_column}: '{group_value}'")
	print(f"DEBUG: Получен output_subfolder = '{output_subfolder}'")
	print(f"DEBUG: Тип output_subfolder = {type(output_subfolder)}")
	
	# Тренд анализ затрат по дисциплине
	if pd.api.types.is_datetime64_any_dtype(df_data['year_month']):
		pass
	elif isinstance(df_data['year_month'].dtype, pd.PeriodDtype):
		df_data['year_month'] = df_data['year_month'].dt.to_timestamp()
	else:
		df_data['year_month'] = pd.to_datetime(df_data['year_month'])
	
	filtered_data = df_data[df_data[group_column] == group_value].copy()
	
	# Группировка по месяцам и суммирование total_contract_amount_eur
	monthly_costs = filtered_data.groupby(pd.Grouper(key='year_month', freq='MS'))[
		'total_contract_amount_eur'].sum().reset_index()
	monthly_costs = monthly_costs.sort_values('year_month')
	
	# Проверка на наличие данных для построения графика
	if monthly_costs.empty or len(monthly_costs) < 2:
		QMessageBox.information(parent_widget, "Нет данных",
		                        f"Недостаточно данных для построения тренда по {group_column}: '{group_value}'.")
		return
	
	# Создание графика Matplotlib
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(monthly_costs['year_month'], monthly_costs['total_contract_amount_eur'], marker='o', linestyle='-',
	        color='blue', label='Фактические затраты')
	
	# --- Добавление линии тренда ---
	x = np.arange(len(monthly_costs['year_month']))
	y = monthly_costs['total_contract_amount_eur']
	
	# Проверка на то, что данных достаточно для polyfit (минимум 2 точки)
	if len(x) >= 2:
		z = np.polyfit(x, y, 1)  # Вычисляем коэффициенты для полинома первой степени (линейная регрессия)
		p = np.poly1d(z)  # Создаем полиномиальную функцию
		ax.plot(monthly_costs['year_month'], p(x), "r--", label='Линия тренда')  # Добавляем линию тренда на график
	else:
		QMessageBox.information(parent_widget, "Внимание",
		                        f"Недостаточно точек для построения линии тренда по {group_column}: '{group_value}'.")
	
	ax.set_title(f'Динамика затрат по {group_column}: "{group_value}"', fontsize=14)
	ax.set_xlabel('Месяц', fontsize=12)
	ax.set_ylabel('Общая сумма контрактов (EUR)', fontsize=12)
	ax.grid(True, linestyle='--', alpha=0.7)
	ax.tick_params(axis='x', rotation=45)
	plt.tight_layout()
	ax.legend()  # Отображаем легенду
	
	# --- Сохранение графика в PNG-файл ---
	safe_value = str(group_value).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?',
	                                                                                                               '_').replace(
		'"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
	
	# Определяем директорию для сохранения
	if output_subfolder:
		OUT_DIR = os.path.join(TREND_ANALYSIS_ROOT_DIR, output_subfolder)
		print(f"DEBUG: Используем новый путь: {OUT_DIR}")
	else:
		print("DEBUG: output_subfolder был None, используем старую логику пути.")
		if group_column == 'discipline':
			OUT_DIR = os.path.join(BASE_DIR, 'discipline_trend_analyze')
		elif group_column == 'project_name':
			OUT_DIR = os.path.join(BASE_DIR, 'project_trend_analyze')
		else:
			OUT_DIR = os.path.join(BASE_DIR, 'general_trend_analyze')
			print(f"DEBUG: Используем старый путь: {OUT_DIR}")

	os.makedirs(OUT_DIR, exist_ok=True)
	file_name = f"trend_analysis_{group_column}_{safe_value}.png"
	file_path = os.path.join(OUT_DIR, file_name)
	
	try:
		plt.savefig(file_path, dpi=300)
	except Exception as e:
		QMessageBox.critical(parent_widget, "Ошибка сохранения", f"Не удалось сохранить график: {e}")
	finally:
		plt.close(fig)
	return

def perform_actual_trend_analysis(df_data: pd.DataFrame, selected_column: str, selected_value: str,
                                  output_subfolder: str = None, parent_widget: QWidget = None):
	if selected_column == 'project_name':
		analyze_time_series_data(df_data, selected_column, selected_value,
		                         output_subfolder=output_subfolder, parent_widget=parent_widget)
	elif selected_column == 'discipline':
		analyze_time_series_data(df_data, selected_column, selected_value, parent_widget=parent_widget)
	else:
		QMessageBox.warning(parent_widget, "Неизвестный тип анализа",
		                    f"Выбран неизвестный тип анализа: {selected_column}")

# Новая функция для анализа нескольких дисциплин в рамках уже отфильтрованного проекта
def analyze_multiple_disciplines_in_project(df_data: pd.DataFrame, selected_column: str,
                                            selected_value: str, output_subfolder: str = None,
                                            parent_widget: QWidget = None):
	"""
	Выполняет тренд-анализ для списка указанных дисциплин в рамках переданного DataFrame.
	Предполагается, что df_data уже отфильтрован по проекту
	"""
	print(f"DEBUG: perform_actual_trend_analysis вызван для {selected_column}: {selected_value}. output_subfolder = {output_subfolder}")
	if selected_column =='project_name':
		analyze_time_series_data(df_data, selected_column, selected_value, output_subfolder=output_subfolder,
		                         parent_widget=parent_widget)
	elif selected_column == 'discipline':
		analyze_time_series_data(df_data, selected_column, selected_value,
		                         output_subfolder=output_subfolder, parent_widget=parent_widget)
	else:
		QMessageBox.warning(parent_widget, "Неизвестный тип анализа",
		                    f"Выбран неизвестный тип анализа: {selected_column}")
	
# Новая функция для анализа нескольких дисциплин в рам ках уже отфильтрованного проекта

def analyze_multiple_disciplines_in_project(df_data: pd.DataFrame, disciplines_to_analyze: list,
                                            output_subfolder: str = None, # Новый аргумент
                                            parent_widget: QWidget = None):
	""" --- Начало анализа по дисциплинам, выбранным в рамках анализируемого проекта ---"""
	print(f"DEBUG: analyze_multiple_disciplines_in_project вызван. output_subfolder = {output_subfolder}")
	if not disciplines_to_analyze:
		QMessageBox.information(parent_widget, "Нет выбранных дисциплин", "Не выбраны дисциплины для анализа.")
		return
	
	if df_data.empty:
		QMessageBox.information(parent_widget, "Нет данных",
		                        "В выбранном проекте отсутствуют данные для анализа дисциплин.")
		return
	
	for discipline in disciplines_to_analyze:
		if pd.isna(discipline) or str(discipline).strip() == '':
			continue  # Пропускаем пустые/NaN значения
		
		# Вызываем универсальную функцию analyze_time_series_data
		# df_data здесь уже отфильтрован по проекту (если он был выбран в UI)
		analyze_time_series_data(df_data, 'discipline', discipline,
		                         output_subfolder=output_subfolder, parent_widget=parent_widget)
	
	QMessageBox.information(parent_widget, "Анализ завершен", "Тренд-анализ выбранных дисциплин завершен.")

# ... (остальные функции, например, analyze_all_disciplines, если она еще нужна для отдельного случая) ...
# Обратите внимание: analyze_all_disciplines, если она была, теперь будет использовать
# analyze_multiple_disciplines_in_project

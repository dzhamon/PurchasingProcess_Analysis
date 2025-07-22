import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
from sklearn.ensemble import IsolationForest
from utils.metal_weight_calculator import weight_calculator
from utils.vizualization_tools import plot_bar_chart, visualize_isolation_forest, create_plot_graf
from utils.functions import CurrencyConverter, clean_filename
from sklearn.neighbors import LocalOutlierFactor
import os


# 2. Анализ эффективности исполнителей
def analyze_efficiency(filtered_df):
	print('Запускается метод analyze_efficiency')
	
	# Проверка на пустые или некорректные значения
	filtered_df = filtered_df.dropna(subset=['unit_price', 'currency', 'total_price', 'supplier_qty'])
	filtered_df = filtered_df[filtered_df['unit_price'] > 0]
	
	# Конвертация валют
	columns_info = [
		('unit_price', 'currency', 'unit_price_in_eur'),
		('total_price', 'currency', 'total_price_in_eur')
	]
	converter = CurrencyConverter()
	filtered_df = converter.convert_multiple_columns(filtered_df, columns_info=columns_info)
	
	# Проверяем данные для модели
	model_columns = ['unit_price_in_eur', 'total_price_in_eur', 'supplier_qty']
	if filtered_df[model_columns].isnull().any().any():
		raise ValueError("Входные данные для модели содержат пропущенные значения.")
	
	# Вызываем Isolation Forest для аномалий
	model = IsolationForest(contamination=0.05, random_state=42)
	filtered_df['is_anomaly'] = model.fit_predict(
		filtered_df[['unit_price_in_eur', 'total_price_in_eur', 'supplier_qty']])
	
	# Аномалии: -1, нормальные: 1
	filtered_df['is_anomaly'] = filtered_df['is_anomaly'] == 1
	
	# Распределение цены за единицу
	output_folder = 'D:\Analysis-Results\efficient_analyses'
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	plt.figure(figsize=(10, 6))
	grouped = filtered_df.groupby('is_anomaly')
	for is_anomaly, group in grouped:
		if not group.empty:
			sns.histplot(group['total_price_in_eur'], bins=30, kde=True, label=f"Anomaly={is_anomaly}")
	
	plt.title('Distribution of Total Prices in EUR')
	plt.xlabel('Total Price (EUR)')
	plt.ylabel('Frequency')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	output_file = os.path.join(output_folder, 'price_distribution.png')
	plt.savefig(output_file)
	plt.close()
	print(f"Гистограмма распределения сохранена в: {output_file}")
	
	# Статистика
	stats = (
		filtered_df.groupby('discipline')
		.agg(
			median_unit_price_in_eur=('unit_price_in_eur', 'median'),
			mean_unit_price_in_eur=('unit_price_in_eur', 'mean'),
			std_unit_price_in_eur=('unit_price_in_eur', 'std'),
			median_total_price_in_eur=('total_price_in_eur', 'median'),
			mean_total_price_in_eur=('total_price_in_eur', 'mean'),
			std_total_price_in_eur=('total_price_in_eur', 'std'),
			median_supplier_qty=('supplier_qty', 'median'),
			mean_supplier_qty=('supplier_qty', 'mean'),
			std_supplier_qty=('supplier_qty', 'std')
		)
		.reset_index()  # Сбрасываем индекс, чтобы он стал колонкой
	)
	
	print("Statistics by anomaly status:")
	print(stats)
	
	return filtered_df, stats


# детальный анализ аномальных лотов
def detailed_anomaly_analysis(analyzed_df):
	"""
	Метод для анализа аномальных лотов из analyzed_df.
	"""
	print("Запущен detailed_anomaly_analysis")
	
	# 1. Проверяем наличие аномалий
	anomalous_lots = analyzed_df[analyzed_df['is_anomaly']]
	
	if anomalous_lots.empty:
		print("Нет аномальных данных для анализа.")
		return None
	
	# 2. Подготовка данных для анализа
	anomalous_summary = anomalous_lots[['lot_number', 'unit_price_in_eur', 'total_price_in_eur', 'supplier_qty',
	                                    'actor_name', 'winner_name']].sort_values(by='total_price_in_eur',
	                                                                              ascending=False)
	
	
	# Группировка по исполнителям и поставщикам
	actors_analysis = anomalous_lots.groupby('actor_name').size().sort_values(ascending=False).reset_index(
		name="Anomalous Lots Count")
	winners_analysis = anomalous_lots.groupby('winner_name').size().sort_values(ascending=False).reset_index(
		name="Anomalous Lots Count")
	
	# Визуализация actors_analysis и winners_analysis
	output_folder =  'D:\Analysis-Results\efficient_analyses'
	
	top_winners = winners_analysis.nlargest(20, "Anomalous Lots Count")  # Топ-20 победителей
	plt.figure(figsize=(16, len(top_winners) * 0.5))
	sns.barplot(
		data=top_winners,
		x="Anomalous Lots Count",
		y="winner_name",
		orient="h",
		palette="viridis",
		dodge=False
	)
	plt.title("Top 20 Anomalous Lots Count by Winner")
	plt.xlabel("Anomalous Lots Count")
	plt.ylabel("Winner Name")
	plt.tight_layout()
	winners_plot_path = os.path.join(output_folder, "winners_analysis.png")
	plt.savefig(winners_plot_path)
	plt.close()
	print(f"Визуализация по победителям сохранена в: {winners_plot_path}")
	
	top_actors = actors_analysis.nlargest(20, "Anomalous Lots Count")  # Топ-10 исполнителей
	plt.figure(figsize=(16, len(top_actors) * 0.5))
	sns.barplot(
		data=top_actors,
		x="Anomalous Lots Count",
		y="actor_name",
		orient="h",
		palette="viridis",
		dodge=False
	)
	plt.title("Top 20 Anomalous Lots Count by Actor")
	plt.xlabel("Anomalous Lots Count")
	plt.ylabel("Actor Name")
	plt.tight_layout()
	actors_plot_path = os.path.join(output_folder, "actors_analysis.png")
	plt.savefig(actors_plot_path)
	plt.close()
	print(f"Визуализация по исполнителям сохранена в: {actors_plot_path}")
	
	# 3. Сохранение результатов
	output_folder = 'D:\Analysis-Results\efficient_analyses'
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	output_file = f"{output_folder}\Anomalous_Lots_Analysis.xlsx"
	with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
		anomalous_summary.to_excel(writer, sheet_name='Anomalous Lots', index=False)
		actors_analysis.to_excel(writer, sheet_name='Actors Analysis', index=False)
		winners_analysis.to_excel(writer, sheet_name='Winners Analysis', index=False)
	
	print(f"Результаты анализа аномалий сохранены в файл: {output_file}")
	return anomalous_summary, actors_analysis, winners_analysis

def detailed_results_analyses(analyzed_df, anomaly_stats):
	# 1. Добавление границ, для выявления аномалий
	k = 3  # Коэффициент чувствительности (например, 3 стандартных отклонения)
	
	# Расширение  anomaly_stats нижними и верхними границами статистических показателей
	anomaly_stats["lower_bound_unit_price"] = anomaly_stats["mean_unit_price_in_eur"] - k * anomaly_stats[
		"std_unit_price_in_eur"]
	anomaly_stats["upper_bound_unit_price"] = anomaly_stats["mean_unit_price_in_eur"] + k * anomaly_stats[
		"std_unit_price_in_eur"]
	
	anomaly_stats["lower_bound_total_price"] = anomaly_stats["mean_total_price_in_eur"] - k * anomaly_stats[
		"std_total_price_in_eur"]
	anomaly_stats["upper_bound_total_price"] = anomaly_stats["mean_total_price_in_eur"] + k * anomaly_stats[
		"std_total_price_in_eur"]
	
	# Шаг 2: Выявление аномалий в analyzed_df
	# Слияние для добавления границ к исходным данным (по discipline)
	analyzed_with_bounds = analyzed_df.merge(
		anomaly_stats[["discipline", "lower_bound_unit_price", "upper_bound_unit_price",
		               "lower_bound_total_price", "upper_bound_total_price"]],
		on="discipline",
		how="left"
	)
	
	# Фильтрация аномалий
	anomalies = analyzed_with_bounds[
		(analyzed_with_bounds["unit_price_in_eur"] < analyzed_with_bounds["lower_bound_unit_price"]) |
		(analyzed_with_bounds["unit_price_in_eur"] > analyzed_with_bounds["upper_bound_unit_price"]) |
		(analyzed_with_bounds["total_price_in_eur"] < analyzed_with_bounds["lower_bound_total_price"]) |
		(analyzed_with_bounds["total_price_in_eur"] > analyzed_with_bounds["upper_bound_total_price"])
		]
	
	# Шаг 3: Визуализация
	# Директория для сохранения графика
	output_dir = r"D:\Analysis-Results\efficient_analyses"
	os.makedirs(output_dir, exist_ok=True)
	
	# Путь к файлу
	output_file_path = os.path.join(output_dir, "anomalies_visualization.png")
	
	# График всех точек с выделением аномалий
	plt.figure(figsize=(10, 6))
	plt.scatter(analyzed_df["unit_price_in_eur"], analyzed_df["total_price_in_eur"], alpha=0.6, label="Normal Data")
	plt.scatter(anomalies["unit_price_in_eur"], anomalies["total_price_in_eur"], color="red", alpha=0.8,
	            label="Anomalies")
	plt.xlabel("Unit Price (EUR)")
	plt.ylabel("Total Price (EUR)")
	plt.title("Anomalies in Unit Price and Total Price")
	plt.legend()
	plt.grid()
	plt.grid()
	
	# Сохранение графика в PNG
	plt.savefig(output_file_path, format='png', dpi=300)  # dpi=300 для высокого качества
	
	# Отображение графика
	plt.show()
	
	print(f"График успешно сохранён в файл: {output_file_path}")
	
	# Шаг 4: Вывод аномальных точек
	# Убедимся, что директория существует
	output_dir = r"D:\Analysis-Results\efficient_analyses"
	os.makedirs(output_dir, exist_ok=True)
	# Путь к файлу
	output_file_path = os.path.join(output_dir, "anomalies.xlsx")
		# Запись в Excel
	anomalies[["lot_number", "discipline", "winner_name", "unit_price_in_eur", "total_price_in_eur"]].to_excel(
		output_file_path, index=False)
	print(f"Аномалии успешно сохранены в файл: {output_file_path}")
	


# вызов функций из главного метода
def main_method(filtered_df, data_df, parent_widget=None):
	print("Мы вошли в метод main_method")
	output_folder = 'D:\Analysis-Results\efficient_analyses'
	print(filtered_df.columns)
	
	# добираем товары выбранной категории в датафрейм
	selected_lots = filtered_df['lot_number'].unique()
	filtered_df = data_df[data_df['lot_number'].isin(selected_lots)]
	
	analyzed_df, stats = analyze_efficiency(filtered_df)
	
	# Сохранение анализа эффективности исполнителей
	file_path = os.path.join(output_folder, "Efficiency_Metrics.xlsx")
	print(f"Сохраняем файл в: {file_path}")
	
	if not analyzed_df.empty:
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		
		file_saved = False
		while not file_saved:
			try:
				# Сохраняем основной анализ
				analyzed_df.to_excel(file_path, index=False)
				print("Анализ эффективности исполнителей сохранен в файл 'Efficiency_Metrics.xlsx'")
				
				# Сохраняем статистику по аномалиям
				with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
					stats.to_excel(writer, sheet_name='Anomaly_Stats')
				print("Статистика по аномалиям добавлена в файл 'Efficiency_Metrics.xlsx' на лист 'Anomaly_Stats'.")
				
				file_saved = True
			except PermissionError:
				print("Файл используется другой программой.")
				msg_box = QMessageBox(parent_widget)
				msg_box.setIcon(QMessageBox.Warning)
				msg_box.setWindowTitle("Файл используется")
				msg_box.setText(
					"Файл 'Efficiency_Metrics.xlsx' уже используется другой программой.\nПожалуйста, закройте файл и нажмите OK для продолжения.")
				msg_box.setStandardButtons(QMessageBox.Ok)
				msg_box.exec_()
			else:
				print("Нет данных для анализа эффективности исполнителей для сохранения.")
			
			# занимаемся анализом аномальных лотов
			detailed_anomaly_analysis(analyzed_df)
			
			# Визуализация Isolation Forest
			visualize_isolation_forest(analyzed_df)
			
			# детальный анализ полученных результатов
			detailed_results_analyses(analyzed_df, stats)
			
			
			
			return analyzed_df


def display_dataframe_to_user(name, dataframe):
	"""
	Отображает DataFrame в PyQt таблице.
	:param name: Название таблицы.
	:param dataframe: DataFrame для отображения.
	"""
	from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel, QDialog
	import pandas as pd
	
	# Создаём диалоговое окно
	dialog = QDialog()
	dialog.setWindowTitle(name)
	
	# Создаём таблицу
	table = QTableWidget()
	table.setRowCount(len(dataframe))
	table.setColumnCount(len(dataframe.columns))
	table.setHorizontalHeaderLabels(dataframe.columns)
	
	# Заполняем таблицу данными
	for i, row in enumerate(dataframe.itertuples(index=False)):
		for j, value in enumerate(row):
			table.setItem(i, j, QTableWidgetItem(str(value)))
	
	# Добавляем виджеты в диалог
	layout = QVBoxLayout()
	layout.addWidget(QLabel(f"<h3>{name}</h3>"))
	layout.addWidget(table)
	dialog.setLayout(layout)
	
	# Показываем окно
	dialog.exec_()

import os
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox

def analyze_suppliers_by_unit_price(parent_widget, mydata_df, update_progress, create_plot):
	import re
	"""Анализ поставщиков по средней цене за единицу товара с расширенной статистикой.

	Args:
		parent_widget: Родительский виджет для QMessageBox.
		mydata_df (pd.DataFrame): DataFrame с данными о закупках.
		update_progress (signal): Сигнал для обновления прогресс-бара.
		create_plot (function): Функция для построения графиков.
	"""
	print("Запуск analyze_suppliers_by_unit_price")
	update_progress.emit(0)
	
	# --- 1. Настройка пути и папки для результатов ---
	output_folder = 'D:/Analysis-Results/suppliers_by_unit_price'
	os.makedirs(output_folder, exist_ok=True)
	update_progress.emit(10)
	
	# --- 2. Конвертация валют ---
	if 'unit_price' in mydata_df.columns and 'currency' in mydata_df.columns:
		columns_info = [('unit_price', 'currency', 'unit_price_eur')]
		converter = CurrencyConverter()
		mydata_df = converter.convert_multiple_columns(mydata_df, columns_info)
	update_progress.emit(20)
	
	# --- 3. Нормализация данных c наименованиями товаров ---
	mydata_df['good_name'] = mydata_df['good_name'].str.strip().str.lower()
	
	# --- 4. Фильтрация товаров с разными поставщиками ---
	goods_stats = mydata_df.groupby('good_name').agg(
		suppliers_count=('winner_name', 'nunique'),
		lots_count=('lot_number', 'nunique')
	).reset_index()
	
	# Берём найденные товары
	filtered_goods = goods_stats['good_name']
	# Добавим метку "единичный поставщик" в основной датафрейм
	mydata_df['is_single'] = mydata_df.groupby(['good_name', 'winner_name'])['lot_number'].transform('nunique') >= 1
	
	if filtered_goods.empty:
		QMessageBox.information(parent_widget, "Результат", "Нет товаров с несколькими поставщиками.")
		return None
	update_progress.emit(30)
	
	# --- 5. Пересчёт единиц измерения ---
	def convert_units(row):
		"""Приводит цену к стандартной единице (например, тонна)."""
		try:
			# определение единицы измерения
			unit = None
			for col in ['supplier_unit', 'unit']:
				if col in row and pd.notna(row[col]):
					unit = str(row[col]).strip().lower()
					break
			if not unit:
				print(f"Внимание: отсутствует единица измерения в строке {row.name}")
				return None
			# 2. Проверка цены
			try:
				price = float(row['unit_price_eur'])
				if price <= 0:
					print(f"Ошибка: некорректная цена ({price}) в строке {row.name}")
					return None
			except (ValueError, TypeError):
				print(f"Ошибка: нечисловая цена в строке {row.name}")
				return None
			
			# 3. Логика конвертации
			CONVERSION_RATES = {
				'кг': 1000,  # кг → тонны
				'kg': 1000,
				'г': 1000000,  # граммы → тонны
				'gram': 1000000,
				'м': lambda x: 1 / (weight_calculator(x) if weight_calculator(x) else None),
				'метр': lambda x: 1 / (weight_calculator(x) if weight_calculator(x) else None)
			}
			
			if unit in CONVERSION_RATES:
				rate = CONVERSION_RATES[unit]
				return price * rate if isinstance(rate, (int, float)) else rate(row)
			else:
				return price
		except Exception as e:
			print(f"Критическая ошибка в строке {row.name}: {str(e)}")
			return None
	def clean_sheet_name(name):
		"""Очищает название листа от недопустимых Excel символов."""
		# Удаляем запрещенные символы: []:*?/\
		cleaned = re.sub(r'[\[\]:*?/\\]', '', name)
		# Обрезаем до 31 символа (ограничение Excel)
		return cleaned[:31]
	
	mydata_df['unit_price_eur_std'] = mydata_df.apply(convert_units, axis=1)
	mydata_df.dropna(subset=['unit_price_eur_std'], inplace=True)
	update_progress.emit(40)
	
	# --- 6. Агрегация данных по поставщикам ---
	result_tables = []
	
	for good_name in filtered_goods:
		good_data = mydata_df[mydata_df['good_name'] == good_name]
		
		supplier_stats = good_data.groupby('winner_name').agg({
			'unit_price_eur_std': ['mean', 'median', 'min', 'max', 'std'],
			'lot_number': ['nunique', lambda x: ', '.join(map(str, x.unique()))],
			'supplier_qty': 'sum',
			'close_date': 'max'  # Дата последней поставки
		}).reset_index()
		
		# Переименование столбцов
		supplier_stats.columns = [
			'winner_name',
			'avg_unit_price',
			'median_price',
			'min_price',
			'max_price',
			'price_std',
			'lots_count',
			'lot_numbers',
			'total_quantity',
			'last_purchase_date'
		]
		
		supplier_stats['good_name'] = good_name
		result_tables.append(supplier_stats)
	update_progress.emit(60)
	
	# --- 7. Сохранение результатов ---
	final_table = pd.concat(result_tables, ignore_index=True)
	final_table = final_table.sort_values(
		by=['good_name', 'avg_unit_price'],
		ascending=[True, True]
	)
	
	# Сохранение в Excel с несколькими листами
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	excel_file = os.path.join(output_folder, f"supplier_analysis_{timestamp}.xlsx")
	
	with pd.ExcelWriter(excel_file) as writer:
		# Сводный отчёт
		final_table.to_excel(writer, sheet_name="Summary", index=False)
		
		# Отдельные листы для каждого товара
		for good_name in filtered_goods:
			good_data = final_table[final_table['good_name'] == good_name]
			if not good_data.empty:
				try:
					sheet_name = clean_sheet_name(good_name)
					good_data.to_excel(writer, sheet_name=sheet_name, index=False)
				except Exception as e:
					print(f"Не удалось создать лист для товара {good_name}: {str(e)}")
					
			# sheet_name = good_name[:30]  # Ограничение длины имени листа
			# good_data.to_excel(writer, sheet_name=sheet_name, index=False)
	update_progress.emit(80)
	
	# --- 8. Визуализация ---
	for good_name in filtered_goods:
		try:
			if 'tруба' in good_name.lower():
				# Нормализация названия (на всякий случай)
				normalized_name = good_name.lower().replace('t', 'т')
				
				# Получаем данные только для текущего товара
				good_data = final_table[final_table['good_name'] == normalized_name]
			else:
				good_data =  final_table[final_table['good_name'] == good_name]
			
			# Проверяем, что данные не пустые
			if good_data.empty:
				print(f"Нет данных для визуализации товара: {good_name}")
				continue  # Пропускаем эту итерацию
			
			# Проверяем наличие необходимых столбцов
			required_columns = ['avg_unit_price', 'min_price', 'max_price']
			missing_cols = [col for col in required_columns if col not in good_data.columns]
			if missing_cols:
				print(f"Отсутствуют столбцы {missing_cols} для товара {good_name}")
				continue
			
			# Создаём график
			create_plot_graf(
				good_name,
				good_data,
				output_folder,
			)
		
		except Exception as e:
			print(f"Ошибка при обработке товара {good_name}: {str(e)}")
			continue  # Продолжаем цикл со следующим товаром
	update_progress.emit(100)
	
	print(f"Анализ завершён. Результаты сохранены в: {excel_file}")
	return final_table

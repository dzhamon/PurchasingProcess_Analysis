import os
import sys
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QTableView, QAbstractItemView, QHBoxLayout,
                             QSizePolicy, QMessageBox, QApplication)
from utils.PandasModel_previous import PandasModel
from widgets.trend_analyze_prepare_widget import TrendAnalyzeWidget
import pandas as pd
from utils.functions import CurrencyConverter, save_analysis_results
from prophet import Prophet


def show_filtered_df(filtered_df):
	"""
	   Отображает отфильтрованные данные во всплывающем диалоговом окне.
	   """
	
	if filtered_df.empty:
		print('Нет данных для отображения!')
		return
	# Создание всплывающего диалогового окна
	dialog = QDialog()
	dialog.setWindowTitle("Отфильтрованные данные")
	dialog.setMinimumSize(800, 600)  # устанавливаем минм размер диалогового окна
	dialog.setMaximumSize(1900, 800)
	layout = QVBoxLayout(dialog)
	
	# Создаем виджет таблицы для отображения данных
	table_view = QTableView()
	model = PandasModel(filtered_df)
	table_view.setModel(model)
	table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
	layout.addWidget(table_view)
	
	# горизонтальная компоновка для кнопок
	button_layout = QHBoxLayout()
	
	# Кнопка для развертывания/сворачивания окна
	expand_button = QPushButton('Развернуть')
	expand_button.setCheckable(True)
	expand_button.toggled.connect(lambda checked: toggle_fullscreen(dialog, checked))
	button_layout.addWidget(expand_button)
	
	# кнопка для передачи данных на анализ
	analyze_button = QPushButton('Передать данные на анализ')
	analyze_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Фиксируем размер кнопки
	analyze_button.clicked.connect(lambda: send_data_to_analysis(filtered_df))
	button_layout.addWidget(analyze_button)
	
	# Добавляем кнопки в основной макет
	layout.addLayout(button_layout)
	dialog.setLayout(layout)
	dialog.exec_()


def send_data_to_analysis(filtered_df):
	"""
	Функция для передачи данных на анализ.
	"""
	print("Данные переданы на анализ:", filtered_df)


def toggle_fullscreen(dialog, checked):
	"""
	Разворачивает или возвращает окно к предыдущему размеру.
	"""
	if checked:
		dialog.showMaximized()
	else:
		dialog.showNormal()


def analyze_discrepancies(filtered_df, project_name, OUT_DIR):
	"""
	    Рассчитывает разницу и статус для каждого лота и возвращает результат в виде DataFrame.
	    """
	print('Мы в методе analyze_discrepancies')
	results = []
	print(filtered_df.columns)
	
	filtered_df = filtered_df[
		filtered_df['total_price_kp'].notna() & filtered_df['total_contract_amount'].notna()]
	
	# # и создаем excel-файл с помощью ExcelWriter
	# file_exls_name = f"Non_EquilSum_{project_name}.xlsx"
	# file_exls_path = os.path.join(OUT_DIR, file_exls_name)
	#
	# filtered_df.to_excel(file_exls_path, index=False, engine='openpyxl')
	
	for _, row in filtered_df.iterrows():
		lot_number = row['lot_number']
		supplier_kp = row['winner_name']
		total_price_kp = row['total_price_kp']
		total_contract_amount = row['total_contract_amount']
		supplier_qty_kp = row['supplier_qty_kp']
		quantity_contract = row['quantity_contract']
		unit_average_kp = row['unit_average_kp']
		unit_average_contract = row['unit_average_contract']
		currency = row['currency']
		actor_name = row['actor_name']
		executor_dak = row['executor_dak']
		discipline = row['discipline']
		
		# Рассчитываем процент увеличения/уменьшения цены за единицу товара
		discrepancy_percent = (unit_average_contract - unit_average_kp) / unit_average_kp * 100
		if discrepancy_percent >= 3.0:
			status = 'Нетипично для честной закупки. Проверить'
		else:
			status = ''
		
		# Добавляем все данные в список
		if status != '':
			results.append({
				'lot_number': lot_number,
				'currency': currency,
				'winner_name': supplier_kp,
				'supplier_qty_kp': supplier_qty_kp,
				'unit_average_kp': unit_average_kp,
				'total_price_kp': total_price_kp,
				'quantity_contract': quantity_contract,
				'unit_average_contract': unit_average_contract,
				'total_contract_amount': total_contract_amount,
				'discr_percent': discrepancy_percent,
				'status': status,
				'actor_name': actor_name,
				'executor_dak': executor_dak,
				'discipline': discipline
			})
	
	return pd.DataFrame(results)


def analyzeNonEquilSums(parent, data_df):
	print("Запускается метод проверки гипотезы неравенства сумм лотов и контрактов")
	try:
		if not data_df.empty:
			# Получим уникальные номера лотов min/max даты из data_df
			unique_lots = data_df['lot_number'].unique()
			min_date = data_df['close_date'].min()
			max_date = data_df['close_date'].max()
			project_name = data_df['project_name'].unique()
			
			# загрузим из таблицы data_contract базы данных соответствующие контракты
			from utils.config import SQL_PATH, BASE_DIR
			import sqlite3
			
			# Создаем папку для результатов, если её еще нет
			OUT_DIR = os.path.join(BASE_DIR, "Non-equil sum")
			os.makedirs(OUT_DIR, exist_ok=True)
			
			db_path = SQL_PATH
			conn = sqlite3.connect(db_path)
			# Создаем строку с плейсхолдерами для каждого lot_number
			placeholders = ','.join(['?'] * len(unique_lots))
			
			query = f"""
						SELECT * FROM data_contract
						WHERE DATE(contract_signing_date) BETWEEN DATE(?) AND DATE(?)
						AND lot_number IN ({placeholders})
					"""
			# Объединяем все параметры в один список
			params = [min_date, max_date] + unique_lots.tolist()
			
			contract_df = pd.read_sql_query(query, conn, params=params)
			conn.close()
			
			# Агрегация данных по лотам в data_kp (для каждой позиции суммируем количество и цену)
			kp_agg = data_df.groupby(['lot_number', 'currency', 'winner_name', 'actor_name', 'discipline']).agg(
				total_price_kp=pd.NamedAgg(column='total_price', aggfunc='sum'),
				supplier_qty_kp=pd.NamedAgg(column='supplier_qty', aggfunc='sum'),
				unit_average_kp=pd.NamedAgg(column='unit_price', aggfunc='mean'),
			).reset_index()
			
			# Агрегация данных по контрактам (по каждой позиции аналогично)
			# Удаляем строки с NaN в столбце 'total_contract_amount' из contract_df
			contract_df_cleaned = contract_df.dropna(subset=['contract_signing_date'])
			
			contract_agg = contract_df_cleaned.groupby(
				['lot_number', 'contract_currency', 'counterparty_name', 'executor_dak']).agg(
				total_contract_amount=pd.NamedAgg(column='total_contract_amount', aggfunc='sum'),
				quantity_contract=pd.NamedAgg(column='quantity', aggfunc='sum'),
				unit_average_contract=pd.NamedAgg(column='unit_price', aggfunc='mean'),
			).reset_index()
			
			merged_df = kp_agg.merge(
				contract_agg,
				left_on=['lot_number', 'currency', 'winner_name'],
				right_on=['lot_number', 'contract_currency', 'counterparty_name'],
				how='inner',  # Используем left join, чтобы сохранить все строки из kp
				suffixes=('_kp', '_contract')
			)
			
			# Фильтрация строк, где разница между суммой лота и контракта больше 0.01
			filtered_df = merged_df[abs(merged_df['total_price_kp'] - merged_df['total_contract_amount']) > 0.01]
			
			if filtered_df.shape[0] == 0.0:
				QMessageBox.information(parent, 'Сообщение', f"Сравнительный анализ завершен."
				                                                f"Для проекта {project_name} Нет разницы в суммах Лотов и Контрактов")
				return
			
			analysis_results = analyze_discrepancies(filtered_df, project_name, OUT_DIR)
			
			save_analysis_results(analysis_results, project_name, OUT_DIR)
			QMessageBox.information(parent, 'Сообщение', f"Сравнительный анализ завершен. "
			                                             f"Результаты успешно сохранены в папке "
			                                             f"{OUT_DIR}")
		
		else:
			QMessageBox(parent, 'Сообщение', "На входе метода - пустой Датафрейм")
		return
	except Exception as e:
		QMessageBox.critical(parent, "Ошибка", f"Не удалось отобразить данные: {str(e)}")


"""
	Тренд - анализ контрарактных данных
"""


def data_preprocessing_and_analysis(df):
	import pandas as pd
	
	try:
		# Выбираем из загруженных Контрактов уникальные лоты и
		# начальную и конечную даты загруженных данных
		unique_lots = df['lot_number'].unique()
		max_date = df['contract_signing_date'].max()
		min_date = df['contract_signing_date'].min()
		
		# приводим даты к обычному формату (ГГ-ММ-ДД)
		max_date_o = max_date.strftime('%Y-%m-%d')
		min_date_o = min_date.strftime('%Y-%m-%d')
		
		# загрузим из таблицы data_kp соотвествующие этим датам и уникальным номерам Лоты
		from utils.config import SQL_PATH, BASE_DIR
		import sqlite3
		
		# Создадим папку для результатов, если ее еще нет
		OUT_DIR = os.path.join(BASE_DIR, "Trend_Analyze")
		os.makedirs(OUT_DIR, exist_ok=True)
		
		# соединяемся сбазой данных
		db_path = SQL_PATH
		conn = sqlite3.connect(db_path)
		# Создаем строку с плейсхолдерами для каждого lot_number
		placeholders = ','.join(['?'] * len(unique_lots))
		
		query = f"""
					SELECT lot_number, project_name FROM data_kp
					WHERE DATE(close_date) BETWEEN DATE(?) AND DATE(?)
					AND lot_number IN ({placeholders})
				"""
		# Объединяем все параметры в один список
		params = [min_date_o, max_date_o] + unique_lots.tolist()
		
		data_kp_df = pd.read_sql_query(query, conn, params=params)
		conn.close()
	
		df['lot_number'] = df['lot_number'].astype(str)  # переводим номера лотов в базе Контрактов в строковый тип
		data_kp_df['lot_number'] = data_kp_df['lot_number'].astype(str)  # то же самое с базой Лотов
		
		data_kp_unique = data_kp_df.drop_duplicates()
		df_merged = pd.merge(df, data_kp_unique[['lot_number', 'project_name']], on='lot_number', how='left')
		
		# В df_merged обнаружены строки, где project_name_x = NA и project_name_y = nan.
		# Это говорит о том, что присутствуют контракты без их Лот-проработки. Нужно всех их выделить
		# в отдельный датафрейм.
		
		# Определим строки, подлежащие удалению из df_merged
		contracts_to_remove_mask = (
		    (df_merged['project_name_x'].isna() | (df_merged['project_name_x'] == 'NA')) &
		    df_merged['project_name_y'].isna()
		)
		# и создадим датафрейм с контрактами без лотов
		cont_less_lots_df = df_merged[contracts_to_remove_mask].copy()
		# по-хорошему, имея доступ к базе 1С можно было-ба организовать
		# автоматическую сверку data_kp на предмет отсутствующих Лотов
		
		# удалим столбец project_name_y и переименуем project_name_x в project_name
		df_merged.drop(columns=['project_name_y'], inplace=True)
		df_merged.rename(columns={'project_name_x': 'project_name'}, inplace=True)
		
		# Преобразование дат с обработкой ошибок
		# logging.info('Преобразование столбцов дат')
		df_merged['contract_signing_date'] = pd.to_datetime(df_merged['contract_signing_date'], errors='coerce')
		df_merged['lot_end_date'] = pd.to_datetime(df_merged['lot_end_date'], errors='coerce')
		
		# Удаление строк с некорректными датами
		# logging.info("Удаление строк с некорректными датами")
		df_merged = df_merged.dropna(subset=['contract_signing_date'])
		
		# Удаление дубликатов и пропусков в ключевых столбцах
		# logging.info('Удаление дубликатов')
		df_merged = df_merged.drop_duplicates()
		df_merged = df_merged.dropna(subset=['unit_price', 'quantity', 'product_name'])
		
		# Фильтрация аномальных значений
		# logging.info('Фильтрация аномальных значений')
		df_merged = df_merged[(df_merged['unit_price'] > 0) & (df_merged['quantity'] > 0)]
		
		# Добавление временных меток
		# logging.info('Добавление временных меток')
		df_merged['year_month'] = df_merged['contract_signing_date'].dt.to_period('M')
		df_merged['year'] = df_merged['contract_signing_date'].dt.year
		df_merged['month'] = df_merged['contract_signing_date'].dt.month
		
		# Переведем стоимости в единую валюту EUR
		
		columns_info = [
			('unit_price', 'contract_currency', 'unit_price_eur'),
			('total_contract_amount', 'contract_currency', 'total_contract_amount_eur')
		]
		converter = CurrencyConverter()
		
		# Конвертируем и сохраняем два столбца
		converted_df = converter.convert_multiple_columns(
			df=df_merged, columns_info=columns_info)
		
		df_merged['total_contract_amount_eur'] = converted_df['total_contract_amount_eur'].copy()
		df_merged['unit_price_eur'] = converted_df['unit_price_eur'].copy()
		
		return df_merged
	
	except Exception as e:
		print(f"Ошибка при формировании ... : {e}")


"""
	Временные ряды и прогнозирование
"""
def prophet_and_arima(filtered_df):
	print('Мы в методе Prophet_and_Arima')
	print(filtered_df.columns)
	converter = CurrencyConverter()
	columns_info = [('total_contract_amount', 'contract_currency', 'total_contract_amount_eur')]
	filtered_df = converter.convert_multiple_columns(filtered_df, columns_info)
	
	# Переименуем столбцы для работы с Prophet
	df_prophet = filtered_df[['contract_signing_date', 'total_contract_amount_eur']].rename(
		columns={'contract_signing_date': 'ds', 'total_contract_amount_eur': 'y'})
	# Инициализируем модель Prophet
	model = Prophet()
	# Обучим модель на наших данных
	model.fit(df_prophet)
	
	# Прогнозируем на 12 месяцев вперед
	future = model.make_future_dataframe(periods=6, freq='ME')
	forecast = model.predict(future)
	
	# Выведем прогноз
	print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
	
	# Построим график прогноза
	model.plot(forecast)


def prepare_contract_data(filtered_df):
	# Функция для выполнения DBSCAN
	def perform_dbscan(data, eps=0.1, min_samples=3):
		dbscan = DBSCAN(eps=eps, min_samples=min_samples)
		data['cluster'] = dbscan.fit_predict(data)
		return data
	
	# Функция для построения k-NN графика
	def plot_knn_graph(data, k=5):
		nbrs = NearestNeighbors(n_neighbors=k).fit(data)
		distances, indices = nbrs.kneighbors(data)
		distances = np.sort(distances[:, k - 1])  # Сортировка по k-му соседу
		plt.figure(figsize=(10, 6))
		plt.plot(distances)
		plt.title(f"k-NN Graph for DBSCAN (k={k})")
		plt.xlabel("Points sorted by distance to k-th nearest neighbor")
		plt.ylabel("Distance")
		plt.grid()
		plt.show()
		return
	
	# Преобразование 'кг' в 'тонны'
	filtered_df.loc[filtered_df['unit'] == 'кг', 'quantity'] /= 1000
	filtered_df.loc[filtered_df['unit'] == 'кг', 'unit'] = 'тонны'
	
	# Нормализация данных
	scaler = MinMaxScaler()
	numeric_features = ['quantity', 'unit_price', 'total_contract_amount']
	filtered_df[numeric_features] = scaler.fit_transform(filtered_df[numeric_features])
	
	# Выбор признаков для кластеризации
	features = filtered_df[['quantity', 'unit_price', 'total_contract_amount']]
	
	# Построение k-NN графика
	plot_knn_graph(features, k=5)
	
	# Выполнение DBSCAN
	eps = 0.15  # Оптимальное значение можно выбрать после анализа графика
	min_samples = 5
	clustered_data = perform_dbscan(features, eps=eps, min_samples=min_samples)
	
	# Анализ результатов
	clusters_summary = clustered_data.groupby('cluster').mean()
	print(f"Количество точек шума: {len(clustered_data[clustered_data['cluster'] == -1])}")
	print(f"Количество кластеров: {len(clustered_data['cluster'].unique()) - 1}")  # -1 для исключения шума
	
	return clustered_data, clusters_summary

# Запуск обработки
# clustered_data, clusters_summary = main(contract_df)

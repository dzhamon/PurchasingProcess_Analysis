import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QTableView, QAbstractItemView, QHBoxLayout,
                             QSizePolicy, QMessageBox, QApplication)
from utils.PandasModel_previous import PandasModel
from widgets.trend_analyze_prepare_widget import TrendAnalyzeWidget
import pandas as pd
from utils.functions import CurrencyConverter, save_analysis_results
from utils.config import BASE_DIR
# from prophet import Prophet


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
	import os
	
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
		# по-хорошему, имея прямой доступ к базе 1С можно было-ба организовать
		# автоматическую сверку data_kp на предмет отсутствующих Лотов

		output_path = os.path.join(OUT_DIR, "Contracts_Without_Lots_Anomaly.xlsx")
		cont_less_lots_df.to_excel(output_path, index=False)
		print(f"Аномальные контракты сохранены в: {output_path}")

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
		
		return df_merged, cont_less_lots_df
	
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
		plt.xlabel("Точки, отсортированные по расстоянию до k-го ближайшего соседа")
		plt.ylabel("Расстояние")
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

def run_sarima_forecast():
	pass

def analyze_monthly_cost_cont(parent_widget, df, start_date, end_date):
	from matplotlib.ticker import FuncFormatter
	from scipy import stats
	import numpy as np
	import os
	from datetime import datetime

	"""
	Расширенный анализ месячных затрат с разбивкой по дисциплинам
	"""
	# Создаем папку для результатов
	OUT_DIR = os.path.join(BASE_DIR, "monthly_cost_analysis")
	os.makedirs(OUT_DIR, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	# Конвертация и фильтрация данных
	df['contract_signing_date'] = pd.to_datetime(df['contract_signing_date'], errors='coerce')
	filtered_df = df[(df['contract_signing_date'] >= start_date) & (df['contract_signing_date'] <= end_date)].copy()

	if filtered_df.empty:
		QMessageBox.warning(parent_widget, "Ошибка", "Нет данных для заданного диапазона дат.")
		return

	# Добавляем столбцы для анализа
	filtered_df['year_month'] = filtered_df['contract_signing_date'].dt.to_period('M')
	filtered_df['month_name'] = filtered_df['contract_signing_date'].dt.strftime('%Y-%m')

	# Конвертация в EUR
	try:
		converter = CurrencyConverter()
		columns_info = [('total_contract_amount', 'contract_currency', 'total_contract_amount_eur'),
						('unit_price', 'contract_currency', 'unit_price_eur')]
		filtered_df = converter.convert_multiple_columns(
			df=filtered_df, columns_info=columns_info)
	except Exception as e:
		QMessageBox.warning(parent_widget, 'Ошибка конвертации', f"Ошибка при конвертации валют: {str(e)}")
		return
	# Константы
	MIN_TRASHOLD_PERCENT = 2 #2% минимум для отображения

	# 1. Анализ по дисциплинам в EUR
	discipline_analysis = filtered_df.groupby(['year_month', 'discipline'])['total_contract_amount_eur'].sum().unstack(fill_value=0)
	discipline_totals = filtered_df.groupby('discipline')['total_contract_amount_eur'].sum()
	total_sum = discipline_totals.sum()

	# показать отдельно только значимые дисциплины
	significant_disciplines = discipline_totals[discipline_totals / total_sum * 100 >= MIN_TRASHOLD_PERCENT]
	other_sum = discipline_totals.sum() - significant_disciplines.sum()

	# Создаем Series для круговой диаграммы (топ-5 + "Другие")
	pie_data = significant_disciplines.copy()
	if other_sum > 0:
		pie_data['Другие'] = other_sum

	# 3. Общие затраты по месяцам в EUR
	monthly_totals = filtered_df.groupby('year_month')['total_contract_amount_eur'].sum()

	# Преобразуем  Series в DataFrame для Z-Score
	monthly_series_df = monthly_totals.rename('total_contract_amount_eur').reset_index()

	# Z-Score (Поиск месяцев, которые сильно отклоняются от CРЕДНЕГО)
	monthly_series_df['Z_Score'] = stats.zscore(monthly_series_df['total_contract_amount_eur'])
	outliers_zscore = monthly_series_df[abs(monthly_series_df['Z_Score']) >= 2] # Порог Z >= 2

	# 2. IQR (Поиск месяцев, которые сильно отклоняются от МЕДИАНЫ)
	Q1 = monthly_series_df['total_contract_amount_eur'].quantile(0.25)
	Q3 = monthly_series_df['total_contract_amount_eur'].quantile(0.75)
	IQR = Q3 - Q1
	# Порог: 1.5 * IQR
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	outliers_iqr = monthly_series_df[
		(monthly_series_df['total_contract_amount_eur'] < lower_bound) |
		(monthly_series_df['total_contract_amount_eur'] > upper_bound)
		].sort_values(by='total_contract_amount_eur', ascending=False)

	if len(outliers_iqr) > 0:
		# определяем месяцы для удаления выбросов (динамически)
		# top_outlier_months_str =outliers_iqr['year_month'].head(2).astype(str).tolist()
		# определяем месяцы для удаления выбросов (все из списка outliers_iqr
		top_outlier_months_str =outliers_iqr['year_month'].astype(str).tolist()

		# 2. Преобразуем список строк в PeriodIndex
		dynamic_outlier_months = pd.PeriodIndex(top_outlier_months_str, freq='M')

		# 3. Создаем очищенную версию данных
		cleaned_monthly_totals = monthly_totals.drop(dynamic_outlier_months, errors='ignore')
	else:
		# Если выбросов нет, работаем со всеми данными
		cleaned_monthly_totals = monthly_totals.copy()

	# Переведем cleaned_monthly_totals в датафрейм, а потом в CSV файл
	df_clean = cleaned_monthly_totals.rename('y').reset_index()
	df_clean.columns = ['ds', 'y']

	# Статистические метрики
	monthly_stats = {
		"Среднемесячные затраты": monthly_totals.mean(),
		"Медиана": monthly_totals.median(),
		"Стандартное отклонение": monthly_totals.std(),
		"Коэффициент вариации": monthly_totals.std() / monthly_totals.mean(),
		"Минимум": monthly_totals.min(),
		"Максимум": monthly_totals.max(),
	}

	# 1. Коэффициент вариации по дисциплинам
	cv_by_discipline = filtered_df.groupby('discipline')['total_contract_amount_eur'].agg(
		cv=lambda x: x.std() / x.mean() if x.mean() != 0 else 0
	).sort_values(by='cv', ascending=False)
	cv_by_discipline.columns = ['Коэффициент вариации']

	# Функция форматирования валют
	def format_currency(x, p):
		if x >= 1e6:
			return f'{x/1e6:.1f}M €'
		elif x >= 1e3:
			return f'{x/1e3:.0f}K €'
		else:
			return f'{x:.0f} €'

	# Визуализация

	# 1. График: Общие затраты по месяцам (EUR) с трендом
	# =======================================================
	plt.figure(figsize=(10, 6)) # Создаем новую фигуру

	# Теперь используем dynamic_outlier_months для подписи графика
	outliers_removed_list = [str(m) for m in dynamic_outlier_months] if 'dynamic_outlier_months' in locals() else []
	title_suffix = f" (Без Outliers: {', '.join(outliers_removed_list)})" if outliers_removed_list else ""

	# Рисуем бары очищенных данных
	cleaned_monthly_totals.plot(kind='bar', color='skyblue', ax=plt.gca(), label='Месячные затраты (Чистые)')

	# Добавление 3-х месячного Скользящего среднего (Moving Average) ---
	window_size = 3 # Окно в 3 месяца (можно попробовать 4 или 6)
	ma_line = cleaned_monthly_totals.rolling(window=window_size, center=False).mean()
	ma_line.plot(kind='line', color='darkgreen', linewidth=3, label=f'{window_size}-мес. Скользящее среднее', ax=plt.gca())
	# -------------------------------------------------------------------------------

	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.title('1. Общие месячные затраты (EUR), Тренд и Скользящее среднее')
	plt.ylabel('Сумма, EUR')
	plt.xticks(rotation=45)

	# Добавляем линию тренда (уже есть, но убедимся, что она ниже ma_line)
	x_numeric = range(len(cleaned_monthly_totals))
	slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, cleaned_monthly_totals.values)
	trend_line = slope * np.array(x_numeric) + intercept
	plt.plot(x_numeric, trend_line, "r--", alpha=0.7, label=f"Линейный Тренд (R²={r_value**2:.3f})")

	plt.legend(loc='upper left') # Поместите легенду в более удобное место
	plt.grid(axis='y', linestyle='--') # Добавим сетку для читаемости
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'1_monthly_totals_trend_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 2. График: Топ-5 дисциплин по затратам (Stacked Bar)
	# =======================================================
	plt.figure(figsize=(12, 7)) # Создаем новую фигуру
	# Используем только значимые дисциплины
	discipline_analysis[significant_disciplines.index].plot(kind='bar', stacked=True, ax=plt.gca())
	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.title('2. Топ-дисциплины по месячным затратам (EUR)')
	plt.ylabel('Сумма, EUR')
	plt.xticks(rotation=45)
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'2_discipline_stacked_bar_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 3. График: Доля дисциплин (круговая диаграмма)
	# =======================================================
	plt.figure(figsize=(9, 9)) # Круговая диаграмма лучше смотрится в квадрате
	colors = plt.cm.tab20.colors
	pie_data.plot(
		kind='pie',
		autopct=lambda p: f'{p:.1f}%\n({p * pie_data.sum() / 100:,.0f} EUR)',
		colors=colors[:len(pie_data)],
		startangle=90,
		counterclock=False,
		ax=plt.gca()
	)
	plt.title('3. Распределение общих затрат (Топ-дисциплины + Другие)')
	plt.ylabel('')
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'3_discipline_pie_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 4. График: Нормализованная динамика по дисциплинам
	# =======================================================
	plt.figure(figsize=(12, 7)) # Создаем новую фигуру
	normalized = discipline_analysis.apply(lambda x: (x / x.max()) * 100)
	for discipline in significant_disciplines.index:
		plt.plot(normalized.index.astype(str), normalized[discipline], marker='o', label=discipline)
	plt.title('4. Нормализованная динамика по дисциплинам (100% = макс. для дисциплины)')
	plt.ylabel('Процент от максимального значения')
	plt.xticks(rotation=45)
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'4_discipline_normalized_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 5. График: Box Plot для анализа выбросов (Новый график)
	# =======================================================
	plt.figure(figsize=(8, 5)) # Создаем новую фигуру
	plt.boxplot(monthly_totals.values, vert=False)
	plt.title('5. Box Plot месячных затрат (Визуализация выбросов)')
	plt.yticks([1], ['Всего (EUR)'])
	plt.gca().xaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'5_monthly_boxplot_{timestamp}.png'), dpi=300)
	plt.close()


	# Экспорт данных в Excel
	with pd.ExcelWriter(os.path.join(OUT_DIR, f'cost_analysis_eur_{timestamp}.xlsx'), engine='openpyxl') as writer:

		# Экспортируем результаты анализа выбросов
		outliers_zscore.to_excel(writer, sheet_name='Выбросы (Z-Score)', index=False)
		outliers_iqr.to_excel(writer, sheet_name='Выбросы (IQR)', index=False)
		# --------------------------------------------------------

		# Сводная таблица по дисциплинам
		pivot_eur = filtered_df.pivot_table(
			index='year_month',
			columns='discipline',
			values='total_contract_amount_eur',
			aggfunc='sum',
			fill_value=0
		)
		pivot_eur.to_excel(writer, sheet_name='По дисциплинам (EUR)')

		# Итоговая статистика
		summary = filtered_df.groupby('discipline').agg({
			'total_contract_amount_eur': ['sum', 'mean', 'count'],
			'total_contract_amount': 'sum'
		})
		summary.columns = ['Сумма (EUR)', 'Среднее (EUR)', 'Кол-во закупок', 'Сумма (ориг валюта)']
		summary['Доля, %'] = (summary['Сумма (EUR)'] / summary['Сумма (EUR)'].sum()) * 100
		summary.to_excel(writer, sheet_name='Итоги')

		# 2. Общая статистика (по месяцам)
		monthly_stats_df = pd.DataFrame.from_dict(monthly_stats, orient='index', columns=['Значение (EUR)'])
		# Добавим Коэффициент вариации из monthly_stats и переформатируем
		if 'Коэффициент вариации' in monthly_stats_df.index:
			monthly_stats_df.loc['Коэффициент вариации', 'Значение (EUR)'] *= 100

		monthly_stats_df.to_excel(writer, sheet_name='Общая статистика')

		# 3. CV по дисциплинам
		cv_by_discipline.to_excel(writer, sheet_name='CV по дисциплинам')

		# Объединяем summary и cv_by_discipline
		cost_control_summary = summary[['Доля, %']].merge(
			cv_by_discipline,
			left_index=True,
			right_index=True
		).sort_values(by='Доля, %', ascending=False)

		cost_control_summary.to_excel(writer, sheet_name='Сводка_Контроль_Затрат')

		from models_analyses.sarima_forecast import run_sarima_forecast

		# =======================================================
		# 6. Прогнозирование (SARIMA) для самой стабильной/значимой дисциплины
		# =======================================================

		# Константа: порог CV для выбора дисциплины (Чем ниже, тем стабильнее)
		SIGNIFICANT_CV_THRESHOLD = 20.0

		# 1. Определение целевой дисциплины для прогноза
		# Критерий: Высокая доля (> MIN_TRASHOLD_PERCENT) И низкая волатильность (CV < 20.0).
		# Берем ту, что с наибольшей Долей, % среди кандидатов.
		forecasting_candidates = cost_control_summary[
			(cost_control_summary['Доля, %'] >= MIN_TRASHOLD_PERCENT) &
			(cost_control_summary['Коэффициент вариации'] < SIGNIFICANT_CV_THRESHOLD)
			].sort_values(by='Доля, %', ascending=False)

		if not forecasting_candidates.empty:
			TARGET_CATEGORY = forecasting_candidates.index[0]

			# 2. Уведомление и запуск прогноза
			QMessageBox.information(
				parent_widget,
				"Прогноз SARIMA",
				f"Выбрана дисциплина для прогноза: {TARGET_CATEGORY} (Доля: {forecasting_candidates['Доля, %'].iloc[0]:.1f}%, CV: {forecasting_candidates['Коэффициент вариации'].iloc[0]:.1f}).\n"
				"Запуск модели SARIMA..."
			)

			try:
				# 3. Вызов функции прогнозирования
				# Используем pivot_eur (созданный в п. "Сводная таблица по дисциплинам")
				forecast_result, plot_path = run_sarima_forecast(
					df_pivot_eur=pivot_eur,
					category=TARGET_CATEGORY,
					OUT_DIR=OUT_DIR,
					timestamp=timestamp,
					forecast_months=12
				)

				if not forecast_result.empty:
					# 4. Экспорт прогноза в Excel (Добавление нового листа в существующий файл)
					# Внимание: для добавления листа нужен отдельный блок записи, т.к. файл уже закрыт.
					try:
						with pd.ExcelWriter(os.path.join(OUT_DIR, f'cost_analysis_eur_{timestamp}.xlsx'), mode='a',
											engine='openpyxl', if_sheet_exists='replace') as writer:
							forecast_df_export = forecast_result.reset_index().rename(columns={'index': 'Дата'})
							forecast_df_export.to_excel(writer, sheet_name=f'Прогноз {TARGET_CATEGORY}', index=False)

						QMessageBox.information(
							parent_widget,
							"Прогноз завершен",
							f"Прогноз SARIMA для {TARGET_CATEGORY} (12 мес.) завершен.\n"
							f"Результаты добавлены в Excel-файл и сохранены как:\n{plot_path}"
						)
					except Exception as e:
						QMessageBox.warning(parent_widget, "Ошибка Экспорта Прогноза",
											f"Не удалось добавить лист прогноза в Excel: {str(e)}")

			except Exception as e:
				QMessageBox.warning(parent_widget, "Ошибка Прогноза",
									f"Не удалось выполнить прогноз SARIMA для {TARGET_CATEGORY}: {str(e)}")

		else:
			QMessageBox.warning(
				parent_widget,
				"Прогноз SARIMA",
				f"Не найдена дисциплина с долей >= {MIN_TRASHOLD_PERCENT}% и CV < {SIGNIFICANT_CV_THRESHOLD} для надежного прогноза."
			)


	QMessageBox.information(
		parent_widget,
		"Анализ завершен",
		f"Анализ месячных затрат в EUR сохранен в:\n{OUT_DIR}"
	)




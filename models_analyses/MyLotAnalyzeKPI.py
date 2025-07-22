import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.functions import CurrencyConverter

class LotAnalyzeKPI:
	def __init__(self, df):
		"""
		Инициализация с DataFrame, содержащим данные по лотам.
		"""
		self.df = df
		self.unique_disciplines = None
		if self.unique_disciplines is None:
			self.unique_discip()
		self.weights = {
			'total_lots': 0.5,
			'avg_time_to_close': 0.3,
			'avg_lot_value': 0.1,
			'sum_lot_value': 0.1
		}
	
	def unique_discip(self):
		"""
		Возвращает количество всех лотов, обрабатываемых исполнителями.
		"""
		self.unique_disciplines = self.df['discipline'].unique()
		return self.unique_disciplines
	
	def lots_per_actor(self):
		"""
		Возвращает количество лотов, обработанных каждым исполнителем внутри своей Дисциплины.
		"""
		# 1. Создаем словарь, который будет хранить информацию - Дисциплина - исполнитель- его количество Лотов
		lots_per_actor = {}
		
		# 2. Проходим по каждой дисциплине
		for discipline in self.unique_disciplines:
			# Фильтруем DataFrame по текущей дисциплине
			df_filtered = self.df[self.df['discipline'] == discipline].copy()
			
			if not df_filtered.empty:
				# Группировка по 'actor_name', подсчет лотов для каждого исполнителя
				lots_count = df_filtered.groupby('actor_name')['lot_number'].count().to_dict()
				
				# Добавляем информацию в словарь, где ключ — дисциплина, а значение — словарь с количеством лотов по исполнителям
				lots_per_actor[discipline] = lots_count
			else:
				lots_per_actor[discipline] = {}
		
		# Теперь lots_per_actor содержит данные для визуализации
		return lots_per_actor
	
	def avg_time_to_close(self):
		"""
		Рассчитывает среднее время обработки лота для каждого исполнителя.
		"""
		avg_time_per_discipline = {}  # Инициализация словаря для сохранения данных
		
		for discipline in self.unique_disciplines:
			# Фильтруем по дисциплине
			df_filtered = self.df[self.df['discipline'] == discipline].copy()
			
			# Проверяем, что фильтр не вернул пустой DataFrame
			if not df_filtered.empty:
				# Вычисляем время обработки
				df_filtered.loc[:, 'processing_time'] = (pd.to_datetime(df_filtered.loc[:, 'close_date']) -
				                                         pd.to_datetime(df_filtered.loc[:, 'open_date']))
				
				# Группируем по исполнителю и вычисляем среднее время обработки
				avg_processing_time = df_filtered.groupby('actor_name')['processing_time'].mean()
				# Сохраняем результаты в словарь
				avg_time_per_discipline[discipline] = avg_processing_time
			else:
				avg_time_per_discipline[discipline] = {}
		return avg_time_per_discipline
	
	
	# Здесь методы расчета итоговой средневзвешенной оценки KPI каждого сотрудника
	def normalize_kpi_table(self, df_kpi):
		"""
		Нормализует значения в таблице KPI по каждому столбцу и возвращает нормализованную таблицу.
		"""
		
		# Выбираем столбцы, которые нужно нормализовать
		columns_to_normalize = ['total_lots', 'avg_time_to_close', 'avg_lot_value', 'sum_lot_value']
		
		# Преобразуем Timedelta в дни для нормализации
		if 'avg_time_to_close' in df_kpi.columns:
			df_kpi['avg_time_to_close'] = df_kpi['avg_time_to_close'].apply(
				lambda x: x if isinstance(x, (int, float)) else x.days)
		
		# Функция для нормализации значений по дисциплинам
		def normalize_by_discipline(df, columns):
			normalized_data = []
			for discipline, group in df.groupby("discipline"):
				scaler = MinMaxScaler()
				group[columns] = scaler.fit_transform(group[columns])  # Нормализуем все указанные столбцы
				normalized_data.append(group)
			return pd.concat(normalized_data)
		
		# Нормализуем указанные столбцы
		df_kpi = normalize_by_discipline(df_kpi, columns_to_normalize)
		
		# Вычисляем итоговый KPI как сумму нормализованных значений
		df_kpi['kpi_score'] = df_kpi[columns_to_normalize].sum(axis=1)
		
		return df_kpi

	
	def get_seed_data(self, df_kpi):
		"""
		Выбирает лучших исполнителей (Seed Data) на основе максимального KPI по каждой дисциплине.
		"""
		# Группируем данные по дисциплинам и выбираем исполнителей с максимальным KPI
		seed_data = df_kpi.loc[df_kpi.groupby('discipline')['kpi_score'].idxmax()]
		print("Это seed_data")
		print(seed_data.columns)
		print(seed_data)
		print('Количество уникальных дисциплин в seed_data')
		print(seed_data['discipline'].nunique())  # Количество уникальных дисциплин в seed_data
		return seed_data
	
	
	def calculate_kpi(self, df):
		"""
		Рассчитывает итоговый KPI для каждого исполнителя по дисциплинам с использованием pandas.
		"""
		# 1. Пересчет стоимости лотов в EUR
		converter = CurrencyConverter()
		columns_info = [('total_price', 'currency', 'total_price_eur')]
		df_converted = converter.convert_multiple_columns(df=df, columns_info=columns_info)
		
		df['total_price_eur'] = df_converted['total_price_eur'].copy()
		
		# 2. Группировка данных по дисциплине и исполнителю
		grouped = df.groupby(['discipline', 'actor_name'])
		
		# Преобразуем даты в фориат datetime
		df['open_date'] = pd.to_datetime(df['open_date'])
		df['close_date'] = pd.to_datetime(df['close_date'])
		
		# Рассчитаем время закрытия  Лотов в днях
		df['time_to_close'] = (df['close_date'] - df['open_date']).dt.days
		
		# 3. Агрегация данных для расчета KPI
		kpi_data = grouped.agg(
			total_lots=('lot_number', 'count'),
			sum_lot_value=('total_price_eur', 'sum'),
			avg_lot_value=('total_price_eur', 'mean'),
			avg_time_to_close=('time_to_close', 'mean'),
			close_date=('close_date', 'first')  # Берем первую дату закрытия для группы
		).reset_index()
		
		# 4. Расчет KPI
		kpi_data['kpi_score'] = (
				self.weights['total_lots'] * kpi_data['total_lots'] +
				self.weights['avg_lot_value'] * kpi_data['avg_lot_value'] +
				self.weights['sum_lot_value'] * kpi_data['sum_lot_value'] -
				self.weights['avg_time_to_close'] * kpi_data['avg_time_to_close']
		)
		
		# 6. Нормализация KPI
		df_kpi_normalized = self.normalize_kpi_table(kpi_data)
		print(df_kpi_normalized)
		max_kpi_score = df_kpi_normalized['kpi_score'].max()
		print(max_kpi_score)
		min_kpi_score = df_kpi_normalized['kpi_score'].min()
		print(min_kpi_score)
		
		return df_kpi_normalized
	
	# def calculate_kpi(self, df):
	# 	"""
	# 	Рассчитывает итоговый KPI для каждого исполнителя по дисциплинам.
	# 	"""
	# 	# Словари для хранения метрик
	# 	total_lots_dict = self.lots_per_actor()
	# 	avg_time_dict = self.avg_time_to_close()
	#
	# 	# Список для хранения данных по KPI
	# 	kpi_data = []
	#
	# 	# Проходим по дисциплинам и собираем данные
	# 	for discipline in self.df['discipline'].unique():
	# 		for actor_name, total_lots in total_lots_dict[discipline].items():
	# 			avg_value_usd = 0
	# 			sum_value_usd = 0
	# 			total_lot_count = 0  # Счетчик общего числа лотов для правильного расчета среднего
	#
	# 			# Фильтруем данные для текущего исполнителя и дисциплины
	# 			df_actor = self.df[(self.df['discipline'] == discipline) & (self.df['actor_name'] == actor_name)]
	#
	# 			# Перебираем все лоты исполнителя для перерасчета в USD
	# 			for _, row in df_actor.iterrows():
	# 				currency = row['currency']
	# 				total_price = row['total_price']
	# 				exchange_rate = self.exchange_rate.get(currency, 1.0)  # Получаем курс для лота
	# 				total_price_usd = total_price * exchange_rate  # Пересчитываем стоимость в USD
	#
	# 				sum_value_usd += total_price_usd
	# 				total_lot_count += 1  # Увеличиваем счетчик лотов
	#
	# 			# рассчитываем среднюю стоимость лота в USD
	# 			if total_lot_count > 0:
	# 				avg_value_usd = sum_value_usd / total_lot_count
	#
	# 			avg_time = avg_time_dict.get(discipline, {}).get(actor_name, 0)
	#
	# 			if isinstance(avg_time, pd.Timedelta):
	# 				avg_time_days = avg_time.days  # Преобразуем Timedelta в количество дней
	# 			elif isinstance(avg_time, (int, float)):
	# 				avg_time_days = avg_time  # Если это уже число (int, float)
	# 			else:
	# 				avg_time_days = 0  # В других случаях присваиваем 0
	#
	# 			# Расчет KPI для исполнителя в рамках дисциплины
	# 			kpi_score = (
	# 					self.weights['total_lots'] * total_lots +
	# 					self.weights['avg_lot_value'] * avg_value_usd +
	# 					self.weights['sum_lot_value'] * sum_value_usd -
	# 					self.weights['avg_time_to_close'] * avg_time_days
	# 			)
	#
	# 			# Добавляем дату 'close_date' в расчет KPI
	# 			close_date = row['close_date'] if 'close_date' in df_actor.columns else None
	#
	# 			kpi_data.append({
	# 				'actor_name': actor_name,
	# 				'discipline': discipline,  # Сохраняем дисциплину для каждой записи
	# 				'total_lots': total_lots,
	# 				'avg_time_to_close': avg_time_days,
	# 				'avg_lot_value': avg_value_usd,
	# 				'sum_lot_value': sum_value_usd,
	# 				'kpi_score': kpi_score,
	# 				'close_date': close_date  # Добавляем дату закрытия
	# 			})
	#
	# 		# Создаем DataFrame с результатами KPI
	# 		df_kpi = pd.DataFrame(kpi_data)
	#
	# 		# Нормализуем KPI оценки
	# 		df_kpi_normalized = self.normalize_kpi_table(df_kpi)
	# 		print(df_kpi_normalized)
	# 		max_kpi_score = df_kpi_normalized.max()
	# 		print(max_kpi_score)
	# 		min_kpi_score = df_kpi_normalized.min()
	# 		print(min_kpi_score)
	#
	# 	return df_kpi_normalized

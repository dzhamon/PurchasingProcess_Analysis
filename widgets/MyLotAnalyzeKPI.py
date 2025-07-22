import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
		
		self.exchange_rate = {
			"AED": 0.27,
			'CNY': 0.14,
			'EUR': 1.2,
			'GBP': 1.32,
			'KRW': 0.00076,
			'KZT': 0.0021,
			'RUB': 0.0113,
			'USD': 1,
			'UZS': 0.000078,
			'JPY': 0.0069,
			'SGD': 0.78
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
		for discipline in self.df['discipline'].unique():
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
		
		for discipline in self.df['discipline'].unique():
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
	
	def avg_total_price_per_actor(self):
		"""
		Рассчитывает среднюю стоимость лотов для каждого сотрудника.
		"""
		avg_price_per_discipline = {}
		
		for discipline in self.df['discipline'].unique():
			df_filtered = self.df[self.df['discipline'] == discipline].copy()
			
			# Применяем курс валют к столбцу currency через .map()
			df_filtered['exchange_rate'] = df_filtered['currency'].map(self.exchange_rate)
			
			# Проверка на отсутствующие курсы валют
			missing_currencies = df_filtered[df_filtered['exchange_rate'].isnull()]['currency'].unique()
			if len(missing_currencies) > 0:
				print(f"Внимание: отсутствуют курсы для валют: {missing_currencies}")
			
			# Убедимся, что у всех валют есть курс
			df_filtered['exchange_rate'] = df_filtered['exchange_rate'].fillna(1.0)
			
			# Теперь пересчитываем стоимость лотов в USD
			df_filtered['lot_value_usd'] = df_filtered['total_price'] * df_filtered['exchange_rate']
			
			# Рассчитываем среднюю стоимость лота для каждого исполнителя
			avg_price_per_actor = df_filtered.groupby('actor_name')['lot_value_usd'].mean().to_dict()
			
			# Сохраняем результаты по дисциплине
			avg_price_per_discipline[discipline] = avg_price_per_actor
		
		return avg_price_per_discipline
	
	def sum_total_price_per_actor(self):
		"""
		Рассчитывает суммарную стоимость лотов для каждого сотрудника.
		"""
		# Словарь для хранения результатов
		sum_price_per_discipline = {}
		
		# Проходим по каждой дисциплине
		for discipline in self.df['discipline'].unique():
			# Фильтруем по дисциплине
			df_filtered = self.df[self.df['discipline'] == discipline].copy()
			
			# Конвертируем стоимость лотов в USD
			df_filtered.loc[:, 'lot_value_usd'] = df_filtered.apply(
				lambda row: row['total_price'] * self.exchange_rate.get(row['currency'], 1.0), axis=1
			)
			
			# Рассчитываем среднюю стоимость лота для каждого исполнителя
			avg_price_per_actor = df_filtered.groupby('actor_name')['lot_value_usd'].sum().to_dict()
			
			# Сохраняем результаты по дисциплине
			sum_price_per_discipline[discipline] = avg_price_per_actor
		
		return sum_price_per_discipline
	
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
		
		# Инициализируем нормализатор MinMaxScaler
		scaler = MinMaxScaler()
		
		# Применяем нормализацию к указанным столбцам
		df_kpi[columns_to_normalize] = scaler.fit_transform(df_kpi[columns_to_normalize])  # Нормализация не работает
		
		# Вычисляем итоговый KPI как сумму нормализованных значений
		df_kpi['kpi_score'] = df_kpi[columns_to_normalize].sum(axis=1)
		return df_kpi
	
	def calculate_kpi(self, df):
		"""
		Рассчитывает итоговый KPI для каждого исполнителя по дисциплинам.
		"""
		# Словари для хранения метрик
		total_lots_dict = self.lots_per_actor()
		avg_time_dict = self.avg_time_to_close()
		
		# Список для хранения данных по KPI
		kpi_data = []
		
		# Проходим по дисциплинам и собираем данные
		for discipline in self.df['discipline'].unique():
			for actor_name, total_lots in total_lots_dict[discipline].items():
				avg_value_usd = 0
				sum_value_usd = 0
				total_lot_count = 0  # Счетчик общего числа лотов для правильного расчета среднего
				
				# Фильтруем данные для текущего исполнителя и дисциплины
				df_actor = self.df[(self.df['discipline'] == discipline) & (self.df['actor_name'] == actor_name)]
				
				# Перебираем все лоты исполнителя для перерасчета в USD
				for _, row in df_actor.iterrows():
					currency = row['currency']
					total_price = row['total_price']
					exchange_rate = self.exchange_rate.get(currency, 1.0)  # Получаем курс для лота
					total_price_usd = total_price * exchange_rate  # Пересчитываем стоимость в USD
					
					sum_value_usd += total_price_usd
					total_lot_count += 1  # Увеличиваем счетчик лотов
				
				# рассчитываем среднюю стоимость лота в USD
				if total_lot_count > 0:
					avg_value_usd = sum_value_usd / total_lot_count
				
				avg_time = avg_time_dict.get(discipline, {}).get(actor_name, 0)
				
				if isinstance(avg_time, pd.Timedelta):
					avg_time_days = avg_time.days  # Преобразуем Timedelta в количество дней
				elif isinstance(avg_time, (int, float)):
					avg_time_days = avg_time  # Если это уже число (int, float)
				else:
					avg_time_days = 0  # В других случаях присваиваем 0
				
				# Расчет KPI для исполнителя в рамках дисциплины
				kpi_score = (
						self.weights['total_lots'] * total_lots +
						self.weights['avg_lot_value'] * avg_value_usd +
						self.weights['sum_lot_value'] * sum_value_usd -
						self.weights['avg_time_to_close'] * avg_time_days
				)
				
				# Добавляем дату 'close_date' в расчет KPI
				close_date = row['close_date'] if 'close_date' in df_actor.columns else None
				
				kpi_data.append({
					'actor_name': actor_name,
					'discipline': discipline,  # Сохраняем дисциплину для каждой записи
					'total_lots': total_lots,
					'avg_time_to_close': avg_time_days,
					'avg_lot_value': avg_value_usd,
					'sum_lot_value': sum_value_usd,
					'kpi_score': kpi_score,
					'close_date': close_date  # Добавляем дату закрытия
				})
			
			# Создаем DataFrame с результатами KPI
			df_kpi = pd.DataFrame(kpi_data)
			
			# Нормализуем KPI оценки
			df_kpi_normalized = self.normalize_kpi_table(df_kpi)
			print(df_kpi_normalized)
			max_kpi_score = df_kpi_normalized.max()
			print(max_kpi_score)
			min_kpi_score = df_kpi_normalized.min()
			print(min_kpi_score)
			return df_kpi_normalized

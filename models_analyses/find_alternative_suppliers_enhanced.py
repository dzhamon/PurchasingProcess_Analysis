import pandas as pd
from utils.config import SQL_PATH
import sqlite3
from utils.functions import cleanDataDF, CurrencyConverter


def find_alternative_suppliers_enhanced(merged_df, hhi_data):
	"""
	   Функция поиска альтернативных поставщиков с учетом HHI и данных из data_kp.
	"""
	
	# 1. Предобработка data_kp_df
	data_kp_df = pd.DataFrame()
	data_kp_df = preprocess_data_kp()
	print('Работаем с датафреймом : ', data_kp_df.columns)
	
	results = []
	
	for discipline in merged_df['discipline'].unique():
		discipline_hhi = hhi_data[hhi_data['discipline'] == discipline]['hhi_index'].values[0]
		concentration_level = get_concentration_level(discipline_hhi) # определение уровня концентрации по величине HHI
		discipline_data = merged_df[merged_df['discipline'] == discipline]
		discipline_historical_data = data_kp_df[
			data_kp_df['discipline'] == discipline]
		print(f"Этап для {discipline} пройден")
		
		if concentration_level == "Высокая":
			alternatives = find_alternatives_all_products(
				discipline_data, discipline_historical_data
			)
		elif concentration_level == "Средняя":
			major_suppliers = get_major_suppliers(discipline_data) # идентификация наиболее значимых поставщиков
			alternatives = find_alternatives_major_suppliers(
				discipline_data, discipline_historical_data, major_suppliers
			) # здесь фокусируемся на поиске замен для найденных ключевых поставщиков
		else:  # Низкая
			file_name = f"Анализ_поставщ_{discipline}_low_concentr.xlsx"
			analysis_output = analyze_supplier_structure(discipline_data, discipline_historical_data)
			
			# создаем датафрейм из списка поставщиков
			suppliers_df = pd.DataFrame({'Поставщик': analysis_output['all_suppliers']})
			# создаем датафрейм из результатов анализа стабильности
			stability_df = pd.DataFrame(analysis_output['supplier_stability'])
			stability_df['years_with_company'] = stability_df['years_with_company'].apply(lambda x: f"{x:.2f}".replace('.', ','))
			# создаем датафрейм с основными метриками
			summary_df = pd.DataFrame({
				'Количество поставщиков': [analysis_output['num_suppliers']],
				'Средняя сумма контракта': [f"{analysis_output['avg_share']:.2f}"],
				'Стандартное отклонение цен': [f"{analysis_output['price_std']:2f}"]
			})
			
			# Записываем датафреймы в разные листы Excel файла
			with pd.ExcelWriter(file_name) as writer:
				suppliers_df.to_excel(writer, sheet_name='Все поставщики', index=False)
				stability_df.to_excel(writer, sheet_name='Стабильность поставщиков', index=False)
				summary_df.to_excel(writer, sheet_name='Основные метрики', index=False)
			
			print("Результаты анализа сохранены в файл 'анализ_поставщиков_низкая_концентрация.xlsx'")
		
	return


def preprocess_data_kp():
	data_kp_df = pd.DataFrame()
	try:
		# соединяемся сбазой данных
		db_path = SQL_PATH
		conn = sqlite3.connect(db_path)
		# Создаем строку с плейсхолдерами для каждого lot_number
		
		query = f"""
					SELECT lot_number, close_date, good_name, good_count, supplier_qty, winner_name, discipline,
							unit_price, total_price, currency  FROM data_kp
				"""
		
		data_kp_df = pd.read_sql_query(query, conn)
		conn.close()
		
		# Отладочные принты
		print("Типы данных после чтения из базы:")
		print(data_kp_df.dtypes)
		print("Пример первых 5 строк:")
		print(data_kp_df.head())
		
		data_kp_df = cleanDataDF(data_kp_df.copy())
		
		data_kp_df['lot_number'] = data_kp_df['lot_number'].astype(str)  # переводим номера лотов в строковый тип
		data_kp_df['lot_number'] = data_kp_df['lot_number'].astype(str)  # то же самое с базой Лотов
		
		# Переведем стоимости в единую валюту EUR
		
		columns_info = [
			('unit_price', 'currency', 'unit_price_eur'),
			('total_price', 'currency', 'total_price_eur')
		]
		converter = CurrencyConverter()
		
		# Конвертируем и сохраняем два столбца
		converted_df = converter.convert_multiple_columns(
			df=data_kp_df, columns_info=columns_info)
		
		data_kp_df['total_price_eur'] = converted_df['total_price_eur'].copy()
		data_kp_df['unit_price_eur'] = converted_df['unit_price_eur'].copy()
		
		# Предобработка good_name
		data_kp_df['good_name'] = data_kp_df['good_name'].str.lower().str.strip()
		data_kp_df['good_name'] = data_kp_df['good_name'].str.replace(r'[^\w\s]', '', regex=True)  # Удаление пунктуации
		
		# Приведеме close_date в datetime
		data_kp_df['close_date'] = pd.to_datetime(data_kp_df['close_date'])
		
		return data_kp_df
	
	except Exception as e:
		print(f"Ошибка при формировании ДатаФрейма : {e}")
		return pd.DataFrame()  # Возвращаем пустой DataFrame в случае ошибки


def get_concentration_level(hhi):
	"""
		Определение уровня концентрации по HHI
	"""
	if hhi < 1500:
		return "Низкая"
	elif 1500 <= hhi <= 2500:
		return "Средняя"
	else:
		return "Высокая"


def find_alternatives_all_products(discipline_data, discipline_historical_data):
	"""
	Поиск альтернатив для всех товаров в дисциплине.
	"""
	alternatives = []
	for product in discipline_data['product_name'].unique():
		current_data = discipline_data[discipline_data['product_name'] == product]
		historical_data = discipline_historical_data[discipline_historical_data['good_name'] == product]
		alternatives.append(compare_and_analyze(current_data, historical_data))
		print(alternatives)
	return alternatives


def find_alternatives_major_suppliers(discipline_data, discipline_historical_data, major_suppliers):
	"""
	Поиск альтернатив только для товаров, поставляемых major_suppliers.
	"""
	alternatives = []
	for product in discipline_data['product_name'].unique():
		if any(supplier in major_suppliers for supplier in
		       discipline_data[discipline_data['product_name'] == product]['counterparty_name'].unique()):
			current_data = discipline_data[discipline_data['product_name'] == product]
			historical_data = discipline_historical_data[discipline_historical_data['good_name'] == product]
			alternatives.append(compare_and_analyze(current_data, historical_data))
		print(alternatives)
	return alternatives


def analyze_supplier_structure(discipline_data, discipline_historical_data):
	"""
	Анализ структуры поставщиков (для низкой концентрации).
	"""
	all_suppliers = discipline_data['counterparty_name'].unique().tolist()
	num_suppliers = len(all_suppliers)
	total_contract_amount = discipline_data['total_contract_amount_eur'].sum()
	avg_share = total_contract_amount / num_suppliers if num_suppliers > 0 else 0
	price_std = discipline_data['unit_price_eur'].std()
	
	# Анализ стабильности
	supplier_stability = discipline_data.groupby('counterparty_name')['contract_signing_date'].min().reset_index()
	supplier_stability['years_with_company'] = (pd.to_datetime('now') - supplier_stability['contract_signing_date']).dt.days / 365
	# Сортировка по убыванию years_with_company
	supplier_stability = supplier_stability.sort_values(by='years_with_company', ascending=False).reset_index(drop=True)
	
	analysis_results = {
		"all_suppliers": all_suppliers,
		"num_suppliers": num_suppliers,
		"avg_share": avg_share,
		"price_std": price_std,
		"supplier_stability": supplier_stability.to_dict(orient='records'),
	}
	return analysis_results

def compare_and_analyze(current_data, historical_data):
	"""
	Сравнение текущих и исторических поставщиков и формирование рекомендаций.
	"""
	comparison_results = {}
	
	# 1. Сравнение цен
	current_avg_price = current_data['unit_price_eur'].mean()
	historical_avg_price = historical_data['unit_price_eur'].mean()
	price_comparison = {
		"current_avg": current_avg_price,
		"historical_avg": historical_avg_price,
		"diff_percent": (
				current_avg_price - historical_avg_price) / historical_avg_price * 100 if historical_avg_price else None,
	}
	comparison_results['price_comparison'] = price_comparison
	
	# 2. Анализ стабильности поставщиков
	current_suppliers = set(current_data['counterparty_name'])
	historical_suppliers = set(historical_data['winner_name'])
	new_suppliers = current_suppliers - historical_suppliers
	lost_suppliers = historical_suppliers - current_suppliers
	stable_suppliers = current_suppliers.intersection(historical_suppliers)
	comparison_results['supplier_dynamics'] = {
		"new": list(new_suppliers),
		"lost": list(lost_suppliers),
		"stable": list(stable_suppliers),
	}
	
	# 3. Формирование рекомендаций
	recommendations = []
	if price_comparison['diff_percent'] and price_comparison[
		'diff_percent'] > 10:  # Пример: если текущие цены выше на 10%
		expensive_current_suppliers = current_data[current_data['unit_price_eur'] > historical_avg_price][
			'counterparty_name'].unique()
		recommendations.append(
			f"Рассмотреть возможность замены поставщиков: {expensive_current_suppliers} (текущие цены выше исторических).")
	
	if lost_suppliers:
		recommendations.append(f"Проанализировать причины потери поставщиков: {list(lost_suppliers)}.")
	
	comparison_results['recommendations'] = recommendations
	
	return comparison_results

def get_major_suppliers(discipline_data):
	"""
	   Определение major поставщиков (доля >= 8% или входит в top-7)
	   """
	threshold = 0.08  # 8%
	top_n = 7
	major_suppliers = set()
	
	# Группируем по поставщику и суммируем сумму контрактов
	supplier_amounts = discipline_data.groupby('counterparty_name')['total_contract_amount_eur'].sum()
	
	# Рассчитываем общую сумму контрактов по дисциплине
	total_discipline_amount = supplier_amounts.sum()
	
	if total_discipline_amount == 0:
		return []  # Избегаем деления на ноль
	
	# Рассчитываем долю рынка каждого поставщика
	market_shares = supplier_amounts / total_discipline_amount
	
	# Создаем DataFrame с долями рынка
	market_share_df = pd.DataFrame({'market_share': market_shares})
	
	# Определяем major поставщиков по порогу
	major_by_threshold = market_share_df[market_share_df['market_share'] >= threshold].index.tolist()
	
	# Определяем top-N поставщиков
	top_n_suppliers = market_share_df.nlargest(top_n, 'market_share').index.tolist()
	
	# Объединяем и удаляем дубликаты
	major_suppliers = list(set(major_by_threshold + top_n_suppliers))
	
	return major_suppliers
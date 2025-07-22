import pandas as pd
import sqlite3
"""
	Перед этим есть ошибки в загрузке параметров металлов в таблицу metal_products:
	1. По Трубам - есть вообще без размеров - что делать с такими.
	2. По листам - вылавливать в процессе дебаггинга
"""

def get_metal_dimensions_from_db(metal_product):
	""" Получает параметры металлопроката из базы данных. """
	DB_PATH = 'D:\Project_DataFiles\data\sql_krd_new.db'
	conn = sqlite3.connect(DB_PATH)
	
	query = "SELECT * FROM metal_products"
	try:
		df = pd.read_sql_query(query, conn)
		# Приводим столбец 'metal_product' к нижнему регистру для удобства поиска
		df['metal_product_lower'] = df['metal_product'].str.lower().str.strip()
		
		# Закрываем соединение
		conn.close()
		
		# Приводим искомый продукт к нижнему регистру
		metal_product_lower = metal_product.lower().strip()
		
		df = df.drop_duplicates(subset=['metal_product_lower'])
		
		# Создаем словарь: {название_металла (в нижнем регистре): вся строка DataFrame}
		metal_dict = df.set_index('metal_product_lower').to_dict(orient='index')
		
		# Ищем в словаре
		if metal_product_lower in metal_dict:
			return pd.Series(metal_dict[metal_product_lower])  # Возвращаем строку как Series
		
		print(f"⚠ Металлопрокат '{metal_product}' не найден в базе данных!")
		return None
	
	except Exception as e:
		print(f"Ошибка при выполнении SQL-запроса: {e}")
		return None


# Словарь с весами швеллеров
channel = {
	"Швеллер 5": 4.84, "Швеллер 6.5": 5.9, "Швеллер 8": 7.05, "Швеллер 10": 8.59,
	"Швеллер 12": 10.4, "Швеллер 14": 12.3, "Швеллер 16": 14.2, "Швеллер 18": 16.3,
	"Швеллер 20": 18.4, "Швеллер 22": 21.0, "Швеллер 24": 24.0, "Швеллер 27": 27.7,
	"Швеллер 30": 31.8, "Швеллер 40": 48.3
}


def weight_calculator(row):
	""" Рассчитывает массу 1 метра металлопроката на основе данных из базы. """
	print("Запускается метод Калькулятор веса металлов")
	
	# Извлекаем название металлопроката
	metal_product = row.get('metal_product', '').strip().lower()
	good_name = row.get('good_name', '').strip().lower()
	
	# Если metal_product пуст, пробуем использовать good_name
	if not metal_product and good_name:
		metal_product = good_name
	
	unit = row.get('supplier_unit', '').lower()
	
	# Проверка на швеллеры
	mark_number = row.get('mark_number', '').strip()
	if mark_number and f"Швеллер {mark_number}" in channel:
		return channel[f"Швеллер {mark_number}"] / 1000  # Перевод в тонны
	
	# Получаем параметры металлопроката из базы
	metal_data = get_metal_dimensions_from_db(metal_product)
	if metal_data is None:
		print(f"⚠ Металлопрокат '{metal_product}' не найден в базе данных!")
		return None
	
	density = 7850  # По умолчанию сталь (кг/м³)
	volume = 0
	
	# Определение типа металлопроката по названию
	if 'лист' in metal_product or 'лист' in good_name:
		width = metal_data['width'] / 1000
		thickness = metal_data['thickness'] / 1000
		volume = width * thickness * 1  # 1 метр длины
	elif 'труба' in metal_product or 'труба' in good_name:
		diameter = metal_data['diameter'] / 1000
		thickness = metal_data['thickness'] / 1000
		inner_d = diameter - 2 * thickness
		volume = (3.1415 * (diameter ** 2 - inner_d ** 2) / 4) * 1
	elif 'уголок' in metal_product or 'уголок' in good_name:
		side_size = metal_data['side_size'] / 1000
		width = metal_data['width'] / 1000
		thickness = metal_data['thickness'] / 1000
		volume = ((side_size * thickness) + (width * thickness) - (thickness ** 2)) * 1
	elif 'полоса' in metal_product or 'полоса' in good_name:
		width = metal_data['width'] / 1000
		thickness = metal_data['thickness'] /1000
		volume = width * thickness * 1
	
	# Рассчитываем массу и переводим в тонны
	mass_kg = volume * density
	return mass_kg / 1000  # Перевод в тонны


def convert_price_to_ton(price_per_unit, row):
	""" Конвертирует цену за метр в цену за тонну, если единица измерения - метры. """
	unit = row.get('supplier_unit', '').lower()
	
	if unit == 'т':
		return price_per_unit
	elif unit == 'м':
		conversion_factor = weight_calculator(row)
		return price_per_unit / conversion_factor if conversion_factor else None
	
	return None

import pandas as pd
import numpy as np
from collections import Counter
from utils.config import SQL_PATH
from utils.functions import del_nan
from datetime import datetime
import sqlite3


# Этот метод не используется. Есть более круче от самого Python basic_df = basic_df.drop_duplicates(subset='lot_number')
# Нет, subset='lot_number' не круче!!! Он из множества одинакоых номеров лота, оставляет только один,
# не обращая внимания на равенство/неравенство значений в столбцах
def del_double_lot_number_rows(df):
	"""
	Обрабатывает DataFrame, удаляя дубликаты в каждой группе lot_number.
	Эффективная реализация с использованием pandas.
	"""
	
	def handle_group(group):
		if len(group) == 1:
			return group
		
		# Сравнение строк целиком
		unique_rows = group.drop_duplicates()
		return unique_rows
	
	return df.groupby('lot_number', group_keys=False).apply(handle_group)


def clean_data_from_xls(file):
	
	dict_names = {'Номер лота': 'lot_number',
	              'Статус лота': 'lot_status',
	              'Дисциплина': 'discipline',
	              'Наименование проекта': 'project_name',
	              'Дата открытия лота': 'open_date',
	              'Дата закрытия лота': 'close_date',
	              'Исполнитель МТО (Ф.И.О.)': 'actor_name',
	              'Наименование ТМЦ': 'good_name',
	              'Количество ТМЦ': 'good_count',
	              'Ед. изм. ТМЦ': 'unit',
	              'Кол-во поставщика': 'supplier_qty',
	              'Ед.изм. поставщика': 'supplier_unit',
	              'Присуждено контрагенту': 'winner_name',
	              'Цена': 'unit_price',
	              'Сумма контракта': 'total_price',
	              'Валюты контракта': 'currency'}
	
	excel_data_df = pd.read_excel(file)
	excel_data_df = excel_data_df.rename(columns=dict_names)
	
	# Преобразование столбцов дат
	date_columns = ['open_date', 'close_date']
	for col in date_columns:
		if excel_data_df[col].dtype == object:
			try:
				excel_data_df[col] = pd.to_datetime(excel_data_df[col], format='%d.%m.%Y', errors='coerce')
			except ValueError:
				print(f"WARNING: Не удалось преобразовать столбец '{col}' в дату. Проверьте формат.")
		# Форматируем в 'YYYY-MM-DD' только если преобразование в datetime было успешным
		if pd.api.types.is_datetime64_any_dtype(excel_data_df[col]):
			excel_data_df[col] = excel_data_df[col].dt.strftime('%Y-%m-%d')
		else:
			excel_data_df[col] = None  # Или другая обработка, если преобразование не удалось
	
	# удаляем дублирующиеся строки в датафрейме
	excel_data_df = del_double_lot_number_rows(excel_data_df)
	
	# заменим в числовых полях excel_data_df все отсутствующие данные (nan) на ноль (0)
	excel_data_df['good_count'] = excel_data_df['good_count'].replace(np.nan, 0)
	excel_data_df['total_price'] = excel_data_df['total_price'].replace(np.nan, 0)
	excel_data_df['supplier_qty'] = excel_data_df['supplier_qty'].replace(np.nan, 0)
	excel_data_df['unit_price'] = excel_data_df['unit_price'].replace(np.nan, 0)
	
	excel_data_df['actor_name'] = excel_data_df['actor_name'].apply(
		lambda x: x.partition(' (')[0] if pd.notna(x) and x != '' else x)
	
	return excel_data_df


def clean_contr_data_from_xls(file_path):
	# Создадим словарь наимеований столбцов
	dict_contract = {'Номер лота': 'lot_number',
	                 'Дата завершения лота': 'lot_end_date',
	                 'Номер контракта/договора по этому лоту': 'contract_number',
	                 'Дата заключения контракта/договора': 'contract_signing_date',
	                 'Наименование контракта/договора': 'contract_name',
	                 'Исполнитель ДАК': 'executor_dak',
	                 'Наименование контрагента-владельца контракта/договора': 'counterparty_name',
	                 'Наименование товара': 'product_name',
	                 'Ед.изм. поставщика': 'supplier_unit',
	                 'Кол-во': 'quantity',
	                 'Ед. изм.': 'unit',
	                 'Цена за единицу товара': 'unit_price',
	                 'Сумма товара': 'product_amount',
	                 'Доп. расходы': 'additional_expenses',
	                 'Общая сумма контракта по лоту': 'total_contract_amount',
	                 'Валюта контракта/договора': 'contract_currency',
	                 'Условия поставки товара': 'delivery_conditions',
	                 'Условия оплаты': 'payment_conditions',
	                 'Срок/количество дней поставки товара': 'delivery_time_days',
	                 'Дисциплина': 'discipline'}
	
	contract_df = pd.read_excel(file_path)
	contract_df = contract_df.rename(columns=dict_contract).copy()

	# 1. Удаление полностью пустых строк
	contract_df.dropna(how='all')
	# 2. Заполнение пропусков в критичных полях
	critical_columns = ['lot_number', 'product_name', 'executor_dak']
	for col in critical_columns:
		try:
			contract_df[col] = contract_df[col].astype(str).fillna('').str.replace('\n', ' ').str.strip()
		except AttributeError as e:
			print(f"Ошибка при обработке столбца '{col}': {e}")
			print(f"Тип данных столбца '{col}' перед преобразованием: {contract_df[col].dtype}")
	# 3. Стандартизация дат
	date_columns = ['lot_end_date', 'contract_signing_date']
	for col in date_columns:
		if contract_df[col].dtype == object:
			try:
				contract_df[col] = pd.to_datetime(contract_df[col], format='%d.%m.%Y', errors='coerce')
			except ValueError:
				print(f"WARNING: Не удалось преобразовать столбец '{col}' в дату. Проверьте формат.")
		# Форматируем в 'YYYY-MM-DD' только если преобразование в datetime было успешным
		if pd.api.types.is_datetime64_any_dtype(contract_df[col]):
			contract_df[col] = contract_df[col].dt.strftime('%Y-%m-%d')
		else:
			contract_df[col] = None  # Или другая обработка, если преобразование не удалось

	# 4. Очистка числовых полей (цена, количество, сумма)
	numeric_columns = ['quantity', 'unit_price', 'product_amount', 'additional_expenses']
	for col in numeric_columns:
		contract_df[col] = pd.to_numeric(contract_df[col], errors='coerce')
		contract_df[col] = contract_df[col].fillna(0)  # или другое значение по умолчанию

	# 5. Очистка текстовых полей (удаление лишних символов, переносов строк)
	text_columns = ['contract_name', 'counterparty_name']
	for col in text_columns:
		try:
			contract_df[col] = contract_df[col].astype(str).fillna('').str.replace('\n', ' ').str.strip()
		except AttributeError as e:
			print(f"Ошибка при обработке столбца '{col}': {e}")
			print(f"Тип данных столбца '{col}' перед преобразованием: {contract_df[col].dtype}")
	# 7. Удаление дубликатов (если строки полностью идентичны)
	contract_df = del_double_lot_number_rows(contract_df)

	# 8. Проверка аномалий (например, отрицательные цены)
	contract_df = contract_df[contract_df['unit_price'] >= 0]

	# у executor_dak оставляем имя, фамилия, отчество. Обрезаем телефоны
	contract_df['executor_dak'] = contract_df['executor_dak'].apply(
		lambda x: x.partition(' (')[0] if pd.notna(x) and x != '' else x)
	
	return contract_df

def upload_to_sql_df(df: pd.DataFrame, conn: sqlite3.Connection, table_name: str):
	"""
	    Загружает DataFrame в указанную таблицу SQLite.

	    Args:
	        df: DataFrame для загрузки.
	        conn: Соединение с базой данных SQLite.
	        table_name: Имя таблицы, в которую нужно загрузить данные.
	"""
	cursor =conn.cursor()
	
	# 1. Получаем имена столбцов из DataFrame
	columns = df.columns.tolist()
	columns_str = ', '.join(f'"{col}"' for col in columns)  # Форматируем имена столбцов для SQL
	placeholders = ', '.join(['?'] * len(columns))  # Создаем строку с плейсхолдерами

	# 2. Создаем строку INSERT INTO
	insert_query = f'INSERT INTO "{table_name}" ({columns_str}) VALUES ({placeholders})'
	
	# 3. Подготавливаем данные для вставки (список кортежей)
	data_to_insert = []
	for _, row in df.iterrows():
		values = []
		for col in columns:
			value =row[col]
			if pd.api.types.is_datetime64_any_dtype(df[col]):
				value = value.strftime('%Y-%m-%d') if pd.notna(value) else None
			values.append(value)
		data_to_insert.append(tuple(values))
		
	# 4. Выполняем вставку данных
	cursor.executemany(insert_query, data_to_insert)
	conn.commit()

# Функция определяет номер квартала по дате
def quarter_of_date(date_tmp):
	quarter = (date_tmp.month - 1) // 3 + 1
	quarter = 'Q' + str(quarter) + '_' + date_tmp[6:10]
	return quarter


# В этом модуле идет подготовка основных укрупненных данных

def connect_to_database(db_name):
	# Эта функция собирет в список (массив) только уникальные элементы
	# из датафрейма data_df вызываем все даты закрытия лотов
	# date_strings = get_unique_only(list(data_df['close_date']))
	# получаем начальную и конечную даты
	conn = sqlite3.connect(db_name)
	cur = conn.cursor()
	min_date = cur.execute('SELECT min(close_date) FROM data_kp').fetchone()
	max_date = cur.execute('SELECT max(close_date) FROM data_kp').fetchone()
	beg_end_date = [min_date, max_date]
	
	return beg_end_date


# Метод проверки наличия файлов уже загруженных в таблицу files_names
def isfilepresent():
	list_files = []
	# подключение к базе дынных
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	# выполнение запроса
	result = cur.execute("select nameoffiles from files_name;").fetchall()
	# получение всех результатов выборки в список
	list_files = [row[0] for row in result]
	return list_files


# добавление имени нового файла в таблицу files_name
def addfilename(file):
	# подключение к базе дынных
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	
	# выполнение добавления
	query = "INSERT INTO files_name(nameoffiles) VALUES(?)"
	cur.execute(query, (file,))
	# сохраняем результат и закрываем базу
	conn.commit()
	conn.close()
	return

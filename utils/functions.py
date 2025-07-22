import pandas as pd
import numpy as np
import sqlite3
from PyQt5.QtWidgets import QMessageBox
import json
from utils.config import SQL_PATH
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

_cached_data = None


def cleanDataDF(data_df):
	# Удаляем строки, где good_name, winner_name, discipline, currency являются None
	data_df = data_df.dropna(subset=['good_name', 'winner_name', 'discipline', 'currency'])
	# Удаляем строки, где unit_price или total_price равны 0.0
	data_df = data_df[data_df['unit_price'] != 0.0]
	data_df = data_df[data_df['total_price'] != 0.0]
	
	# Удаляем дубликаты
	data_df = data_df.drop_duplicates()
	
	# Определяем столбцы, которые являются числовыми
	numeric_cols = ['lot_number', 'good_count', 'supplier_qty', 'unit_price', 'total_price']
	for col in data_df.columns:
		if 'price' in col.lower() or 'number' in col.lower():
			if col not in numeric_cols:
				numeric_cols.append(col)
	numeric_cols = list(set(numeric_cols)) # удаляются дубликаты
	
	# итерируемся по числовым столбцам и очищаем, а затем преобразуем
	for col in numeric_cols:
		if col in data_df.columns:
	
			# Шаг 1: Попытка прямого преобразования
			data_df.loc[:, col] = pd.to_numeric(data_df[col], errors='coerce')
			
			# Шаг 2: Проверка на наличие NaN после преобразования (не удалось преобразовать)
			nan_mask = data_df[col].isnull()
			if nan_mask.any():
				# Шаг 3: Удаление строк с NaN в текущем числовом столбце
				data_df = data_df[~nan_mask]  # Сохраняем только строки, где нет NaN
				# здесь нужно поставить оператор изменения типа столбца на numeric
				data_df[col] = pd.to_numeric(data_df[col], errors='raise').astype('float64')
			else:
				print(f"Нечисловых значений, которые не удалось преобразовать в столбце '{col}', не обнаружено.")
		else:
			print(f" Предупреждение: Числовой столбец '{col}' не найден.")
	
	# Нормализация названий компаний
	data_df.loc[:, 'winner_name'] = data_df['winner_name'].replace({
		'Не использовать V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
		'[Удалено]СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
		'ТСК-РЕГИОН ООО': 'ООО ТСК-РЕГИОН',
		'ТСК РЕГИОН': 'ООО ТСК-РЕГИОН',
		'ТСК Регион ': 'ООО ТСК-РЕГИОН',
		'ООО "УПСК-экспорт"': 'УПСК-ЭКСПОРТ',
		'УПСК-ЭКСПОРТ ООО': 'УПСК-ЭКСПОРТ',
		'не использовать УПСК-экспорт ООО': 'УПСК-ЭКСПОРТ',
		'ООО «УПСК-экспорт»' : 'УПСК-ЭКСПОРТ',
		'ООО "Темир Бетон Конструкциялари Комбинати"': 'ООО "Темир Бетон',
		'не использовать DREAM-ALLIANCE': 'DREAM ALLIANCE',
		'не использовать NEW FORMAT TASHKENT': 'NEW FORMAT TASHKENT',
		'не использовать NASIBA GAVHAR': 'NASIRA GAVHAR',
		'Daromad Munira Fayz Textile<не использовать>': 'Daromad Munira Fayz',
		'СП ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ AZIA METALL PROF': 'СП OOO AZIA METAL PRPF',
		'ТОРГОВЫЙ ДОМ ОМСКИЙ ЗАВОД ТРУБОПРОВОДНОЙ АРМАТУРЫ ООО' : 'ООО ОМСКИЙ ЗАВОД ТРУБ АРМАТ',
		'ОМСКИЙ ЗАВОД ЗАПОРНОЙ АРМАТУРЫ ООО': 'ООО ОМСКИЙ ЗАВОД ЗАПОРН АРМАТ',
		'ООО «Омский завод запорной арматуры»': 'ООО ОМСКИЙ ЗАВОД ЗАПОРН АРМАТ'
	})
	
	return data_df

def clean_contract_data(df_c):
	# Заполнение пропусков в текстовых полях
	df_c.fillna({'lot_number': 'Не указано', 'discipline': 'Не указано', 'contract_name': 'Не указано',
	             'executor_dak': 'Не указано',
	             'counterparty_name': 'Не указано', 'product_name': 'Не указано', 'unit': 'Не указано',
	             'contract_currency': 'Не указано'}, inplace=True)
	
	# Удаляем пробелы (разделители тысяч) и заменяем запятую на точку (десятичный разделитель)
	df_c['total_contract_amount'] = pd.to_numeric(
		df_c['total_contract_amount'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False),
		errors='coerce')
	df_c['unit_price'] = pd.to_numeric(
		df_c['unit_price'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False),
		errors='coerce')
	# Преобразуем значения в числовой формат, несовместимые значения станут NaN
	df_c['unit_price'] = pd.to_numeric(df_c['unit_price'], errors='coerce')
	df_c.fillna(df_c['unit_price'].median(), inplace=True)  # Заполняем пропущенные значения медианой
	
	# Удаляем строки с NaN или нулевыми значениями в 'total_price' и 'unit_price'
	df_c = df_c.dropna(subset=['total_contract_amount', 'product_amount', 'unit_price', 'quantity'])
	df_c = df_c[df_c['total_contract_amount'] > 0]
	df_c = df_c[df_c['unit_price'] > 0]
	df_c = df_c[df_c['product_amount'] > 0]
	df_c = df_c[df_c['quantity'] > 0]
	# Преобразуем колонку contract_signing_date и lot_end_date в формат datetime, игнорируя ошибки
	df_c['contract_signing_date'] = pd.to_datetime(df_c['contract_signing_date'], format='%Y-%m-%d', errors='coerce')
	df_c['lot_end_date'] = pd.to_datetime(df_c['lot_end_date'], format='%Y-%m-%d', errors='coerce')
	
	# Замена разных написаний на единое название для компании
	df_c['counterparty_name'] = df_c['counterparty_name'].replace({
		'Не использовать V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"',
		'[Удалено]СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'СП "Ташкентский трубный завод"': 'СП "Ташкент трубный завод"',
		'V L GALPERIN NOMIDAGI TOSHKENT TRUBA ZAVODI СП ООО': 'СП "Ташкент трубный завод"'
	})
	return df_c

def get_cached_data():
	global _cached_data
	if _cached_data is not None:
		print("Данные получены из кэша")
		return _cached_data["invalid_dates_df"], _cached_data["contract_df"]
	else:
		return


def del_nan(list_name):
	L1 = [item for item in list_name if not (pd.isnull(item)) is True]
	L1, list_name = list_name, L1
	return list_name


def get_unique_only(st):
	# Empty list
	lst1 = []
	count = 0
	# traverse the array
	for i in st:
		if i != 0:
			if i not in lst1:
				count += 1
				lst1.append(i)
	return lst1


# Функция "обрезки" строки до нужного символа
def cut_list(lstt_act):
	last_act = []
	for lst_act in lstt_act:
		try:
			if pd.notna(lst_act) and lst_act != '':
				last_act.append(lst_act.partition(' (')[0])
			else:
				last_act.append(np.nan)  # добавление NaN для соответствия длине
		except AttributeError:
			last_act.append(np.nan)  # добавление NaN при ошибке
	return last_act


def calc_indicators(query):
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	res = cur.execute(query).fetchall()
	return res


def prepare_main_datas(sql_query=None):
	# Суммы и средние значения контрактов в разрезе Дисциплин и валют контрактов
	# материал по работе SQLite_Python заимствован из
	# https://sky.pro/wiki/sql/preobrazovanie-rezultatov-zaprosa-sqlite-v-slovar/
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	cur.execute(sql_query)
	columns = [column[0] for column in cur.description]
	values = cur.fetchall()
	row_dict = {}
	k = 0
	for column in columns:
		list_tmp = []
		for value in values:
			list_tmp.append(value[k])
		row_dict[column] = list_tmp
		k += 1
	df = pd.DataFrame(row_dict)
	return df


def create_treeview_table(df):
	columns = df.columns
	print('Our DF columns is ', columns)
	list_of_rows = []
	print('df.shape =', df.shape)
	for i in range(df.shape[0]):
		list_of_rows.append(df.T[i].tolist())
	param1 = columns
	param2 = list_of_rows
	return param1, param2


# функция параметризации запроса
def param_query(qry):
	conn = sqlite3.connect(SQL_PATH)
	cur = conn.cursor()
	cur.execute(qry)
	
	print(cur.fetchall())
	
	conn.close()


def trim_actor_name(name):
	"""
	Обрезает строку до первой открывающей скобки.
	Например, 'Алишеров Асадбек Абдулла угли (вн. 31902) (моб. +998936457575)' станет 'Алишеров Асадбек Абдулла угли'.
	"""
	return name.split('(')[0].strip()


class CurrencyConverter:
	"""
	Класс для конвертации валют в евро.
	"""
	
	def __init__(self, exchange_rates=None):
		"""
		Инициализация курсов валют.
		:param exchange_rates: Словарь с курсами валют к EUR.
		"""
		# Курсы валют по умолчанию
		self.exchange_rates = exchange_rates or {
			'AED': 0.23, 'CNY': 0.13, 'EUR': 1.0, 'GBP': 1.13,
			'KRW': 0.00077, 'KZT': 0.002, 'RUB': 0.011, 'USD': 0.83,
			'UZS': 0.000071, 'JPY': 0.0073, 'SGD': 0.61
		}
	
	def update_rates(self, new_rates):
		"""
		Обновляет курсы валют.
		:param new_rates: Словарь с новыми курсами валют.
		"""
		self.exchange_rates.update(new_rates)
	
	def convert_column(self, df, amount_column, currency_column, result_column=None):
		"""
		Конвертирует значения одного столбца суммы в EUR.
		:param df: DataFrame с данными.
		:param amount_column: Столбец с суммой.
		:param currency_column: Столбец с валютами.
		:param result_column: Столбец для сохранения результата (если None, перезаписывает amount_column).
		:return: Обновленный DataFrame.
		"""
		result_column = result_column or amount_column
		df = df.copy()
		
		# Добавляем столбец с курсами валют
		df['exchange_rate'] = df[currency_column].map(self.exchange_rates)
		df = df.dropna(subset=['exchange_rate'])  # Удаляем строки с неизвестными валютами
		
		# Конвертируем сумму
		df[result_column] = df[amount_column] * df['exchange_rate']
		return df.drop(columns=['exchange_rate'])
	
	def convert_multiple_columns(self, df, columns_info):
		"""
		Конвертирует несколько столбцов в EUR.
		:param df: DataFrame с данными.
		:param columns_info: Список кортежей [(amount_column, currency_column, result_column), ...].
		:return: Обновленный DataFrame.
		"""
		df = df.copy()
		for amount_column, currency_column, result_column in columns_info:
			df = self.convert_column(df, amount_column, currency_column, result_column)
		return df


def save_analysis_results(analysis_results, project_name, OUT_DIR):
	try:
		file_exls_name = f"Equil_sumResult_{project_name}.xlsx"
		file_exls_path = os.path.join(OUT_DIR, file_exls_name)
		
		analysis_results.to_excel(file_exls_path, index=False, engine='openpyxl')
		print("Файл успешно сохранен.")
	
	except PermissionError:
		msg = QMessageBox()
		msg.setIcon(QMessageBox.Warning)
		msg.setWindowTitle("Ошибка")
		msg.setText("Файл открыт. Закройте файл и попробуйте снова.")
		msg.exec_()
	return


# Функция для загрузки подсказок для JSON-файла
def load_menu_hints():
	with open('menu_hints.json', 'r', encoding='utf-8') as file:
		return json.load(file)


menu_hints = load_menu_hints()


# очистка имени файла от запрещенных символов
def clean_filename(filename):
	"""
	Очищает имя файла от недопустимых символов.
	Заменяет запрещённые символы на "_" и убирает лишние пробелы.
	"""
	# Заменяем недопустимые символы на "_"
	cleaned = re.sub(r'[\\/*?:"<>|()]', '_', filename)
	# Убираем лишние пробелы
	cleaned = re.sub(r'\s+', ' ', cleaned).strip()
	return cleaned


def plot_supplier_prices_by_currency(results_by_currency):
	"""
	   Построение графиков для каждого валютного анализа.
	   Args:
	       results_by_currency (dict): Результаты анализа, сгруппированные по валютам.
	"""
	print("Началось рисование графиков")
	for currency, data in results_by_currency.items():
		if data['top_suppliers'].empty or data['bottom_suppliers'].empty:
			print(f"Нет данных для валюты {currency}. Пропускаем ...")
			continue
		
		top_suppliers = data['top_suppliers']
		bottom_suppliers = data['bottom_suppliers']
		
		# График для топ-10 поставщиков с высокими ценами
		plt.figure(figsize=(10, 6))
		sns.barplot(
			data=top_suppliers,
			x='avg_unit_price',
			y='winner_name',
			hue='winner_name',
			dodge=False,
			palette='Reds_r',
			legend=False
		)
		
		plt.title(f"Топ-10 поставщиков с высокими ценами ({currency})")
		plt.xlabel("Средняя цена за единицу")
		plt.ylabel("Поставщик")
		plt.tight_layout()
		plt.show()
		
		# График для топ-10 поставщиков с низкими ценами
		plt.figure(figsize=(10, 6))
		sns.barplot(
			data=bottom_suppliers,
			x='avg_unit_price',
			y='winner_name',
			hue='winner_name',
			dodge=False,
			palette='Greens',
			legend=False
		)
		plt.title(f"Топ-10 поставщиков с низкими ценами ({currency})")
		plt.xlabel("Средняя цена за единицу")
		plt.ylabel("Поставщик")
		plt.tight_layout()
		plt.show()


def check_file_access(file_path):
	"""
	Проверяет доступ к файлу. Если файл открыт, выводит сообщение об ошибке.
	"""
	try:
		# Пробуем открыть файл для записи
		with open(file_path, 'a'):
			pass
		return True
	except IOError:
		# Если файл занят, выводим сообщение
		msg_box = QMessageBox()
		msg_box.setIcon(QMessageBox.Warning)
		msg_box.setWindowTitle("Ошибка доступа к файлу")
		msg_box.setText(f"Файл {file_path} открыт в другом приложении.\nПожалуйста, закройте его.")
		msg_box.setStandardButtons(QMessageBox.Ok)
		msg_box.exec_()
		return False


def load_contracts(parent_widget, min_date, max_date):
	# функция загрузки данных по Контрактам
	# получаем выбранные даты
	start_date = min_date  # self.start_date_edit.date().toPyDate()
	end_date = max_date  # self.end_date_edit.date().toPyDate()
	
	# Проверяем корректность диапазона дат
	if start_date > end_date:
		QMessageBox.warning(parent_widget, "Предупреждение",
		                    "Начальная дата позже конечной даты. Проверьте корректность значений")
		return
	
	# Загружаем данные из таблицы data_contract по выбранным датам
	db_path = SQL_PATH
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	
	# Проверяем, есть ли в таблице data_contract записи с такими датами
	cursor.execute("""
			       SELECT COUNT(*) FROM data_contract
			       WHERE DATE(contract_signing_date) BETWEEN DATE(?) AND DATE(?);
			   """, (start_date, end_date))
	
	count = cursor.fetchone()[0]
	
	if count == 0:
		QMessageBox.warning(parent_widget, "Ошибка",
		                    "В выбранном диапазоне нет данных.")
		conn.close()
		return
	
	# Если данные есть - загружаем их
	query = f"""
			SELECT * FROM data_contract
			WHERE DATE(contract_signing_date) BETWEEN DATE(?) AND DATE(?);
			"""
	# Загрузка данных в датафрейм
	contract_df = pd.read_sql_query(query, conn, params=(start_date, end_date))
	# закрыть соединение с базой данных
	conn.close()
	
	contract_df = clean_contract_data(contract_df)  # очистка данных полученного df
	
	return contract_df

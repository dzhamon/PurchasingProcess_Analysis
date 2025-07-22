# метод определения контрактов без соответствующих лотов
# метод определения контрактов, дата плписания которых меньше даты завершения лотов
import pandas as pd
# Убедимся, что _cached_data импортирован, если он определён в другом модуле
from utils.functions import get_cached_data
import os
import pandas as pd


def process_lots(main_window):
	data_df = main_window.get_data_df()


def check_contracts_less_dates(data_df):
	print('Мы в методе??')
	print(data_df[data_df['lot_number'] == 84640])
	# Получаем данные из кэша или загружаем их
	invalid_dates_df, contract_df = get_cached_data()
	
	# Подсчитываем количество вхождений для каждого уникального executor_dak
	executor_counts = invalid_dates_df['executor_dak'].value_counts().reset_index()
	
	# Переименовываем колонки для удобства
	executor_counts.columns = ['executor_dak', 'count']
	
	# Сортируем по количеству вхождений в порядке убывания
	executor_counts_sorted = executor_counts.sort_values(by='count', ascending=False)
	
	# Выводим отсортированный результат
	print(executor_counts_sorted)
	
	# Папка для сохранения файлов
	output_folder = "D:\Analytics\executor_contracts"
	os.makedirs(output_folder, exist_ok=True)
	"""------------ получение файлов с работой исполнителей контрактов-----"""
	""" Здесь нужно будет разместить код анализа данных  """
	
	# Получаем список исполнителей
	executors = executor_counts_sorted['executor_dak'].tolist()
	
	# Сохраняем контракты для каждого исполнителя из топ-10
	for executor in executors:
		# Фильтрация данных для текущего исполнителя
		executor_data = invalid_dates_df[invalid_dates_df['executor_dak'] == executor]
		print(executor_data['lot_number'].dtype)
		
		# Пропускаем, если данных нет
		if executor_data.empty:
			print(f"Нет данных для исполнителя {executor}, пропускаем.")
			continue
	
		# Получаем уникальные lot_number из executor_data
		unique_lots = executor_data['lot_number'].unique()
		
		# Идем по каждому уникальному лоту
		for lot in unique_lots:
			# Фильтруем данные для текущего lot_number
			data_df_lot = data_df[data_df['lot_number'] == lot]
			executor_data_lot = executor_data[executor_data['lot_number'] == lot]
			
			# Подсчитываем суммы total_price и total_contract_amount
			total_price_sum = data_df_lot['total_price'].sum()
			total_contract_sum = executor_data_lot['total_contract_amount'].sum()
			
			# Проверка разницы сумм с допуском 0.01
			print(f"Lot {lot}: total_price_sum={total_price_sum}, total_contract_sum={total_contract_sum}")
			if abs(total_price_sum - total_contract_sum) <= 0.01:
				print(f"Суммы совпадают для lot_number {lot}. Переход к следующему.")
				continue
			else:
				# Если суммы не совпадают, проверяем количество строк
				print(f"Количество строк в Лотах {len(data_df_lot)}, в Контрактах {len(executor_data_lot)}")
				if len(data_df_lot) == len(executor_data_lot):
					# Если количество строк совпадает, сравниваем построчно
					print(f"Количество строк совпало для lot_number {lot}. Начинаем построчное сравнение.")
					mismatch_found = False
					for (idx_kp, row_kp), (idx_exec, row_exec) in zip(data_df_lot.iterrows(),
					                                                  executor_data_lot.iterrows()):
						if row_kp['good_count'] != row_exec['quantity'] or row_kp['unit_price'] != row_exec[
							'unit_price']:
							mismatch_found = True
							print(f"Несоответствие для lot_number {lot} в строке {idx_kp}:")
							print(f"data_kp: good_count={row_kp['good_count']}, unit_price={row_kp['unit_price']}")
							print(
								f"executor_data: quantity={row_exec['quantity']}, unit_price={row_exec['unit_price']}")
							break
					if not mismatch_found:
						print(f"Количество строк совпадает для lot_number {lot}.")
				elif len(data_df_lot) < len(executor_data_lot):
					print(f"Количество строк в Контрактах увеличилось для lot_number {lot}")
				else:
					print(f"Количество строк в Контрактах уменьшилось для lot_number {lot}")
					continue


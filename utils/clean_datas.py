import sqlite3
from .config import SQL_PATH


def clean_database():
	# Функция для выполнения SQL-запроса
	def execute_query(connection, query):
		with connection:
			cursor = connection.cursor()
			cursor.execute(query)
	
	# Подключение к базе данных
	with sqlite3.connect(SQL_PATH) as conn:
		# Замена NaN на 0 в числовых столбцах
		update_queries = [
			"""
			UPDATE data_kp
			SET good_count = COALESCE(good_count, 0)
			WHERE good_count IS NULL;
			""",
			"""
			UPDATE data_contract
			SET quantity = COALESCE(quantity, 0)
			WHERE quantity IS NULL;
			"""
		]
		
		# Удаление строк с нулевыми значениями
		delete_queries = [
			"""
			DELETE FROM data_kp
			WHERE good_count = 0;
			""",
			"""
			DELETE FROM data_contract
			WHERE quantity = 0;
			"""
		]
		
		# Удаление дубликатов
		delete_duplicate_queries = [
			"""
			DELETE FROM data_kp
			WHERE rowid NOT IN (
				SELECT MIN(rowid)
				FROM data_kp
				GROUP BY lot_number, lot_status, discipline, project_name,
						 open_date, close_date, actor_name, good_name,
						 good_count, unit, supplier_qty, supplier_unit,
						 winner_name, unit_price, total_price, currency
			);
			""",
			"""
			DELETE FROM data_contract
			WHERE rowid NOT IN (
				SELECT MIN(rowid)
				FROM data_contract
				GROUP BY 'lot_number', 'lot_end_date', 'contract_number',
       'contract_signing_date', 'contract_name', 'executor_dak',
       'counterparty_name', 'product_name', 'supplier_unit', 'quantity',
       'unit', 'unit_price', 'product_amount', 'additional_expenses',
       'total_contract_amount', 'contract_currency', 'delivery_conditions',
       'payment_conditions', 'delivery_time_days', 'discipline'
			);
			"""
		]
		
		# Обрезка инициалов Исполнителя
		trim_actor_name_query = """
        UPDATE data_kp
        SET actor_name = TRIM(SUBSTR(actor_name, 1, INSTR(actor_name, '(') - 1))
        WHERE actor_name LIKE '%(%';
        """
		
		# Выполнение всех запросов
		for query in update_queries + delete_queries + delete_duplicate_queries + [trim_actor_name_query]:
			execute_query(conn, query)
			
	print('Таблицы почищены!')

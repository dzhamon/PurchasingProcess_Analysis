import sqlite3

# Путь к базе данных
db_path = r"D:\Project_DataFiles\data\sql_krd_new.db"

# Подключение к SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# SQL-запрос на создание таблицы
create_table_query = """
CREATE TABLE IF NOT EXISTS metal_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metal_product TEXT NOT NULL,
    diameter REAL,
    side_size REAL,
    width REAL,
    thicness REAL,
    mark_number INTEGER
);
"""

# Выполнение запроса
cursor.execute(create_table_query)

# Сохранение изменений и закрытие соединения
conn.commit()
conn.close()

print("Таблица metal_products успешно создана!")

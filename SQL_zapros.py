import sqlite3
import pandas as pd

# Подключение к базе данных
conn = sqlite3.connect("D:\Project_DataFiles\data\sql_krd_new.db")

# SQL-запрос
query = """
SELECT DISTINCT good_name AS metal_product
FROM data_kp
WHERE discipline IN ('Трубная продукция', 'МП и МК');
"""

# Читаем результат в DataFrame
df = pd.read_sql_query(query, conn)

# Сохраняем в CSV
df.to_csv("metal_products.csv", index=False, encoding="utf-8-sig")

# Закрываем соединение
conn.close()

print("Файл metal_products.csv успешно создан!")

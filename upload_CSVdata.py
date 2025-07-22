import sqlite3
import pandas as pd
import re

# Путь к базе данных
db_path = r"D:\Project_DataFiles\data\sql_krd_new.db"

# Путь к файлу CSV
csv_path = r"D:\My_PyQt5_Project_app\metal_products.csv"

# Подключаемся к базе данных
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Загружаем CSV
df = pd.read_csv(csv_path, encoding="utf-8-sig")


# Функция для разбора названия и извлечения параметров
def extract_metal_parameters(product_name):
	product_name_lower = str(product_name).lower()
	
	# Определяем тип изделия
	if "арматура" in product_name_lower:
		shape = "арматура"
	elif "балка" in product_name_lower:
		shape = "балка"
	elif "труба" in product_name_lower:
		shape = "труба"
	elif "уголок" in product_name_lower:
		shape = "уголок"
	elif "круг" in product_name_lower:
		shape = "круг"
	elif "лист" in product_name_lower:
		shape = "лист"
	elif "швеллер" in product_name_lower:
		shape = "швеллер"
	elif "двутавр" in product_name_lower:
		shape = "двутавр"
	elif "полоса" in product_name_lower:
		shape = "полоса"
	elif "квадрат" in product_name_lower:
		shape = "квадрат"
	elif "лента" in product_name_lower:
		shape = "лента"
	elif "профиль" in product_name_lower:
		shape = "профиль"
	else:
		shape = "другое"
	
	# Поиск размеров (формат: 21.3x2.77 или 33,4х3,38)
	size_match = re.search(r'(\d+[.,]?\d*)[xх](\d+[.,]?\d*)', product_name_lower)
	if size_match:
		outer_diameter = float(size_match.group(1).replace(",", "."))
		wall_thickness = float(size_match.group(2).replace(",", "."))
	else:
		outer_diameter, wall_thickness = None, None
	
	return shape, outer_diameter, wall_thickness


# Применяем функцию для всех строк
df[['shape', 'outer_diameter', 'wall_thickness']] = df['metal_product'].apply(
	lambda x: pd.Series(extract_metal_parameters(x))
)

# Удаляем пустые значения (если нужно)
df = df.where(pd.notna(df), None)

# SQL-запрос для вставки данных
insert_query = """
INSERT INTO metal_products (product_name, shape, outer_diameter, wall_thickness)
VALUES (?, ?, ?, ?)
"""

# Вставляем данные в таблицу SQLite
cursor.executemany(insert_query, df[['metal_product', 'shape', 'outer_diameter', 'wall_thickness']].values.tolist())

# Сохраняем изменения
conn.commit()
conn.close()

print("Данные успешно загружены в таблицу metal_products!")

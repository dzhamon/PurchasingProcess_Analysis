import pandas as pd
import re
import sqlite3

# Загрузка CSV-файла
file_path = 'D:\My_PyQt5_Project_app\metal_products.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Проверка доступных столбцов
column_name = 'metal_product'
if column_name not in df.columns:
	raise KeyError(f"Столбец '{column_name}' не найден в данных")

# Фильтрация строк, оставляем только нужные материалы, проверяя только первое слово
materials = ['арматура', 'квадрат', 'круг', 'полоса', 'лист', 'труба', 'профиль', 'уголок', 'швеллер']
df = df[df[column_name].str.lower().str.split().str[0].isin(materials)].reset_index(drop=True)


# Функция для извлечения размеров
def extract_dimensions(name):
	dimensions = {'diameter': None, 'side_size': None, 'width': None, 'thickness': None, 'mark_number': None}
	
	if not isinstance(name, str):
		return None
	
	name = name.lower()
	words = name.split()
	first_word = words[0] if words else ""
	second_word = words[1] if len(words) > 1 else ""
	
	# Извлечение размеров в круглых скобках
	bracket_match = re.search(r'\((\d+[.,]?\d*)[xх](\d+[.,]?\d*)\)', name)
	if bracket_match:
		matches = [float(bracket_match.group(1).replace(',', '.')), float(bracket_match.group(2).replace(',', '.'))]
	else:
		matches = re.findall(r'\d+[.,]?\d*', name)
		matches = [float(m.replace(',', '.')) for m in matches]
	
	if first_word == 'арматура':
		dimensions['diameter'] = matches[0] if matches else None
	elif first_word == 'квадрат':
		dimensions['side_size'] = matches[0] if matches else None
	elif first_word == 'круг':
		dimensions['diameter'] = matches[0] if matches else None
	elif first_word == 'полоса':
		if len(matches) >= 2:
			dimensions['width'] = matches[0]
			dimensions['thickness'] = matches[1]
	elif first_word == 'лист':
		size_match = re.findall(r'(\d+)[xх](\d+)[xх](\d+)', name)
		if size_match:
			dimensions['thickness'], dimensions['width'], dimensions['side_size'] = map(float, size_match[0])
		else:
			thickness_match = re.search(r'[tт]=?(\d+)', name)
			dimensions['thickness'] = float(thickness_match.group(1)) if thickness_match else (
				matches[0] if matches else None)
			dimensions['width'], dimensions['side_size'] = 1500, 6000  # стандартные размеры
	elif first_word == 'труба':
		if second_word in ['шовная', 'бесшовная'] and bracket_match:
			dimensions['diameter'] = matches[0]
			dimensions['thickness'] = matches[1]
		elif second_word == 'профильная' and len(matches) >= 3:
			dimensions['side_size'] = matches[0]
			dimensions['width'] = matches[1]
			dimensions['thickness'] = matches[2]
		elif len(matches) >= 2:
			dimensions['diameter'] = matches[0]
			dimensions['thickness'] = matches[1]
	elif first_word == 'профиль':
		if len(matches) >= 3:
			dimensions['side_size'] = matches[0]
			dimensions['width'] = matches[1]
			dimensions['thickness'] = matches[2]
	elif first_word == 'уголок':
		if len(matches) == 2:
			dimensions['side_size'] = matches[0]
			dimensions['width'] = matches[0]  # Равнополочный уголок
			dimensions['thickness'] = matches[1]
		elif len(matches) >= 3:
			dimensions['side_size'] = matches[0]
			dimensions['width'] = matches[1]
			dimensions['thickness'] = matches[2]
	elif first_word == 'швеллер':
		mark_match = re.search(r'(\d+[А-Я]?)', name)
		if mark_match:
			dimensions['mark_number'] = mark_match.group(1)
	
	return dimensions


# Поиск размеров для отфильтрованных строк
df_dimensions = df[column_name].apply(lambda x: extract_dimensions(x))
df_dimensions = pd.DataFrame(df_dimensions.tolist(), index=df.index)

# Объединение данных
df_final = pd.concat([df, df_dimensions], axis=1)

# Подключение к базе данных
db_path = 'D:\Project_DataFiles\data\sql_krd_new.db'
conn = sqlite3.connect(db_path)

# Запись датафрейма в таблицу metal_products
df_final.to_sql('metal_products', conn, if_exists='replace', index=False)

# Проверка количества записей
count = conn.execute("SELECT COUNT(*) FROM metal_products").fetchone()[0]
print(f"Данные успешно загружены в таблицу metal_products. Всего записей: {count}")

# Закрытие соединения
conn.close()


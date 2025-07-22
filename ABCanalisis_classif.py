import sqlite3
from utils.config import SQL_PATH
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as ticker
import seaborn as sns

conn =sqlite3.connect(SQL_PATH)

# Выполнение SQL-запроса для извлечения данных
query = 'SELECT * FROM data_kp LIMIT 1000'  # Извлекаем первые 1000 строк, чтобы уменьшить объем данных
df = pd.read_sql(query, conn)
conn.close()

# Классификация по категориям, поставщикам и валютам
category_summary = df.groupby(['discipline', 'winner_name', 'currency'])['total_price'].sum().reset_index()

# Форматирование чисел с разделителями групп разрядов
category_summary['total_price'] = category_summary['total_price'].apply(lambda x: '{:,.2f}'.format(x))

# Рассчет процентного вклада каждой строки в общей сумме расходов по валютам
df['percentage'] = (df['total_price'] / df.groupby('currency')['total_price'].transform('sum')) * 100

# Сортировка данных и расчет кумулятивного процента по валютам
df = df.sort_values(by='percentage', ascending=False)
df['cumulative_percentage'] = df.groupby('currency')['percentage'].cumsum()

# Display results


# Визуализация классификации
plt.figure(figsize=(12, 6))
for currency in category_summary['currency'].unique():
    subset = category_summary[category_summary['currency'] == currency]
    plt.bar(subset['discipline'], subset['total_price'], label=currency)
plt.title('Классификация расходов по дисциплинам и валютам')
plt.xlabel('Дисциплины')
plt.ylabel('Общая сумма затрат')
plt.xticks(rotation=15)
plt.legend(title='Валюта')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'x:,.0f'))
plt.tight_layout()
plt.show()

# Определение категорий A, B и C
def abc_category(percentage):
    if percentage <= 70:
        return 'A'
    elif percentage <= 90:
        return 'B'
    else:
        return 'C'

df['ABC_Category'] = df['cumulative_percentage'].apply(abc_category)

# Форматирование чисел
df['total_price'] = df['total_price'].apply(lambda x: '{:,.2f}'.format(x))

# Визуализация ABC-анализа
plt.figure(figsize=(10, 6))
sns.barplot(x='ABC_Category', y='total_price', hue='currency', data=df)

plt.title('ABC-анализ затрат по валютам')
plt.xlabel('Категория')
plt.ylabel('Общая сумма затрат')
plt.legend(title='Валюта')

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.show()
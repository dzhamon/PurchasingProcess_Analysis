import pandas as pd
import re

def search_by_regular_expressions(df):
    """
    :param df: отфильтрованный по выбранным товарам DataFrame self._current_filtered_df
    :return: DataFrame, содержащий Лоты (и их номера), содержащие выбранные нами товары
    """
    target_items = df['good_name'] # здесь получаем Series, который далее преобразуем в итерируемый объект
    
    # Шаг 1: Создаем одно большое регулярное выражение
    # Экранируем специальные символы в названиях товаров, чтобы избежать ошибок в regex
    regex_pattern = '|'.join(re.escape(item) for item in target_items.values)
    
    # Шаг 2: Фильтруем DataFrame одним векторизованным вызовом
    processed_df = df[df['good_name'].str.contains(regex_pattern, case=False, na=False)]
    
    # Шаг 3: Группируем результат
    # group_by создает объект, который позволяет легко и эффективно агрегировать данные
    item_to_lots = processed_df.groupby('good_name')['lot_number'].unique().to_dict()
    
    df_result = pd.DataFrame(item_to_lots.items(), columns=['good_name', 'lot_number'])
    df_exploded = df_result.explode('lot_number')
    print(df_exploded)
    return df_exploded
    
    # далее полученный словарь данных преобразовать в DataFrame и вернуть в вызывающую функцию(модуль)
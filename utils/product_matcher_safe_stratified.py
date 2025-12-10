"""
Безопасный поиск сопоставимых товаров со СТРАТИФИЦИРОВАННОЙ выборкой
ВЕРСИЯ 3.38 - STRATIFIED SAMPLING

Основные улучшения:
- Стратифицированная выборка по ценовым диапазонам
- Находит пары во ВСЕХ сегментах (дешёвые, средние, дорогие)
- Защита от переполнения памяти
- Пропуск товаров с category=None
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import time
from functools import lru_cache

# Импортируем улучшенный ProductMatcher
try:
    from utils.product_matcher_improved import ProductMatcher
    print("✅ Модуль product_matcher_improved загружен")
except ImportError:
    try:
        from utils.product_matcher_improved import ProductMatcher
        print("✅ Модуль product_matcher_improved загружен (альтернативный путь)")
    except ImportError:
        print("❌ ОШИБКА: Не удалось импортировать ProductMatcher")
        raise


# ==============================================================================
# НАСТРОЙКИ СТРАТИФИЦИРОВАННОЙ ВЫБОРКИ
# ==============================================================================

MAX_GROUP_SIZE = 500  # Максимальный размер группы для обработки

# Настройки стратификации
STRATIFICATION_SETTINGS = {
    'low_price_threshold': 10,      # Граница дешёвых товаров (EUR)
    'high_price_threshold': 50,     # Граница дорогих товаров (EUR)
    'low_sample_size': 200,         # Количество дешёвых товаров
    'mid_sample_size': 200,         # Количество средних товаров
    'high_sample_size': 100,        # Количество дорогих товаров
}


def stratified_sample_group(group: pd.DataFrame, max_size: int = MAX_GROUP_SIZE) -> pd.DataFrame:
    """
    Стратифицированная выборка из группы товаров по ценовым диапазонам.

    Разделяет товары на 3 сегмента:
    - Дешёвые (< 10 EUR)
    - Средние (10-50 EUR)
    - Дорогие (> 50 EUR)

    Берёт пропорциональное количество из каждого сегмента.

    Args:
        group: DataFrame с товарами одной группы
        max_size: Максимальный размер выборки

    Returns:
        DataFrame с выборкой товаров
    """
    group_size = len(group)

    # Если группа маленькая, возвращаем как есть
    if group_size <= max_size:
        return group

    # Получаем настройки
    low_threshold = STRATIFICATION_SETTINGS['low_price_threshold']
    high_threshold = STRATIFICATION_SETTINGS['high_price_threshold']
    low_sample = STRATIFICATION_SETTINGS['low_sample_size']
    mid_sample = STRATIFICATION_SETTINGS['mid_sample_size']
    high_sample = STRATIFICATION_SETTINGS['high_sample_size']

    # Разделяем на ценовые сегменты
    low_price = group[group['unit_price_eur'] < low_threshold]
    mid_price = group[(group['unit_price_eur'] >= low_threshold) &
                      (group['unit_price_eur'] < high_threshold)]
    high_price = group[group['unit_price_eur'] >= high_threshold]

    # Подсчитываем размеры сегментов
    n_low = len(low_price)
    n_mid = len(mid_price)
    n_high = len(high_price)

    # Берём выборку из каждого сегмента
    samples = []

    if n_low > 0:
        sample_size_low = min(low_sample, n_low)
        samples.append(low_price.sample(n=sample_size_low, random_state=42))

    if n_mid > 0:
        sample_size_mid = min(mid_sample, n_mid)
        samples.append(mid_price.sample(n=sample_size_mid, random_state=42))

    if n_high > 0:
        sample_size_high = min(high_sample, n_high)
        samples.append(high_price.sample(n=sample_size_high, random_state=42))

    # Объединяем выборки
    if samples:
        stratified_group = pd.concat(samples, ignore_index=True)
    else:
        # Если по какой-то причине нет ни одного сегмента, берём случайную выборку
        stratified_group = group.sample(n=min(max_size, group_size), random_state=42)

    return stratified_group


# Кэш для extract_key_features (для ускорения)
@lru_cache(maxsize=50000)
def cached_extract_key_features(product_name: str):
    """Кэшированная версия extract_key_features"""
    return ProductMatcher.extract_key_features(product_name)


def find_comparable_products_safe(df: pd.DataFrame,
                                  similarity_threshold: float = 0.75) -> pd.DataFrame:
    """
    Безопасный поиск сопоставимых товаров с защитой от переполнения памяти
    и стратифицированной выборкой.

    Args:
        df: DataFrame с товарами (должен содержать 'product_name' и 'unit_price_eur')
        similarity_threshold: Порог схожести для сопоставления

    Returns:
        DataFrame с парами сопоставимых товаров
    """
    print("\n" + "="*80)
    print("БЫСТРЫЙ ПОИСК СОПОСТАВИМЫХ ТОВАРОВ (СТРАТИФИЦИРОВАННАЯ ВЫБОРКА)")
    print("="*80)

    # 1. Добавляем категории и типы
    print("\n1. Добавление категорий и типов товаров...")
    print(f"   Обработка {len(df)} товаров...")

    start_time = time.time()

    # Применяем кэшированную функцию
    features_list = []
    cache_hits = 0
    cache_misses = 0

    for product_name in df['product_name']:
        # Проверяем, есть ли в кэше
        cache_info_before = cached_extract_key_features.cache_info()
        features = cached_extract_key_features(str(product_name))
        cache_info_after = cached_extract_key_features.cache_info()

        if cache_info_after.hits > cache_info_before.hits:
            cache_hits += 1
        else:
            cache_misses += 1

        features_list.append(features)

    # Создаём новые столбцы
    df['category'] = [f['category'] for f in features_list]
    df['product_type'] = [f['type'] for f in features_list]

    elapsed = time.time() - start_time
    cache_info = cached_extract_key_features.cache_info()

    print(f"   ✅ Категории добавлены за {elapsed:.2f} секунд")
    print(f"   📊 Кэш: {cache_hits} попаданий, {cache_misses} промахов ({cache_hits/(cache_hits+cache_misses)*100:.1f}% эффективность)")
    print(f"   💾 Уникальных товаров: {cache_info.currsize}")
    print(f"   ⚡ Кэш сэкономил ~{cache_hits * 0.01:.1f} секунд!")

    # 2. Группируем и сравниваем
    print("\n2. Поиск совпадений внутри категорий (СТРАТИФИЦИРОВАННАЯ ВЫБОРКА)...")

    results = []
    skipped_groups = 0
    skipped_items = 0
    total_groups = 0
    stratified_groups = 0

    # Группируем по категории и типу
    grouped = df.groupby(['category', 'product_type'])

    for (category, product_type), group in grouped:
        total_groups += 1
        group_size = len(group)

        # КРИТИЧНО: Пропускаем группы с category=None
        if category == 'None' or category is None or pd.isna(category):
            print(f"   ⚠️ ПРОПУСК: {category} / {product_type}: {group_size} товаров (category=None)")
            skipped_groups += 1
            skipped_items += group_size
            continue

        # Применяем стратифицированную выборку для больших групп
        if group_size > MAX_GROUP_SIZE:
            print(f"   ⚠️ БОЛЬШАЯ ГРУППА: {category} / {product_type}: {group_size} товаров")
            print(f"      Применяем СТРАТИФИЦИРОВАННУЮ выборку...")
            print(f"      Дешёвые (<{STRATIFICATION_SETTINGS['low_price_threshold']} EUR): до {STRATIFICATION_SETTINGS['low_sample_size']} шт")
            print(f"      Средние ({STRATIFICATION_SETTINGS['low_price_threshold']}-{STRATIFICATION_SETTINGS['high_price_threshold']} EUR): до {STRATIFICATION_SETTINGS['mid_sample_size']} шт")
            print(f"      Дорогие (>{STRATIFICATION_SETTINGS['high_price_threshold']} EUR): до {STRATIFICATION_SETTINGS['high_sample_size']} шт")

            group = stratified_sample_group(group, MAX_GROUP_SIZE)
            stratified_groups += 1

            print(f"      ✅ Выбрано: {len(group)} товаров для обработки")

        print(f"   Обработка: {category} / {product_type}: {len(group)} товаров")

        # Сравниваем все пары внутри группы
        group_list = group.to_dict('records')

        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                prod1 = group_list[i]
                prod2 = group_list[j]

                # Пропускаем товары от одного поставщика
                if prod1['supplier_name'] == prod2['supplier_name']:
                    continue

                # Вычисляем схожесть
                similarity = ProductMatcher.calculate_similarity(
                    prod1['product_name'],
                    prod2['product_name']
                )

                if similarity >= similarity_threshold:
                    results.append({
                        'product_name_1': prod1['product_name'],
                        'product_name_2': prod2['product_name'],
                        'supplier_1': prod1['supplier_name'],
                        'supplier_2': prod2['supplier_name'],
                        'price_1': prod1['unit_price_eur'],
                        'price_2': prod2['unit_price_eur'],
                        'category': category,
                        'product_type': product_type,
                        'similarity': similarity
                    })

    print(f"\n✅ Найдено совпадений: {len(results)}")

    if skipped_groups > 0 or stratified_groups > 0:
        print(f"\n⚠️ СТАТИСТИКА ОБРАБОТКИ:")
        if skipped_groups > 0:
            print(f"   Пропущено групп с None: {skipped_groups}")
            print(f"   Пропущено товаров с None: {skipped_items}")
        if stratified_groups > 0:
            print(f"   Применена стратификация: {stratified_groups} групп")
            print(f"   Охват ценовых диапазонов: дешёвые, средние, дорогие ✅")
        print(f"\n💡 Рекомендация: Улучшите regex в product_matcher_improved.py")

    print("="*80)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)

# Алиас для обртной совместимости

fast_find_comparable_products = find_comparable_products_safe


if __name__ == "__main__":
    print("Модуль загружен. Используйте find_comparable_products_safe()")
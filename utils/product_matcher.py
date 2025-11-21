"""
product_matcher.py

Модуль для интеллектуального сопоставления товаров
Использует алгоритмы нечеткого сравнения строк (Fuzzy Matching)

Автор: Контрольно-ревизионный департамент
Дата: 2024
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from typing import Tuple, Dict, List, Any


# ============================================================================
# АЛГОРИТМЫ НЕЧЕТКОГО СРАВНЕНИЯ СТРОК
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Расстояние Левенштейна - минимальное количество операций
    (вставка, удаление, замена) для превращения одной строки в другую.

    Args:
        s1: первая строка
        s2: вторая строка

    Returns:
        int: расстояние (0 = идентичные строки)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Нормализованное сходство Левенштейна.

    Returns:
        float: значение от 0 до 1 (1 = идентичные)
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Сходство Джаро - учитывает совпадающие символы и транспозиции.

    Returns:
        float: значение от 0 до 1
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 1:
        match_distance = 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3.0

    return jaro


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Сходство Джаро-Винклера - улучшенная версия Джаро,
    дает больший вес совпадениям в начале строки.

    Args:
        s1: первая строка
        s2: вторая строка
        p: вес префикса (обычно 0.1)

    Returns:
        float: значение от 0 до 1
    """
    jaro_sim = jaro_similarity(s1, s2)

    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro_sim + (prefix * p * (1 - jaro_sim))


def token_set_similarity(s1: str, s2: str) -> float:
    """
    Токенизированное сравнение - разбивает строки на слова
    и сравнивает наборы токенов (коэффициент Жаккара).

    Returns:
        float: значение от 0 до 1
    """
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    jaccard = len(intersection) / len(union) if union else 0.0

    return jaccard


def sequence_matcher_similarity(s1: str, s2: str) -> float:
    """
    Встроенный Python SequenceMatcher (быстрый и эффективный).

    Returns:
        float: значение от 0 до 1
    """
    return SequenceMatcher(None, s1, s2).ratio()


# ============================================================================
# НОРМАЛИЗАЦИЯ И ПРЕДОБРАБОТКА
# ============================================================================

def normalize_product_name(name: str) -> str:
    """
    Нормализует название товара для лучшего сравнения.

    Args:
        name: исходное название товара

    Returns:
        str: нормализованное название
    """
    name = str(name).lower()

    # Убираем лишние пробелы
    name = ' '.join(name.split())

    # Заменяем различные написания
    replacements = {
        'ø': 'o',
        'ф': 'f',
        '∅': 'o',
        'х': 'x',
        '×': 'x',
        'ё': 'е',
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Убираем множественные пробелы
    name = re.sub(r'\s+', ' ', name)

    return name.strip()


def extract_key_features(product_name: str) -> Dict[str, Any]:
    """
    Извлекает ключевые характеристики из названия товара.

    Args:
        product_name: название товара

    Returns:
        dict: словарь с извлеченными характеристиками
    """
    name_lower = str(product_name).lower()

    features = {
        'type': None,
        'diameter': None,
        'thickness': None,
        'number': None,
        'material': None,
        'gost': None
    }

    # Тип изделия
    types = ['труба', 'швеллер', 'двутавр', 'арматура', 'уголок',
             'профиль', 'лист', 'полоса', 'круг', 'проволока']
    for t in types:
        if t in name_lower:
            features['type'] = t
            break

    # Диаметр
    diameter_patterns = [r'[øф∅d]\s*(\d+)', r'(\d+)\s*мм']
    for pattern in diameter_patterns:
        match = re.search(pattern, name_lower)
        if match:
            try:
                features['diameter'] = int(match.group(1))
                break
            except:
                pass

    # Номер (для швеллера, двутавра)
    if features['type'] in ['швеллер', 'двутавр']:
        number_pattern = r'[№#]?\s*(\d+\.?\d*)[упа]?'
        match = re.search(number_pattern, name_lower)
        if match:
            try:
                features['number'] = float(match.group(1))
            except:
                pass

    # ГОСТ
    gost_match = re.search(r'гост\s+[\d\-]+', name_lower)
    if gost_match:
        features['gost'] = gost_match.group(0)

    # Материал
    materials = ['ст3', 'ст20', 'с245', 'с255', '09г2с', 'оцинк']
    for mat in materials:
        if mat in name_lower:
            features['material'] = mat
            break

    return features


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ СОПОСТАВЛЕНИЯ
# ============================================================================

def smart_product_match(
        product1: str,
        product2: str,
        method: str = 'combined'
) -> Tuple[float, Dict[str, Any]]:
    """
    Умное сопоставление двух товаров с использованием выбранного алгоритма.

    Args:
        product1: название первого товара
        product2: название второго товара
        method: метод сравнения - 'levenshtein', 'jaro', 'jaro_winkler',
                'token', 'sequence', 'combined'

    Returns:
        tuple: (оценка_сходства, детали)
            - оценка_сходства: float от 0 до 1 (1 = идентичные)
            - детали: dict с подробной информацией

    Example:
        >>> score, details = smart_product_match(
        ...     "Швеллер 14П ГОСТ 8240-97 С245",
        ...     "Швеллер 14 С245 ГОСТ 8240-97"
        ... )
        >>> print(f"Сходство: {score:.3f}")
        Сходство: 0.923
    """
    # Нормализация
    norm1 = normalize_product_name(product1)
    norm2 = normalize_product_name(product2)

    # Извлечение признаков
    feat1 = extract_key_features(product1)
    feat2 = extract_key_features(product2)

    # Пре-фильтр: если типы разные - точно не совпадают
    if feat1['type'] and feat2['type'] and feat1['type'] != feat2['type']:
        return 0.0, {"reason": "Разные типы изделий"}

    # Выбор метода сравнения
    scores = {}

    if method in ['levenshtein', 'combined']:
        scores['levenshtein'] = levenshtein_similarity(norm1, norm2)

    if method in ['jaro', 'combined']:
        scores['jaro'] = jaro_similarity(norm1, norm2)

    if method in ['jaro_winkler', 'combined']:
        scores['jaro_winkler'] = jaro_winkler_similarity(norm1, norm2)

    if method in ['token', 'combined']:
        scores['token'] = token_set_similarity(norm1, norm2)

    if method in ['sequence', 'combined']:
        scores['sequence'] = sequence_matcher_similarity(norm1, norm2)

    # Комбинированный подход
    if method == 'combined':
        weights = {
            'jaro_winkler': 0.3,
            'levenshtein': 0.25,
            'token': 0.25,
            'sequence': 0.2
        }
        final_score = sum(scores[k] * weights[k] for k in weights)
    else:
        final_score = scores[method]

    # Бонус за совпадение ключевых характеристик
    bonus = 0.0
    bonus_details = []

    if feat1['diameter'] and feat2['diameter']:
        if feat1['diameter'] == feat2['diameter']:
            bonus += 0.1
            bonus_details.append("диаметр")

    if feat1['number'] and feat2['number']:
        if abs(feat1['number'] - feat2['number']) < 0.1:
            bonus += 0.1
            bonus_details.append("номер")

    if feat1['gost'] and feat2['gost']:
        if feat1['gost'] == feat2['gost']:
            bonus += 0.05
            bonus_details.append("ГОСТ")

    if feat1['material'] and feat2['material']:
        if feat1['material'] == feat2['material']:
            bonus += 0.05
            bonus_details.append("материал")

    final_score = min(1.0, final_score + bonus)

    details = {
        'scores': scores,
        'bonus': bonus,
        'matching_features': bonus_details,
        'normalized1': norm1,
        'normalized2': norm2,
        'features1': feat1,
        'features2': feat2
    }

    return final_score, details


# ============================================================================
# ФУНКЦИЯ ДЛЯ ПОИСКА В ДАТАФРЕЙМЕ
# ============================================================================

def find_comparable_products(
        df: pd.DataFrame,
        threshold: float = 0.85,
        method: str = 'combined',
        verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Находит группы сопоставимых товаров в датафрейме.

    Args:
        df: DataFrame с колонками 'product_name', 'counterparty_name', 'unit_price_uzs'
        threshold: минимальный порог сходства (0.0-1.0), по умолчанию 0.85
        method: алгоритм сравнения, по умолчанию 'combined'
        verbose: выводить ли прогресс, по умолчанию True

    Returns:
        list: список групп с результатами сравнения

    Example:
        >>> results = find_comparable_products(df, threshold=0.85)
        >>> for group in results:
        ...     if group['price_diff_pct'] > 30:
        ...         print(f"Разница в ценах: {group['price_diff_pct']:.1f}%")
    """
    if verbose:
        print(f"\nПоиск сопоставимых товаров...")
        print(f"Записей: {len(df)}, Порог: {threshold}, Метод: {method}")
        print("-" * 80)

    results = []
    processed = set()

    # Проверяем наличие необходимых колонок
    required_cols = ['product_name', 'counterparty_name', 'unit_price_eur']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Отсутствует колонка: {col}")

    products = df[required_cols].values

    for i in range(len(products)):
        if i in processed:
            continue

        matches = [i]
        prod1, supplier1, price1 = products[i]

        for j in range(i + 1, len(products)):
            if j in processed:
                continue

            prod2, supplier2, price2 = products[j]

            # Не сравниваем товары от одного поставщика
            if supplier1 == supplier2:
                continue

            # Сравнение
            from utils.product_matcher_improved import ProductMatcher
            score = ProductMatcher.calculate_similarity(prod1, prod2)

            if score >= threshold:
                matches.append(j)
                processed.add(j)

        if len(matches) > 1:
            # Нашли группу сопоставимых товаров
            group = []
            for idx in matches:
                prod, supplier, price = products[idx]
                group.append({
                    'index': idx,
                    'product': prod,
                    'supplier': supplier,
                    'price': price
                })
                processed.add(idx)

            # Анализ цен в группе
            prices = [item['price'] for item in group]
            min_price = min(prices)
            max_price = max(prices)
            price_diff_pct = ((max_price - min_price) / min_price * 100) if min_price > 0 else 0

            results.append({
                'group': group,
                'count': len(group),
                'min_price': min_price,
                'max_price': max_price,
                'price_diff_pct': price_diff_pct
            })

    if verbose:
        print(f"\n✓ Найдено групп сопоставимых товаров: {len(results)}")

    return results


# ============================================================================
# ФУНКЦИЯ ДЛЯ ВЫЯВЛЕНИЯ ЦЕНОВЫХ РАСХОЖДЕНИЙ
# ============================================================================

def find_price_discrepancies(
        df: pd.DataFrame,
        threshold: float = 0.85,
        method: str = 'combined',
        price_diff_threshold: float = 30.0
) -> pd.DataFrame:
    """
    Находит товары с похожими названиями но значительно различающимися ценами.

    Args:
        df: DataFrame с данными о закупках
        threshold: порог сходства названий (0-1)
        method: метод сравнения
        price_diff_threshold: минимальная разница в ценах в % для отчета

    Returns:
        DataFrame с результатами анализа

    Example:
        >>> discrepancies = find_price_discrepancies(df, price_diff_threshold=30)
        >>> print(discrepancies[['Категория', 'Поставщиков', 'Разница_%']])
    """
    comparable_groups = find_comparable_products(df, threshold, method, verbose=False)

    # Фильтруем группы с большой разницей в ценах
    high_diff_groups = [
        g for g in comparable_groups
        if g['price_diff_pct'] > price_diff_threshold
    ]

    print(f"\nАНАЛИЗ ЦЕНОВЫХ РАСХОЖДЕНИЙ")
    print("=" * 80)
    print(f"Всего групп сопоставимых товаров: {len(comparable_groups)}")
    print(f"Групп с разницей цен >{price_diff_threshold}%: {len(high_diff_groups)}")

    if high_diff_groups:
        print(f"\n⚠️ КРИТИЧЕСКИЕ РАСХОЖДЕНИЯ:")
        print("-" * 80)

        results = []
        for i, group_data in enumerate(high_diff_groups, 1):
            print(f"\nГруппа {i}: {group_data['count']} поставщиков")
            print(f"Разница в ценах: {group_data['price_diff_pct']:.1f}%")

            for item in group_data['group']:
                print(f"  • {item['supplier']}")
                print(f"    {item['product'][:70]}")
                print(f"    Цена: {item['price']:,.0f} UZS")

                results.append({
                    'Группа': i,
                    'Товар': item['product'],
                    'Поставщик': item['supplier'],
                    'Цена_UZS': item['price'],
                    'Разница_%': group_data['price_diff_pct']
                })

        return pd.DataFrame(results)
    else:
        print("\n✓ Критических расхождений не обнаружено")
        return pd.DataFrame()


# ============================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ БЫСТРОЙ ПРОВЕРКИ
# ============================================================================

def quick_match(product1: str, product2: str, threshold: float = 0.85) -> bool:
    """
    Быстрая проверка - совпадают ли два товара.

    Args:
        product1: первый товар
        product2: второй товар
        threshold: порог сходства

    Returns:
        bool: True если товары совпадают

    Example:
        >>> if quick_match("Швеллер 14П", "Швеллер 14"):
        ...     print("Совпадают!")
    """
    score, _ = smart_product_match(product1, product2, method='combined')
    return score >= threshold


# ============================================================================
# ИНФОРМАЦИЯ О МОДУЛЕ
# ============================================================================

def get_module_info():
    """Возвращает информацию о модуле."""
    info = """
    Product Matcher Module v1.0
    ============================

    Модуль для интеллектуального сопоставления товаров.

    Основные функции:
    - smart_product_match(): сравнение двух товаров
    - find_comparable_products(): поиск групп похожих товаров
    - find_price_discrepancies(): выявление ценовых аномалий
    - quick_match(): быстрая проверка совпадения

    Поддерживаемые алгоритмы:
    - Levenshtein Distance
    - Jaro Similarity
    - Jaro-Winkler Similarity
    - Token Set (Jaccard)
    - Sequence Matcher
    - Combined (рекомендуется)

    Использование:
        from utils.product_matcher import smart_product_match

        score, details = smart_product_match(
            "Швеллер 14П ГОСТ 8240-97",
            "Швеллер 14 С245"
        )
        print(f"Сходство: {score:.3f}")
    """
    return info


if __name__ == "__main__":
    # Тестирование модуля при прямом запуске
    print(get_module_info())

    # Примеры использования
    print("\n" + "="*80)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ")
    print("="*80)

    test_pairs = [
        ("Швеллер 14П ГОСТ 8240-97 С245", "Швеллер 14 С245 ГОСТ 8240-97"),
        ("Труба стальная Ø530х10-К50", "Труба стальная ф530"),
        ("Арматура 12 А500", "АРМАТУРА 12-А500"),
    ]

    for prod1, prod2 in test_pairs:
        score, details = smart_product_match(prod1, prod2)
        match = "✓ СОВПАДАЮТ" if score >= 0.85 else "✗ РАЗНЫЕ"
        print(f"\n{prod1}")
        print(f"{prod2}")
        print(f"Сходство: {score:.3f} {match}")
        if details['matching_features']:
            print(f"Совпадают: {', '.join(details['matching_features'])}")
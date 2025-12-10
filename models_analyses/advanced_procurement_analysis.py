"""
Расширенные метод анализа закупочных процессов на данных
таблицы data_contract базы данных
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
from utils.functions import CurrencyConverter

# Настройка для русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_summary_report(comparable_df, critical_df):
    """
    Создает Excel-отчет с несколькими листами
    """
    import pandas as pd
    from datetime import datetime

    filename = f'отчет_по_ценам_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:

        # Лист 1: Критические расхождения
        if not critical_df.empty:
            critical_df.to_excel(writer, sheet_name='Критические', index=False)

        # Лист 2: Все совпадения
        comparable_df.to_excel(writer, sheet_name='Все совпадения', index=False)

        # Лист 3: Статистика по категориям
        category_stats = comparable_df.groupby('category').agg({
            'price_diff_pct': ['count', 'mean', 'median', 'max'],
            'price_diff': ['sum', 'mean']
        }).round(2)
        category_stats.to_excel(writer, sheet_name='По категориям')

        # Лист 4: Рейтинг поставщиков (у кого чаще дешевле)
        cheaper_rating = comparable_df['cheaper_supplier'].value_counts().to_frame('Количество')
        cheaper_rating['Средняя_экономия_%'] = comparable_df.groupby('cheaper_supplier')['price_diff_pct'].mean()
        cheaper_rating.to_excel(writer, sheet_name='Лучшие поставщики')

        # Лист 5: Рейтинг поставщиков (у кого чаще дороже)
        expensive_rating = comparable_df['expensive_supplier'].value_counts().to_frame('Количество')
        expensive_rating['Средняя_переплата_%'] = comparable_df.groupby('expensive_supplier')['price_diff_pct'].mean()
        expensive_rating.to_excel(writer, sheet_name='Дорогие поставщики')

        # Лист 6: Топ-20 товаров с наибольшей разницей
        top_20 = comparable_df.nlargest(20, 'price_diff')[
            ['product1', 'supplier1', 'price1', 'supplier2', 'price2',
             'price_diff', 'price_diff_pct', 'category']
        ]
        top_20.to_excel(writer, sheet_name='Топ-20 разницы', index=False)

    print(f"✓ Сводный отчет: {filename}")

    return filename

def advanced_procurement_analysis(df):
    # Импорт модуля сопоставимых товаров
    try:
        # from utils.product_matcher_optimized import (
        #     fast_find_comparable_products,
        #     find_price_discrepancies,
        #     smart_product_match
        # )
        from utils.product_matcher_save import fast_find_comparable_products
        MATCHER_AVAILABLE = True
        print("Модуль product_matched загружен")
    except ImportError:
        MATCHER_AVAILABLE = False
        print("⚠ Модуль product_matcher не найден - умное сопоставление отключено")
        print("  Разместите product_matcher.py в папку utils/")

    # В датафрейме удаляем повторяющиеся строки
    df = df.drop_duplicates()

    print("Мы в модуле advanced_procurement_analysis")
    from utils.config import BASE_DIR
    OUT_DIR = os.path.join(BASE_DIR, "Расширенный анализ закупок")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Конвертация валют")
    # ======== Конвертация валют =======
    columns_info = [
        ('unit_price', 'contract_currency', 'unit_price_eur'),
        ('total_contract_amount', 'contract_currency', 'total_contract_amount_eur')
    ]
    converter = CurrencyConverter()

    # Конвертируем и сохраняем два столбца
    converted_df = converter.convert_multiple_columns(
        df=df, columns_info=columns_info)
    # импорт модуля сопоставления товаров


    # ======== БЛОК 1. ИНТЕЛЛЕКТУАЛЬНОЕ СРАВНЕНИЕ ЦЕН МЕЖДУ ПОСТАВЩИКАМИ ======
    if MATCHER_AVAILABLE:
        # Поиск сопоставимых товаров

        comparable_groups = fast_find_comparable_products(
            converted_df,
            threshold=0.85,
        )

        comparable_df = comparable_groups

        if not comparable_df.empty:
            print(f"\n✓ Найдено {len(comparable_df)} групп сопоставимых товаров")

            # Основной файл с результатами
            output_file = 'comparable_products_DEBUG.xlsx'
            comparable_df.to_excel(output_file, index=False)
            print(f"✓ Сохранено: {output_file}")
            print(f"  Строк: {len(comparable_df)}")
            print(f"  Колонок: {len(comparable_df.columns)}")

            # Выявление критических расхождений
            critical_df = comparable_df[comparable_df['price_diff_pct'] > 30].copy()

            print(f"[DEBUG] Из них критических (>30%): {len(critical_df)}")

            if not critical_df.empty:
                print(f"\n⚠️ КРИТИЧЕСКИЕ РАСХОЖДЕНИЯ В ЦЕНАХ: {len(critical_df)} групп")
                print("-" * 80)

                # Показываем топ-10
                top_critical =critical_df.head(10)

                # Детальный отчет
                for idx, row in top_critical.iterrows():
                    print(f"\n┌─ Пара #{idx+1} {'─' * 65}┐")
                    print(f"│ Товар: {row['product1'][:70]}")
                    print(f"│")
                    print(f"│ 💰 {row['supplier1'][:35]:<35}: {row['price1']:>12,.2f} EUR")
                    print(f"│ 💰 {row['supplier2'][:35]:<35}: {row['price2']:>12,.2f} EUR")
                    print(f"│")
                    print(f"│ 📈 Разница: {row['price_diff_pct']:>6.1f}% ({row['price_diff']:>10,.2f} EUR)")
                    print(f"│ 🔍 Схожесть товаров: {row['similarity']:>5.1%}")
                    print(f"│ ✓  Дешевле у: {row['cheaper_supplier']}")
                    print(f"│ 📦 Категория: {row['category']} / {row['type']}")
                    print("└" + "─" * 78 + "┘")

                if len(critical_df) > 10:
                    print(f"\n... и ещё {len(critical_df) - 10} критических расхождений")

                # Статистика по критическим расхождениям
                print("\n" + "="*80)
                print("СТАТИСТИКА ПО КРИТИЧЕСКИМ РАСХОЖДЕНИЯМ")
                print("="*80)

                print(f"\n📊 Общие показатели:")
                print(f"   Средняя разница: {critical_df['price_diff_pct'].mean():.1f}%")
                print(f"   Максимальная разница: {critical_df['price_diff_pct'].max():.1f}%")
                print(f"   Медианная разница: {critical_df['price_diff_pct'].median():.1f}%")

                # По категориям
                print(f"\n📊 По категориям:")
                category_stats = critical_df.groupby('category').agg({
                    'price_diff_pct': ['count', 'mean', 'max'],
                    'price_diff': 'sum'
                }).round(1)

                category_stats.columns = ['Количество', 'Средняя_%', 'Макс_%', 'Сумма_разницы']
                print(category_stats.to_string())

                # По поставщикам
                print(f"\n📊 Топ-10 поставщиков с завышенными ценами:")

                # Считаем сколько раз каждый поставщик был дороже
                expensive_suppliers = critical_df['expensive_supplier'].value_counts().head(10)

                for supplier, count in expensive_suppliers.items():
                    # Средняя переплата у этого поставщика
                    supplier_data = critical_df[critical_df['expensive_supplier'] == supplier]
                    avg_overprice = supplier_data['price_diff_pct'].mean()

                    print(f"   {supplier[:45]:<45}: {count:>3} раз, в среднем +{avg_overprice:.1f}%")

                # Рекомендации
                print("\n" + "="*80)
                print("💡 РЕКОМЕНДАЦИИ")
                print("="*80)

                # Потенциальная экономия
                potential_savings = critical_df['price_diff'].sum()
                print(f"\n💰 Потенциальная экономия при переходе на лучшие цены:")
                print(f"   {potential_savings:,.2f} EUR")

                # Топ-5 товаров для пересмотра
                print(f"\n🎯 Топ-5 товаров для срочного пересмотра поставщиков:")
                top_savings = critical_df.nlargest(5, 'price_diff')

                for i, (idx, row) in enumerate(top_savings.iterrows(), 1):
                    print(f"\n   {i}. {row['product1'][:60]}")
                    print(f"      Текущий: {row['expensive_supplier']} - {row['price2']:,.2f} EUR")
                    print(f"      Лучший: {row['cheaper_supplier']} - {row['price1']:,.2f} EUR")
                    print(f"      Экономия: {row['price_diff']:,.2f} EUR ({row['price_diff_pct']:.1f}%)")

            # Общая статистика по ВСЕМ совпадениям
            print("\n" + "="*80)
            print("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ СОПОСТАВИМЫМ ТОВАРАМ")
            print("="*80)

            print(f"\n📊 Распределение разницы цен:")
            print(f"   0-10%:    {len(comparable_df[comparable_df['price_diff_pct'] <= 10])} пар")
            print(f"   10-20%:   {len(comparable_df[(comparable_df['price_diff_pct'] > 10) & (comparable_df['price_diff_pct'] <= 20)])} пар")
            print(f"   20-30%:   {len(comparable_df[(comparable_df['price_diff_pct'] > 20) & (comparable_df['price_diff_pct'] <= 30)])} пар")
            print(f"   >30%:     {len(comparable_df[comparable_df['price_diff_pct'] > 30])} пар")

            # Сохранение результатов
            print("\n" + "="*80)
            print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
            print("="*80)

            # Сохраняем все совпадения
            comparable_df.to_excel('все_сопоставимые_товары.xlsx', index=False)
            print("✓ Все совпадения: все_сопоставимые_товары.xlsx")

            # Сохраняем только критические
            if not critical_df.empty:
                critical_df.to_excel('критические_расхождения.xlsx', index=False)
                print("✓ Критические: критические_расхождения.xlsx")

            # Создаем сводный отчет
            create_summary_report(comparable_df, critical_df)

        else:
            print("\n⚠️  Сопоставимых товаров не найдено")
            print("Попробуйте снизить порог схожести (threshold)")

    return comparable_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def diagnose_data_quality(f_contracts, sub_folder):
    """
    Диагностика качества данных перед построением регрессии
    """

    print("\n" + "="*80)
    print("ДИАГНОСТИКА ДАННЫХ")
    print("="*80)

    # 1. Базовая информация о данных
    print(f"\n1. Общий размер данных: {len(f_contracts)} контрактов")
    print(f"   Период данных: {f_contracts['contract_signing_date'].min()} - {f_contracts['contract_signing_date'].max()}")

    # 2. Агрегация по неделям
    df_weekly = f_contracts.groupby('contract_week').agg(
        total_amount=('total_contract_amount_eur', 'sum'),
        contract_count=('contract_number', 'nunique')
    ).reset_index()

    print(f"\n2. После агрегации по неделям: {len(df_weekly)} наблюдений")
    print(f"   ВНИМАНИЕ: Для регрессии рекомендуется минимум 50-100 наблюдений")

    if len(df_weekly) < 30:
        print(f"   !!! У вас СЛИШКОМ МАЛО данных ({len(df_weekly)} недель)")
        print(f"   !!! Регрессия будет работать плохо")

    # 3. Проверка на пропуски в неделях
    df_weekly['contract_week'] = df_weekly['contract_week'].apply(lambda x: x.start_time)
    df_weekly = df_weekly.sort_values('contract_week')

    # 4. Статистика целевой переменной
    print(f"\n3. Статистика целевой переменной (total_amount):")
    print(f"   Среднее: {df_weekly['total_amount'].mean():,.0f} EUR")
    print(f"   Медиана: {df_weekly['total_amount'].median():,.0f} EUR")
    print(f"   Стд. отклонение: {df_weekly['total_amount'].std():,.0f} EUR")
    print(f"   Мин: {df_weekly['total_amount'].min():,.0f} EUR")
    print(f"   Макс: {df_weekly['total_amount'].max():,.0f} EUR")

    # Коэффициент вариации (volatility)
    cv = df_weekly['total_amount'].std() / df_weekly['total_amount'].mean()
    print(f"   Коэффициент вариации: {cv:.2f}")

    if cv > 1.0:
        print(f"   !!! Данные ОЧЕНЬ волатильны (CV > 1.0)")
        print(f"   !!! Это затрудняет прогнозирование")

    # 5. Визуализация временного ряда
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df_weekly['contract_week'], df_weekly['total_amount'], marker='o', linewidth=1.5)
    plt.title('Общая сумма контрактов по неделям')
    plt.ylabel('Сумма (EUR)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.plot(df_weekly['contract_week'], df_weekly['contract_count'], marker='o', color='green', linewidth=1.5)
    plt.title('Количество контрактов по неделям')
    plt.ylabel('Количество')
    plt.xlabel('Дата')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    diagnostics_path = os.path.join(sub_folder, 'data_diagnostics.png')
    plt.savefig(diagnostics_path, dpi=100)
    plt.close()
    print(f"\n   График сохранен: {diagnostics_path}")

    # 6. Корреляция с лагами
    print(f"\n4. Автокорреляция (насколько прошлые значения связаны с будущими):")
    for lag in [1, 2, 3, 4]:
        if len(df_weekly) > lag:
            corr = df_weekly['total_amount'].corr(df_weekly['total_amount'].shift(lag))
            print(f"   Лаг {lag}: {corr:.3f}", end="")
            if abs(corr) < 0.3:
                print(" (слабая связь)")
            elif abs(corr) < 0.7:
                print(" (средняя связь)")
            else:
                print(" (сильная связь)")

    print("\n" + "="*80)
    print("РЕКОМЕНДАЦИИ")
    print("="*80)

    return df_weekly


def improved_feature_engineering(f_contracts, sub_folder):
    """
    Улучшенная версия с диагностикой и адаптивными параметрами
    """

    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    # Диагностика данных
    df_weekly = diagnose_data_quality(f_contracts, sub_folder)

    num_observations = len(df_weekly)

    # Адаптивный выбор параметров в зависимости от размера данных
    if num_observations < 20:
        print("\n!!! КРИТИЧЕСКИ МАЛО ДАННЫХ - регрессия невозможна")
        print("Рекомендация: Соберите больше данных или используйте месячную агрегацию")
        return None

    elif num_observations < 40:
        print("\n>>> Мало данных - используем упрощенную модель:")
        num_lags = 1  # Только 1 лаг
        test_size = 0.2  # Меньше тестовая выборка
        print(f"    - Количество лагов: {num_lags}")
        print(f"    - Размер тестовой выборки: {test_size*100:.0f}%")

    elif num_observations < 80:
        print("\n>>> Средний объем данных - стандартная модель:")
        num_lags = 2
        test_size = 0.25
        print(f"    - Количество лагов: {num_lags}")
        print(f"    - Размер тестовой выборки: {test_size*100:.0f}%")

    else:
        print("\n>>> Достаточно данных - полная модель:")
        num_lags = 4
        test_size = 0.3
        print(f"    - Количество лагов: {num_lags}")
        print(f"    - Размер тестовой выборки: {test_size*100:.0f}%")

    # Создание признаков
    df_weekly['week_number'] = np.arange(len(df_weekly))
    df_weekly['total_amount_smooth'] = df_weekly['total_amount'].rolling(window=3, min_periods=1).mean()

    # Целевая переменная - НЕ логарифмируем, если данных мало
    if num_observations < 40:
        Y = df_weekly['total_amount_smooth'].shift(-1).dropna()
        use_log = False
        print("    - Без логарифмирования (из-за малого объема данных)")
    else:
        Y = np.log1p(np.maximum(df_weekly['total_amount_smooth'].shift(-1), 1)).dropna()
        use_log = True
        print("    - С логарифмированием")

    # Простые признаки
    features_to_lag = ['total_amount', 'contract_count']

    X_list = []
    for feature in features_to_lag:
        for i in range(1, num_lags + 1):
            col_name = f'{feature}_lag_{i}'
            df_weekly[col_name] = df_weekly[feature].shift(i)
            X_list.append(col_name)

    X_list.append('week_number')

    X = df_weekly[X_list].dropna()

    # Выравнивание индексов
    common_index = X.index.intersection(Y.index)
    X = X.loc[common_index]
    Y = Y.loc[common_index]

    print(f"\n>>> Итого для модели:")
    print(f"    - Признаков: {X.shape[1]}")
    print(f"    - Наблюдений: {X.shape[0]}")
    print(f"    - Для обучения: ~{int(X.shape[0]*(1-test_size))}")
    print(f"    - Для теста: ~{int(X.shape[0]*test_size)}")

    if X.shape[0] < 15:
        print("\n!!! Слишком мало наблюдений после создания лагов")
        print("Рекомендация: используйте меньше лагов или соберите больше данных")
        return None

    return {
        'X': X,
        'Y': Y,
        'use_log': use_log,
        'test_size': test_size,
        'num_lags': num_lags,
        'df_weekly': df_weekly
    }

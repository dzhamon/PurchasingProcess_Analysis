import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_enhanced_features(f_contracts, sub_folder):
    """
    Создание расширенного набора признаков с учетом сезонности,
    паттернов активности и структуры контрактов
    """

    print("\n" + "="*80)
    print("СОЗДАНИЕ РАСШИРЕННЫХ ПРИЗНАКОВ")
    print("="*80)

    # ===== 1. БАЗОВАЯ АГРЕГАЦИЯ ПО НЕДЕЛЯМ =====
    print("\n1. Базовая агрегация по неделям...")

    df_features = f_contracts.groupby('contract_week').agg(
        # Основные метрики
        total_amount=('total_contract_amount_eur', 'sum'),
        avg_unit_price=('unit_price_eur', 'mean'),
        contract_count=('contract_number', 'nunique'),
        unique_counterparties=('counterparty_name', 'nunique'),

        # Статистика по суммам контрактов
        max_contract=('total_contract_amount_eur', 'max'),
        min_contract=('total_contract_amount_eur', 'min'),
        std_contract=('total_contract_amount_eur', 'std'),
        median_contract=('total_contract_amount_eur', 'median'),
    ).reset_index()

    df_features['contract_week'] = df_features['contract_week'].apply(lambda x: x.start_time)
    df_features = df_features.sort_values('contract_week').reset_index(drop=True)

    print(f"   Создано базовых признаков: {df_features.shape[1] - 1}")

    # ===== 2. СЕЗОННОСТЬ И ВРЕМЕННЫЕ ПРИЗНАКИ =====
    print("\n2. Добавление сезонности...")

    # Номер недели в году (для сезонности)
    df_features['week'] = df_features['contract_week'].dt.isocalendar().week

    # Тригонометрическое кодирование сезонности (цикличность)
    df_features['week_sin'] = np.sin(2 * np.pi * df_features['week'] / 52)
    df_features['week_cos'] = np.cos(2 * np.pi * df_features['week'] / 52)

    # Месяц и квартал
    df_features['month'] = df_features['contract_week'].dt.month
    df_features['quarter'] = df_features['contract_week'].dt.quarter

    # Индекс времени (линейный тренд)
    df_features['week_number'] = np.arange(len(df_features))

    print(f"   Добавлено временных признаков: 6")

    # ===== 3. СТРУКТУРА КОНТРАКТОВ =====
    print("\n3. Анализ структуры контрактов...")

    # Средний размер контракта
    df_features['avg_contract_size'] = df_features['total_amount'] / df_features['contract_count'].replace(0, 1)

    # Концентрация (доля максимального контракта)
    df_features['max_concentration'] = df_features['max_contract'] / df_features['total_amount'].replace(0, 1)

    # Вариативность размеров контрактов
    df_features['contract_cv'] = df_features['std_contract'] / df_features['avg_contract_size'].replace(0, 1)

    # Флаг наличия крупного контракта (> 75-го процентиля)
    large_threshold = df_features['max_contract'].quantile(0.75)
    df_features['has_large_contract'] = (df_features['max_contract'] > large_threshold).astype(int)

    print(f"   Добавлено структурных признаков: 4")

    # ===== 4. ПАТТЕРНЫ АКТИВНОСТИ =====
    print("\n4. Создание паттернов активности...")

    # Сглаженные значения (скользящее среднее)
    df_features['amount_ma3'] = df_features['total_amount'].rolling(window=3, min_periods=1).mean()
    df_features['amount_ma4'] = df_features['total_amount'].rolling(window=4, min_periods=1).mean()

    # Тренд (изменение относительно прошлой недели)
    df_features['amount_change'] = df_features['total_amount'].diff()
    df_features['amount_pct_change'] = df_features['total_amount'].pct_change().fillna(0)

    # Волатильность (стандартное отклонение за последние 4 недели)
    df_features['amount_volatility'] = df_features['total_amount'].rolling(window=4, min_periods=1).std()

    # Ускорение (изменение тренда)
    df_features['amount_acceleration'] = df_features['amount_change'].diff()

    # Флаг роста/падения
    df_features['is_growing'] = (df_features['amount_change'] > 0).astype(int)

    print(f"   Добавлено паттернов активности: 7")

    # ===== 5. ПРИЗНАКИ КОНТРАГЕНТОВ =====
    print("\n5. Анализ активности контрагентов...")

    # Средний контракт на контрагента
    df_features['amount_per_counterparty'] = df_features['total_amount'] / df_features['unique_counterparties'].replace(0, 1)

    # Изменение количества контрагентов
    df_features['counterparties_change'] = df_features['unique_counterparties'].diff()

    print(f"   Добавлено признаков контрагентов: 2")

    # ===== 6. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (опционально) =====
    print("\n6. Обработка категориальных признаков...")

    category_features_count = 0

    # Если есть данные по дисциплинам
    if 'discipline' in f_contracts.columns:
        # Топ-5 дисциплин
        top_disciplines = f_contracts['discipline'].value_counts().head(5).index

        for discipline in top_disciplines:
            col_name = f'discipline_{discipline}'.replace(' ', '_').replace('/', '_')
            discipline_contracts = f_contracts[f_contracts['discipline'] == discipline].groupby('contract_week').agg(
                count=('contract_number', 'nunique'),
                amount=('total_contract_amount_eur', 'sum')
            )

            # Количество контрактов по дисциплине
            df_features[f'{col_name}_count'] = discipline_contracts['count'].reindex(
                df_features['contract_week'].dt.to_period('W').apply(lambda x: x.start_time),
                fill_value=0
            ).values

            category_features_count += 1

    print(f"   Добавлено категориальных признаков: {category_features_count}")

    # ===== 7. СОЗДАНИЕ ЛАГОВЫХ ПРИЗНАКОВ =====
    print("\n7. Создание лаговых признаков...")

    # Определяем, какие признаки использовать для лагов
    features_for_lags = [
        'total_amount',
        'contract_count',
        'avg_contract_size',
        'unique_counterparties',
        'amount_ma3',
        'max_concentration',
        'week_sin',
        'week_cos'
    ]

    # Определяем количество лагов в зависимости от объема данных
    num_observations = len(df_features)
    if num_observations < 40:
        num_lags = 1
    elif num_observations < 80:
        num_lags = 2
    else:
        num_lags = 4

    print(f"   Количество наблюдений: {num_observations}")
    print(f"   Количество лагов: {num_lags}")

    lag_features = []
    for feature in features_for_lags:
        if feature in df_features.columns:
            for lag in range(1, num_lags + 1):
                lag_col = f'{feature}_lag_{lag}'
                df_features[lag_col] = df_features[feature].shift(lag)
                lag_features.append(lag_col)

    print(f"   Создано лаговых признаков: {len(lag_features)}")

    # ===== 8. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ =====
    print("\n8. Создание целевой переменной...")

    # Целевая переменная - сглаженная сумма следующей недели
    df_features['target_smooth'] = df_features['amount_ma3'].shift(-1)

    # Логарифмирование (если достаточно данных)
    if num_observations >= 40:
        Y = np.log1p(np.maximum(df_features['target_smooth'], 1)).dropna()
        use_log = True
        print("   Используется логарифмирование целевой переменной")
    else:
        Y = df_features['target_smooth'].dropna()
        use_log = False
        print("   Без логарифмирования (малый объем данных)")

    # ===== 9. ФИНАЛЬНЫЙ СПИСОК ПРИЗНАКОВ =====
    print("\n9. Формирование финального набора признаков...")

    # Список всех признаков (исключаем служебные и целевую переменную)
    exclude_cols = ['contract_week', 'target_smooth', 'total_amount',
                    'max_contract', 'min_contract', 'std_contract', 'median_contract']

    feature_columns = [col for col in df_features.columns
                       if col not in exclude_cols and not df_features[col].isna().all()]

    X = df_features[feature_columns].dropna()

    # Выравнивание индексов X и Y
    common_index = X.index.intersection(Y.index)
    X = X.loc[common_index]
    Y = Y.loc[common_index]

    print(f"\n   ИТОГО признаков: {X.shape[1]}")
    print(f"   ИТОГО наблюдений: {X.shape[0]}")

    # Вывод списка всех признаков по категориям
    print("\n" + "="*80)
    print("СПИСОК СОЗДАННЫХ ПРИЗНАКОВ ПО КАТЕГОРИЯМ")
    print("="*80)

    print("\nВременные признаки:")
    temporal = [c for c in X.columns if any(k in c for k in ['week', 'month', 'quarter', 'number'])]
    for i, f in enumerate(temporal, 1):
        print(f"   {i}. {f}")

    print("\nСтруктурные признаки:")
    structural = [c for c in X.columns if any(k in c for k in ['avg_', 'concentration', 'cv', 'has_'])]
    for i, f in enumerate(structural, 1):
        print(f"   {i}. {f}")

    print("\nПаттерны активности:")
    patterns = [c for c in X.columns if any(k in c for k in ['ma', 'change', 'volatility', 'acceleration', 'growing'])]
    for i, f in enumerate(patterns, 1):
        print(f"   {i}. {f}")

    print("\nЛаговые признаки:")
    lags = [c for c in X.columns if 'lag' in c]
    print(f"   Всего лаговых признаков: {len(lags)}")
    print(f"   Примеры: {', '.join(lags[:5])}")

    print("\n" + "="*80)

    return {
        'X': X,
        'Y': Y,
        'use_log': use_log,
        'num_lags': num_lags,
        'df_features': df_features,
        'feature_columns': feature_columns
    }


# Пример использования в вашей функции feature_engineering
"""
def feature_engineering(f_contracts, sub_folder):

    # Создаем расширенные признаки
    data_dict = create_enhanced_features(f_contracts, sub_folder)

    if data_dict is None or data_dict['X'].shape[0] < 15:
        print("Недостаточно данных для построения модели")
        return

    X = data_dict['X']
    Y = data_dict['Y']
    use_log = data_dict['use_log']

    # Определяем размер тестовой выборки
    test_size = 0.2 if X.shape[0] < 50 else 0.3

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на train/test (без перемешивания для временных рядов)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=test_size, shuffle=False
    )

    print(f"\nОбучающая выборка: {X_train.shape[0]}, Тестовая выборка: {X_test.shape[0]}")

    # Вызов функции сравнения моделей
    best_model_name, results_df, best_model = compare_regression_models(
        X_train, X_test, Y_train, Y_test, scaler, X.columns, sub_folder
    )

    # Прогнозирование (используйте код из предыдущего ответа)
    ...
"""
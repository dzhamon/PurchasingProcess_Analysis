import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler

def smart_feature_selection(X_train, Y_train, X_test, Y_test, feature_names, sub_folder, top_k=15):
    """
    Интеллектуальный отбор признаков с использованием нескольких методов

    Параметры:
    - X_train, Y_train: обучающие данные
    - X_test, Y_test: тестовые данные
    - feature_names: названия признаков
    - sub_folder: папка для сохранения результатов
    - top_k: количество лучших признаков (по умолчанию 15)
    """

    print("\n" + "="*80)
    print("АВТОМАТИЧЕСКИЙ ОТБОР ПРИЗНАКОВ")
    print("="*80)

    print(f"\nИсходное количество признаков: {X_train.shape[1]}")
    print(f"Будем отбирать топ-{top_k} признаков")

    # ===== МЕТОД 1: Random Forest Feature Importance =====
    print("\n1. Random Forest - важность признаков...")

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=8,
        n_jobs=-1
    )
    rf.fit(X_train, Y_train)

    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)

    print(f"   Топ-5 по RF: {', '.join(rf_importance.head(5)['feature'].tolist())}")

    # ===== МЕТОД 2: F-статистика (корреляция с целевой переменной) =====
    print("\n2. F-статистика - корреляция с целевой переменной...")

    selector_f = SelectKBest(score_func=f_regression, k='all')
    selector_f.fit(X_train, Y_train)

    f_scores = pd.DataFrame({
        'feature': feature_names,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)

    print(f"   Топ-5 по F-stat: {', '.join(f_scores.head(5)['feature'].tolist())}")

    # ===== МЕТОД 3: Mutual Information (нелинейная зависимость) =====
    print("\n3. Mutual Information - нелинейная зависимость...")

    mi_scores = mutual_info_regression(X_train, Y_train, random_state=42)

    mi_importance = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    print(f"   Топ-5 по MI: {', '.join(mi_importance.head(5)['feature'].tolist())}")

    # ===== ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ =====
    print("\n4. Объединение результатов всех методов...")

    # Нормализуем все скоры от 0 до 1
    rf_importance['rf_norm'] = (rf_importance['rf_importance'] - rf_importance['rf_importance'].min()) / \
                               (rf_importance['rf_importance'].max() - rf_importance['rf_importance'].min())

    f_scores['f_norm'] = (f_scores['f_score'] - f_scores['f_score'].min()) / \
                         (f_scores['f_score'].max() - f_scores['f_score'].min())

    mi_importance['mi_norm'] = (mi_importance['mi_score'] - mi_importance['mi_score'].min()) / \
                               (mi_importance['mi_score'].max() - mi_importance['mi_score'].min())

    # Объединяем все методы
    combined = rf_importance[['feature', 'rf_norm']].merge(
        f_scores[['feature', 'f_norm']], on='feature'
    ).merge(
        mi_importance[['feature', 'mi_norm']], on='feature'
    )

    # Средний рейтинг (можно использовать веса)
    combined['combined_score'] = (
            combined['rf_norm'] * 0.4 +      # RF - 40% (хорошо для важности)
            combined['f_norm'] * 0.3 +       # F-stat - 30% (линейная корреляция)
            combined['mi_norm'] * 0.3        # MI - 30% (нелинейная зависимость)
    )

    combined = combined.sort_values('combined_score', ascending=False)

    # Выбираем топ-K признаков
    top_features = combined.head(top_k)
    selected_feature_names = top_features['feature'].tolist()

    print(f"\n   Отобрано признаков: {len(selected_feature_names)}")

    # ===== ВИЗУАЛИЗАЦИЯ =====
    print("\n5. Создание визуализации...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Random Forest Importance
    top_rf = rf_importance.head(20)
    axes[0, 0].barh(range(len(top_rf)), top_rf['rf_importance'], color='steelblue')
    axes[0, 0].set_yticks(range(len(top_rf)))
    axes[0, 0].set_yticklabels(top_rf['feature'], fontsize=8)
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Random Forest - Топ-20 признаков')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # График 2: F-статистика
    top_f = f_scores.head(20)
    axes[0, 1].barh(range(len(top_f)), top_f['f_score'], color='coral')
    axes[0, 1].set_yticks(range(len(top_f)))
    axes[0, 1].set_yticklabels(top_f['feature'], fontsize=8)
    axes[0, 1].set_xlabel('F-score')
    axes[0, 1].set_title('F-статистика - Топ-20 признаков')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # График 3: Mutual Information
    top_mi = mi_importance.head(20)
    axes[1, 0].barh(range(len(top_mi)), top_mi['mi_score'], color='seagreen')
    axes[1, 0].set_yticks(range(len(top_mi)))
    axes[1, 0].set_yticklabels(top_mi['feature'], fontsize=8)
    axes[1, 0].set_xlabel('MI Score')
    axes[1, 0].set_title('Mutual Information - Топ-20 признаков')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # График 4: Комбинированный рейтинг
    axes[1, 1].barh(range(len(top_features)), top_features['combined_score'], color='purple')
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'], fontsize=9, weight='bold')
    axes[1, 1].set_xlabel('Combined Score')
    axes[1, 1].set_title(f'ОТОБРАННЫЕ Топ-{top_k} признаков (комбинированный рейтинг)', weight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    plt.savefig(sub_folder, 'file_png', dpi=100)
    plt.show()
    plt.close()

    # print(f"   График сохранен: {selection_plot_path}")

    # ===== ВЫВОД РЕЗУЛЬТАТОВ =====
    print("\n" + "="*80)
    print(f"ИТОГОВЫЙ СПИСОК ОТОБРАННЫХ {top_k} ПРИЗНАКОВ")
    print("="*80)

    for i, row in top_features.iterrows():
        print(f"{list(top_features.index).index(i)+1:2d}. {row['feature']:40s} (score: {row['combined_score']:.4f})")

    # Сохранение в CSV
    # combined.to_csv(os.path.join(sub_folder, 'feature_ranking.csv'), index=False, encoding='utf-8-sig')
    # top_features.to_csv(os.path.join(sub_folder, 'selected_features.csv'), index=False, encoding='utf-8-sig')
    #
    # print(f"\nПолный рейтинг сохранен: {os.path.join(sub_folder, 'feature_ranking.csv')}")
    # print(f"Отобранные признаки сохранены: {os.path.join(sub_folder, 'selected_features.csv')}")

    # ===== ВОЗВРАТ ОТФИЛЬТРОВАННЫХ ДАННЫХ =====
    # Находим индексы отобранных признаков
    selected_indices = [list(feature_names).index(f) for f in selected_feature_names]

    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    print(f"\nРазмерность данных после отбора:")
    print(f"   X_train: {X_train_selected.shape}")
    print(f"   X_test: {X_test_selected.shape}")
    print("="*80)

    return X_train_selected, X_test_selected, selected_feature_names


# Пример использования в вашей функции feature_engineering
"""
def feature_engineering(f_contracts, sub_folder):

    # Создаем расширенные признаки
    data_dict = create_enhanced_features(f_contracts, sub_folder)

    X = data_dict['X']
    Y = data_dict['Y']
    use_log = data_dict['use_log']
    test_size = data_dict['test_size']

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=test_size, shuffle=False
    )

    # === ОТБОР ПРИЗНАКОВ ===
    # Определяем количество признаков для отбора (примерно 1 признак на 10-15 наблюдений)
    optimal_k = max(10, min(20, X_train.shape[0] // 15))

    X_train_selected, X_test_selected, selected_features = smart_feature_selection(
        X_train, Y_train, X_test, Y_test, 
        X.columns, sub_folder, top_k=optimal_k
    )

    # Теперь используем отобранные признаки для моделей
    best_model_name, results_df, best_model = compare_regression_models(
        X_train_selected, X_test_selected, Y_train, Y_test, 
        scaler, selected_features, sub_folder
    )

    # ... далее прогнозирование ...
"""
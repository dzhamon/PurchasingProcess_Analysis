import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import os

def compare_regression_models(X_train, X_test, Y_train, Y_test, scaler, X_columns, sub_folder):
    """
    Сравнивает несколько моделей регрессии и выбирает лучшую по R²

    Параметры:
    - X_train, X_test, Y_train, Y_test: данные для обучения и тестирования
    - scaler: обученный StandardScaler для обратного преобразования
    - X_columns: названия признаков
    - sub_folder: папка для сохранения результатов
    """

    # Словарь моделей для тестирования
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=8,
            max_features='sqrt',
            min_samples_leaf=5
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5
        ),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(
            random_state=42,
            max_depth=8,
            min_samples_leaf=5
        ),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }

    results = []
    predictions = {}
    feature_importances = {}

    print("\n" + "="*80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ РЕГРЕССИИ")
    print("="*80)

    # Обучение и оценка каждой модели
    for model_name, model in models.items():
        print(f"\n--- Обучение модели: {model_name} ---")

        try:
            # Обучение модели
            model.fit(X_train, Y_train)

            # Предсказания в log-шкале
            Y_pred_log = model.predict(X_test)
            Y_train_pred_log = model.predict(X_train)

            # Обратное преобразование
            Y_pred = np.expm1(Y_pred_log)
            Y_test_orig = np.expm1(Y_test)
            Y_train_orig = np.expm1(Y_train)
            Y_train_pred = np.expm1(Y_train_pred_log)

            # Метрики
            r2_train = r2_score(Y_train_orig, Y_train_pred)
            r2_test = r2_score(Y_test_orig, Y_pred)
            mse_test = mean_squared_error(Y_test_orig, Y_pred)
            mae_test = mean_absolute_error(Y_test_orig, Y_pred)
            rmse_test = np.sqrt(mse_test)

            # Проверка на переобучение
            overfit_diff = r2_train - r2_test

            print(f"R2 на обучающих данных: {r2_train:.4f}")
            print(f"R2 на тестовых данных: {r2_test:.4f}")
            print(f"MSE: {mse_test:.2f}")
            print(f"RMSE: {rmse_test:.2f}")
            print(f"MAE: {mae_test:.2f}")
            print(f"Разница R2 (train-test): {overfit_diff:.4f}")

            if overfit_diff > 0.3:
                print("ВНИМАНИЕ: Возможное переобучение модели!")

            # Сохранение результатов
            results.append({
                'Model': model_name,
                'R2_train': r2_train,
                'R2_test': r2_test,
                'MSE': mse_test,
                'RMSE': rmse_test,
                'MAE': mae_test,
                'Overfit': overfit_diff
            })

            predictions[model_name] = (Y_test_orig, Y_pred)

            # Сохранение важности признаков (если доступно)
            if hasattr(model, 'feature_importances_'):
                feature_importances[model_name] = pd.DataFrame({
                    'Feature': X_columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            elif hasattr(model, 'coef_'):
                feature_importances[model_name] = pd.DataFrame({
                    'Feature': X_columns,
                    'Coefficient': np.abs(model.coef_)
                }).sort_values(by='Coefficient', ascending=False)

        except Exception as e:
            print(f" Ошибка при обучении модели {model_name}: {str(e)}")
            continue

    # Создание сводной таблицы результатов
    results_df = pd.DataFrame(results).sort_values('R2_train', ascending=False)

    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (отсортировано по R2 на обучающих данных)")
    print("="*80)
    print(results_df.to_string(index=False))

    # Определение лучшей модели
    best_model_name = results_df.iloc[0]['Model']
    best_r2 = results_df.iloc[0]['R2_train']

    print("\n" + "="*80)
    print(f" ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
    print(f"   R2 на обучающих данных: {best_r2:.4f}")
    print("="*80)

    # Визуализация 1: Сравнение R² всех моделей
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(results_df))
    plt.bar(x_pos, results_df['R2_test'], alpha=0.7, color='steelblue', edgecolor='black')
    plt.xticks(x_pos, results_df['Model'], rotation=45, ha='right')
    plt.ylabel('R2 Score')
    plt.title('Сравнение моделей по R2 на тестовых данных')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_plot_path = os.path.join(sub_folder, 'models_comparison.png')
    plt.savefig(comparison_plot_path, dpi=100)
    plt.close()
    print(f"\nГрафик сравнения сохранен: {comparison_plot_path}")

    # Визуализация 2: Факт vs Прогноз для лучшей модели
    if best_model_name in predictions:
        Y_test_orig, Y_pred = predictions[best_model_name]

        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test_orig, Y_pred, alpha=0.6, edgecolor='k', s=80)
        plt.plot([Y_test_orig.min(), Y_test_orig.max()],
                 [Y_test_orig.min(), Y_test_orig.max()], 'r--', lw=2)
        plt.xlabel('Фактические значения', fontsize=12)
        plt.ylabel('Предсказанные значения', fontsize=12)
        plt.title(f'Лучшая модель: {best_model_name} (R2={best_r2:.3f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        best_plot_path = os.path.join(sub_folder, 'best_model_predictions.png')
        plt.savefig(best_plot_path, dpi=100)
        plt.close()
        print(f"График лучшей модели сохранен: {best_plot_path}")

    # Визуализация 3: Важность признаков лучшей модели
    if best_model_name in feature_importances:
        importance_df = feature_importances[best_model_name].head(15)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df.iloc[:, 1], color='coral', edgecolor='black')
        plt.yticks(range(len(importance_df)), importance_df.iloc[:, 0])
        plt.xlabel('Важность / Коэффициент', fontsize=12)
        plt.title(f'Топ-15 признаков ({best_model_name})', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        importance_plot_path = os.path.join(sub_folder, 'feature_importance_best.png')
        plt.savefig(importance_plot_path, dpi=100)
        plt.close()
        print(f"График важности признаков сохранен: {importance_plot_path}")

        print(f"\nВажность признаков ({best_model_name}):")
        print(importance_df.to_string(index=False))

    # Сохранение результатов в CSV
    results_csv_path = os.path.join(sub_folder, 'models_comparison.csv')
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"\nРезультаты сохранены в CSV: {results_csv_path}")

    return best_model_name, results_df, models[best_model_name]

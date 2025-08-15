import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

warnings.filterwarnings("ignore")

# Настройки для графиков
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (12, 8)


def load_and_prepare_data(file_path):
    """Загрузка и подготовка данных"""
    df = pd.read_csv(file_path, encoding="utf-8")
    print(f"Загружено {len(df)} записей")
    print(f"Столбцы: {list(df.columns)}")
    return df


def create_features_v2(df):
    """Создание расширенного набора признаков для кластеризации"""
    features_df = pd.DataFrame()

    # Базовые метрики
    features_df["total_amount"] = df["Сумма договора"]
    features_df["contract_count"] = df.groupby("Поставщик")["Поставщик"].transform(
        "count"
    )

    # Средние значения
    features_df["avg_amount"] = df.groupby("Поставщик")["Сумма договора"].transform(
        "mean"
    )
    features_df["median_amount"] = df.groupby("Поставщик")["Сумма договора"].transform(
        "median"
    )

    # Вариативность
    features_df["amount_std"] = (
        df.groupby("Поставщик")["Сумма договора"].transform("std").fillna(0)
    )
    features_df["amount_cv"] = features_df["amount_std"] / features_df["avg_amount"]
    features_df["amount_cv"] = (
        features_df["amount_cv"].fillna(0).replace([np.inf, -np.inf], 0)
    )

    # Логарифмические преобразования для нормализации распределения
    features_df["log_total_amount"] = np.log1p(features_df["total_amount"])
    features_df["log_avg_amount"] = np.log1p(features_df["avg_amount"])
    features_df["log_contract_count"] = np.log1p(features_df["contract_count"])

    # Квантильные признаки
    features_df["amount_percentile"] = (
        df.groupby("Поставщик")["Сумма договора"]
        .transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
        .fillna(0)
    )

    # Добавим поставщика для группировки
    features_df["supplier"] = df["Поставщик"]

    # Группируем по поставщику и берем первое значение (так как все одинаковые для каждого поставщика)
    supplier_features = features_df.groupby("supplier").first().reset_index()

    print(f"Создано {len(supplier_features)} уникальных поставщиков")
    print(f"Признаки: {list(supplier_features.columns[1:])}")

    return supplier_features


def detect_outliers_iqr(data, columns, threshold=3):
    """Обнаружение выбросов методом IQR"""
    outlier_indices = set()

    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
        outlier_indices.update(outliers)

        print(
            f"{col}: выбросы {len(outliers)} из {len(data)} ({len(outliers)/len(data)*100:.1f}%)"
        )

    return list(outlier_indices)


def try_multiple_scalers(X, feature_columns):
    """Тестирование разных методов масштабирования"""
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }

    scaler_results = {}

    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)

        # Тестируем разное количество кластеров
        best_score = -1
        best_k = 2

        for k in range(2, min(11, len(X) // 10)):
            if len(X) >= k:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                if len(np.unique(labels)) > 1:  # Проверяем, что есть разные кластеры
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

        scaler_results[name] = {
            "scaler": scaler,
            "best_k": best_k,
            "best_score": best_score,
            "X_scaled": scaler.fit_transform(X),
        }

        print(f"{name}: лучший результат k={best_k}, silhouette={best_score:.3f}")

    # Выбираем лучший скейлер
    best_scaler_name = max(
        scaler_results.keys(), key=lambda x: scaler_results[x]["best_score"]
    )
    return scaler_results[best_scaler_name], best_scaler_name


def advanced_clustering_analysis(supplier_features):
    """Продвинутый кластерный анализ с множественными подходами"""

    # Подготовка признаков (исключаем поставщика)
    feature_columns = [col for col in supplier_features.columns if col != "supplier"]
    X = supplier_features[feature_columns].values

    print("\n=== АНАЛИЗ ДАННЫХ ===")
    print(f"Размер данных: {X.shape}")
    print("\nСтатистика по признакам:")
    print(supplier_features[feature_columns].describe())

    # Обнаружение и анализ выбросов
    print("\n=== АНАЛИЗ ВЫБРОСОВ ===")
    outliers = detect_outliers_iqr(supplier_features, feature_columns, threshold=2.5)
    print(
        f"Обнаружено {len(outliers)} выбросов из {len(supplier_features)} ({len(outliers)/len(supplier_features)*100:.1f}%)"
    )

    # Опционально: создаем версию без выбросов
    X_no_outliers = np.delete(X, outliers, axis=0) if outliers else X
    suppliers_no_outliers = (
        supplier_features.drop(outliers) if outliers else supplier_features
    )

    print(f"Данные без выбросов: {X_no_outliers.shape}")

    # Тестируем разные скейлеры
    print("\n=== ТЕСТИРОВАНИЕ СКЕЙЛЕРОВ ===")
    best_scaler_result, best_scaler_name = try_multiple_scalers(
        X_no_outliers, feature_columns
    )
    X_scaled = best_scaler_result["X_scaled"]

    print(f"Выбран скейлер: {best_scaler_name}")

    # Применяем PCA для снижения размерности
    print("\n=== PCA АНАЛИЗ ===")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Определяем оптимальное количество компонент (95% дисперсии)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1

    print(f"Компонент для 95% дисперсии: {n_components_95}")
    print(f"Объясненная дисперсия по компонентам: {pca.explained_variance_ratio_[:5]}")

    # Используем оптимальное количество компонент
    pca_optimal = PCA(n_components=min(n_components_95, X_scaled.shape[1]))
    X_pca_optimal = pca_optimal.fit_transform(X_scaled)

    # Тестируем множественные алгоритмы кластеризации
    print("\n=== ТЕСТИРОВАНИЕ АЛГОРИТМОВ КЛАСТЕРИЗАЦИИ ===")

    clustering_results = {}

    # 1. K-Means с разным количеством кластеров
    print("Тестируем K-Means...")
    kmeans_scores = []
    k_range = range(2, min(21, len(X_pca_optimal) // 5))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca_optimal)

        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X_pca_optimal, labels)
            ch_score = calinski_harabasz_score(X_pca_optimal, labels)

            kmeans_scores.append(
                {
                    "k": k,
                    "silhouette": silhouette_avg,
                    "calinski_harabasz": ch_score,
                    "labels": labels,
                    "model": kmeans,
                }
            )

    # Выбираем оптимальное k для K-Means
    best_kmeans = max(kmeans_scores, key=lambda x: x["silhouette"])
    clustering_results["KMeans"] = best_kmeans

    # 2. Agglomerative Clustering
    print("Тестируем Agglomerative Clustering...")
    agg_scores = []

    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X_pca_optimal)

        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X_pca_optimal, labels)
            ch_score = calinski_harabasz_score(X_pca_optimal, labels)

            agg_scores.append(
                {
                    "k": k,
                    "silhouette": silhouette_avg,
                    "calinski_harabasz": ch_score,
                    "labels": labels,
                    "model": agg,
                }
            )

    if agg_scores:
        best_agg = max(agg_scores, key=lambda x: x["silhouette"])
        clustering_results["Agglomerative"] = best_agg

    # 3. DBSCAN с разными параметрами
    print("Тестируем DBSCAN...")
    eps_values = np.linspace(0.1, 2.0, 20)
    min_samples_values = [3, 5, 10, 15]

    dbscan_scores = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_pca_optimal)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters > 1 and n_noise < len(labels) * 0.5:  # Не слишком много шума
                try:
                    silhouette_avg = silhouette_score(X_pca_optimal, labels)
                    ch_score = calinski_harabasz_score(X_pca_optimal, labels)

                    dbscan_scores.append(
                        {
                            "eps": eps,
                            "min_samples": min_samples,
                            "n_clusters": n_clusters,
                            "n_noise": n_noise,
                            "silhouette": silhouette_avg,
                            "calinski_harabasz": ch_score,
                            "labels": labels,
                            "model": dbscan,
                        }
                    )
                except:
                    continue

    if dbscan_scores:
        best_dbscan = max(dbscan_scores, key=lambda x: x["silhouette"])
        clustering_results["DBSCAN"] = best_dbscan

    # Выводим результаты
    print("\n=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ===")
    for method, result in clustering_results.items():
        if method == "DBSCAN":
            print(
                f"{method}: кластеров={result['n_clusters']}, шум={result['n_noise']}, "
                f"silhouette={result['silhouette']:.3f}"
            )
        else:
            print(f"{method}: k={result['k']}, silhouette={result['silhouette']:.3f}")

    # Выбираем лучший метод
    best_method = max(
        clustering_results.keys(), key=lambda x: clustering_results[x]["silhouette"]
    )
    best_result = clustering_results[best_method]

    print(
        f"\nЛучший метод: {best_method} с silhouette score = {best_result['silhouette']:.3f}"
    )

    # Создаем финальную визуализацию
    create_comprehensive_visualization(
        X_pca_optimal,
        X_scaled,
        suppliers_no_outliers,
        best_result,
        best_method,
        pca_optimal,
        feature_columns,
    )

    return {
        "best_method": best_method,
        "best_result": best_result,
        "suppliers_data": suppliers_no_outliers,
        "X_scaled": X_scaled,
        "X_pca": X_pca_optimal,
        "pca_model": pca_optimal,
        "scaler": best_scaler_result["scaler"],
        "feature_columns": feature_columns,
        "all_results": clustering_results,
    }


def create_comprehensive_visualization(
    X_pca, X_scaled, suppliers_df, best_result, method_name, pca_model, feature_columns
):
    """Создание комплексной визуализации результатов"""

    labels = best_result["labels"]

    # Создаем фигуру с субплотами
    fig = plt.figure(figsize=(20, 15))

    # 1. PCA визуализация кластеров
    ax1 = plt.subplot(2, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)")
    plt.title(f"Кластеры в пространстве PCA ({method_name})")

    # 2. Распределение размеров кластеров
    ax2 = plt.subplot(2, 3, 2)
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    bars = plt.bar(range(len(unique_labels)), counts, color=colors)
    plt.xlabel("Кластер")
    plt.ylabel("Количество поставщиков")
    plt.title("Размеры кластеров")
    plt.xticks(
        range(len(unique_labels)),
        [f"Кластер {l}" if l != -1 else "Шум" for l in unique_labels],
    )

    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{count}",
            ha="center",
            va="bottom",
        )

    # 3. Heatmap средних значений признаков по кластерам
    ax3 = plt.subplot(2, 3, 3)

    # Добавляем кластерные метки к данным
    cluster_data = suppliers_df.copy()
    cluster_data["cluster"] = labels

    # Вычисляем средние значения по кластерам
    cluster_means = cluster_data.groupby("cluster")[feature_columns].mean()

    # Нормализуем для лучшей визуализации
    from sklearn.preprocessing import StandardScaler

    scaler_viz = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler_viz.fit_transform(cluster_means.T).T,
        index=cluster_means.index,
        columns=cluster_means.columns,
    )

    sns.heatmap(
        cluster_means_scaled,
        annot=True,
        cmap="RdYlBu_r",
        center=0,
        fmt=".2f",
        cbar_kws={"label": "Нормализованное значение"},
    )
    plt.title("Профили кластеров (средние значения)")
    plt.ylabel("Кластер")

    # 4. Объясненная дисперсия PCA
    ax4 = plt.subplot(2, 3, 4)
    explained_var = pca_model.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)

    plt.plot(
        range(1, len(explained_var) + 1), explained_var, "bo-", label="Индивидуальная"
    )
    plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, "ro-", label="Кумулятивная")
    plt.xlabel("Компонента")
    plt.ylabel("Объясненная дисперсия")
    plt.title("Объясненная дисперсия PCA")
    plt.legend()
    plt.grid(True)

    # 5. Детальная статистика по кластерам
    ax5 = plt.subplot(2, 3, 5)

    # Создаем таблицу со статистикой
    stats_data = []
    for cluster_id in unique_labels:
        if cluster_id != -1:  # Исключаем шум для DBSCAN
            cluster_mask = labels == cluster_id
            cluster_suppliers = suppliers_df[cluster_mask]

            stats_data.append(
                {
                    "Кластер": cluster_id,
                    "Размер": sum(cluster_mask),
                    "Ср. сумма": cluster_suppliers["avg_amount"].mean(),
                    "Ср. кол-во\nконтрактов": cluster_suppliers[
                        "contract_count"
                    ].mean(),
                    "Общая\nсумма": cluster_suppliers["total_amount"].sum(),
                }
            )

    stats_df = pd.DataFrame(stats_data)

    # Создаем таблицу
    ax5.axis("tight")
    ax5.axis("off")
    table = ax5.table(
        cellText=stats_df.round(2).values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    plt.title("Статистика по кластерам", y=0.8)

    # 6. Топ поставщики по кластерам
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    top_suppliers_text = "Топ-3 поставщика по кластерам:\n\n"

    for cluster_id in sorted(unique_labels):
        if cluster_id != -1:
            cluster_mask = labels == cluster_id
            cluster_suppliers = suppliers_df[cluster_mask]
            top_3 = cluster_suppliers.nlargest(3, "total_amount")

            top_suppliers_text += (
                f"Кластер {cluster_id} ({sum(cluster_mask)} поставщиков):\n"
            )
            for idx, (_, supplier) in enumerate(top_3.iterrows(), 1):
                supplier_name = (
                    supplier["supplier"][:30] + "..."
                    if len(supplier["supplier"]) > 30
                    else supplier["supplier"]
                )
                top_suppliers_text += f"  {idx}. {supplier_name}\n"
                top_suppliers_text += f"     Сумма: {supplier['total_amount']:,.0f}\n"
            top_suppliers_text += "\n"

    plt.text(
        0.05,
        0.95,
        top_suppliers_text,
        transform=ax6.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.show()

    return cluster_data


def main():
    """Основная функция для запуска анализа"""
    print("=== УЛУЧШЕННЫЙ КЛАСТЕРНЫЙ АНАЛИЗ ПОСТАВЩИКОВ v13 ===\n")

    # Загрузка данных
    file_path = input("Введите путь к CSV файлу: ")
    df = load_and_prepare_data(file_path)

    # Создание признаков
    supplier_features = create_features_v2(df)

    # Запуск анализа
    results = advanced_clustering_analysis(supplier_features)

    # Сохранение результатов
    final_data = results["suppliers_data"].copy()
    final_data["cluster"] = results["best_result"]["labels"]

    output_file = file_path.replace(".csv", "_clusters_v13.csv")
    final_data.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\nРезультаты сохранены в: {output_file}")

    # Финальная сводка
    print("\n=== ФИНАЛЬНАЯ СВОДКА ===")
    print(f"Лучший метод: {results['best_method']}")
    print(f"Silhouette Score: {results['best_result']['silhouette']:.3f}")

    labels = results["best_result"]["labels"]
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(
        f"Количество кластеров: {len(unique_labels) - (1 if -1 in unique_labels else 0)}"
    )
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  Шум: {count} поставщиков")
        else:
            print(f"  Кластер {label}: {count} поставщиков")

    return results


if __name__ == "__main__":
    results = main()

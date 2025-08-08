import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from utils.functions import clean_filename
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem


def create_plot_graf(good_name, filtered_data, output_folder):
    print("\n--- Начало создания графика ---")
    print(f"Товар: {good_name}")
    print(f"Данные:\n{filtered_data[['winner_name', 'avg_unit_price']]}")

    try:
        # 1. Проверка данных
        if filtered_data.empty:
            raise ValueError("Пустой DataFrame")

        if (
            "winner_name" not in filtered_data.columns
            or "avg_unit_price" not in filtered_data.columns
        ):
            raise ValueError("Отсутствуют необходимые столбцы")

        # 2. Очистка данных
        plot_data = filtered_data.dropna(
            subset=["winner_name", "avg_unit_price"]
        ).copy()
        plot_data["avg_unit_price"] = pd.to_numeric(
            plot_data["avg_unit_price"], errors="coerce"
        )
        plot_data = plot_data.dropna(subset=["avg_unit_price"])

        if plot_data.empty:
            raise ValueError("Нет валидных данных после очистки")

        # 3. Подготовка пути
        cleaned_good_name = re.sub(r'[\\/*?:"<>|]', "_", good_name)
        os.makedirs(output_folder, exist_ok=True)
        png_file = os.path.join(
            output_folder, f"{cleaned_good_name}_price_analysis.png"
        )
        print(f"Путь для сохранения: {png_file}")

        # 4. Создание графика
        plt.figure(figsize=(12, 6))
        bars = plt.bar(
            x=plot_data["winner_name"],
            height=plot_data["avg_unit_price"],
            color="skyblue",
        )

        # Добавление значений на столбцы
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        plt.title(f"Средняя цена за единицу: {good_name[:50]}...", pad=20)
        plt.xlabel("Поставщик")
        plt.ylabel("Цена (EUR)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # 5. Сохранение и отображение
        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"!!! Ошибка при создании графика для '{good_name}': {str(e)}")
    return


def plot_bar_chart(x, y, title, x_label, y_label, output_file):
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def preprocess_supplier_names(supplier_names):
    supplier_names = supplier_names.str.strip()
    supplier_names = supplier_names.str.replace(
        r"^Не использовать дубль\s*", "", regex=True
    )
    supplier_names = supplier_names.str[:15]

    return supplier_names


def save_top_suppliers_bar_chart(top_suppliers, currency, interval_text, output_dir):
    """
    Создание и сохранение графиков топ-10 поставщиков
    """
    top_suppliers.index = preprocess_supplier_names(top_suppliers.index)

    fig, ax = plt.subplots(figsize=(12, 8))
    top_suppliers.plot(kind="bar", color="skyblue", ax=ax)

    ax.set_xticks(range(len(top_suppliers)))
    ax.set_xticklabels(top_suppliers.index, rotation=30)

    ax.set_title(
        f"Top-10 Suppliers by Total Costs for {interval_text} (Currency: {currency})"
    )
    ax.set_xlabel("Supplier")
    ax.set_ylabel(f"Total Costs ({currency})")

    for i, v in enumerate(top_suppliers):
        ax.text(i, v + 0.07 * v, f"{v:,.0f}", ha="center", va="bottom")

    ax.grid(axis="y")

    file_path = os.path.join(output_dir, f"top_suppliers_{currency}.png")
    plt.savefig(file_path)
    plt.close()
    gc.collect()


def visualize_price_differences(df):
    import matplotlib.pyplot as plt
    import numpy as np

    output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
    os.makedirs(output_folder, exist_ok=True)
    unique_disciplines = df["discipline1"].unique()
    for discipline in unique_disciplines:
        filtered_df = df[df["discipline1"] == discipline]
        materials = filtered_df["good_name"]
        prices_discipline1 = filtered_df["price_discipline1"]
        prices_discipline2 = filtered_df["price_discipline2"]

        x = np.arange(len(materials))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(12, 8))

        bar1 = ax.bar(
            x - bar_width / 2,
            prices_discipline1,
            width=bar_width,
            label="Discipline 1",
            color="skyblue",
        )
        bar2 = ax.bar(
            x + bar_width / 2,
            prices_discipline2,
            width=bar_width,
            label="Discipline 2",
            color="orange",
        )

        ax.set_xlabel("Materials", fontsize=12)
        ax.set_ylabel("Unit Price", fontsize=12)
        ax.set_title(
            f"Comparison of Unit Prices for Discipline: {discipline}", fontsize=16
        )
        ax.set_xticks(x)
        ax.set_xticklabels(materials, rotation=45, ha="right", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()

        plt.tight_layout()
        output_file = os.path.join(output_folder, f"{discipline}_price_comparison.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

    print(f"Графики успешно сохранены в папку: {output_folder}")

    return


def heatmap_common_suppliers(comparison_results):
    print("Запускается метод heatmap_common_suppliers")

    output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
    os.makedirs(output_folder, exist_ok=True)

    disciplines = set(comparison_results["discipline1"]).union(
        set(comparison_results["discipline2"])
    )
    heatmap_data = pd.DataFrame(
        index=list(disciplines), columns=list(disciplines), dtype=float
    ).fillna(0)

    for _, row in comparison_results.iterrows():
        discip1, discip2 = row["discipline1"], row["discipline2"]
        heatmap_data.at[discip1, discip2] += 1
        heatmap_data.at[discip2, discip1] += 1

    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        fmt="g",
        cbar=True,
        square=True,
        linewidths=0.5,
    )

    plt.title("Количество общих товаров между дисциплинами", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=2.0)

    output_file = os.path.join(output_folder, "heatmap_common_suppliers.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Тепловая карта успешно сохранена в файл: {output_file}")

    return


def visualize_isolation_forest(analyzed_df):
    if (
        "unit_price_in_eur" not in analyzed_df.columns
        or "total_price_in_eur" not in analyzed_df.columns
    ):
        print(
            "Для визуализации нужны столбцы 'unit_price_in_eur' и 'total_price_in_eur'"
        )
        return

    data = analyzed_df[["unit_price_in_eur", "total_price_in_eur"]].dropna()

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)
    data["is_anomaly"] = model.predict(data)

    xx, yy = np.meshgrid(
        np.linspace(
            data["unit_price_in_eur"].min(), data["unit_price_in_eur"].max(), 100
        ),
        np.linspace(
            data["total_price_in_eur"].min(), data["total_price_in_eur"].max(), 100
        ),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid_points).reshape(xx.shape)

    output_dir = r"D:\Analysis-Results\efficient_analyses"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "isolation_forest_visualization.png")

    plt.figure(figsize=(10, 8))

    plt.contourf(xx, yy, scores, levels=50, cmap=plt.cm.RdYlBu_r, alpha=0.6)

    plt.scatter(
        data["unit_price_in_eur"],
        data["total_price_in_eur"],
        c=data["is_anomaly"].map({1: "blue", -1: "red"}),
        edgecolors="k",
        alpha=0.8,
        label="Data Points",
    )

    plt.title("Isolation Forest - Visualization of Anomalies")
    plt.xlabel("Unit Price (EUR)")
    plt.ylabel("Total Price (EUR)")
    plt.colorbar(label="Anomaly Score")
    plt.legend(["Normal", "Anomalies"], loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file_path, format="png", dpi=300)
    plt.close()

    print(f"График успешно сохранён в файл: {output_file_path}")


def save_herfind_hirshman_results(supplier_stats, hhi):
    """
    Сохраняет результаты анализа Херфиндаля-Хиршмана
    
    Args:
        supplier_stats (pd.DataFrame): Статистика по поставщикам
        hhi (pd.DataFrame): Индексы HHI по дисциплинам
    
    Returns:
        bool: True если сохранение прошло успешно, False в противном случае
    """
    
    print("--- Запуск сохранения результатов Херфиндаля-Хиршмана ---")
    
    # Создаем папку для результатов
    output_dir = r"D:\Analysis-Results\hirshman_results"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Ошибка создания папки {output_dir}: {e}")
        return False
    
    all_major_suppliers = pd.DataFrame()
    
    try:
        # Проверяем входные данные
        if supplier_stats is None or supplier_stats.empty:
            print("Ошибка: supplier_stats пуст или равен None")
            return False
    
        if hhi is None or hhi.empty:
            print("Ошибка: hhi пуст или равен None")
            return False
    
        print(f"Обрабатываем данные: supplier_stats.shape = {supplier_stats.shape}")
        print(f"Столбцы в supplier_stats: {list(supplier_stats.columns)}")
    
        # Проверяем наличие необходимых столбцов
        required_columns = ["discipline", "counterparty_name", "supp_share"]
        missing_columns = [
            col for col in required_columns if col not in supplier_stats.columns
        ]
    
        if missing_columns:
            print(f"Отсутствуют необходимые столбцы в supplier_stats: {missing_columns}")
            return False
    
        # Генерация круговых диаграмм для каждой дисциплины
        disciplines = supplier_stats["discipline"].unique()
        print(f"Найдено дисциплин: {len(disciplines)}")
    
        for discipline in disciplines:
            try:
                print(f"Обрабатываем дисциплину: {discipline}")
    
                filtered = supplier_stats[supplier_stats["discipline"] == discipline].copy()
    
                if filtered.empty:
                    print(f"Нет данных для дисциплины: {discipline}")
                    continue
    
                # Фильтруем крупных поставщиков (доля >= 8%)
                major_suppliers = filtered[filtered["supp_share"] >= 8].copy()
                major_suppliers["discipline"] = discipline
    
                all_major_suppliers = pd.concat(
                    [all_major_suppliers, major_suppliers], ignore_index=True
                )
    
                # Вычисляем долю остальных поставщиков
                other_suppliers_share = filtered[filtered["supp_share"] < 8][
                    "supp_share"
                ].sum()
    
                # Подготавливаем данные для круговой диаграммы
                labels = list(major_suppliers["counterparty_name"])
                sizes = list(major_suppliers["supp_share"])
    
                if other_suppliers_share > 0:
                    labels.append("Другие")
                    sizes.append(other_suppliers_share)
    
                if not sizes:  # Если нет данных для диаграммы
                    print(f"Нет данных для создания диаграммы для дисциплины: {discipline}")
                    continue
    
                # Создаем круговую диаграмму
                plt.figure(figsize=(10, 8))
                colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    
                wedges, texts, autotexts = plt.pie(
                    sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors
                )
    
                plt.title(
                    f"Доли поставщиков для дисциплины: {discipline}", fontsize=14, pad=20
                )
    
                # Улучшаем читаемость
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontsize(9)
                    autotext.set_weight("bold")
    
                plt.tight_layout()
    
                # Сохраняем диаграмму
                pie_chart_path = os.path.join(
                    output_dir,
                    f"{clean_filename(str(discipline))}_supplier_shares_pie_chart.png",
                )
                plt.savefig(pie_chart_path, dpi=300, bbox_inches="tight")
                plt.close()
    
                print(f"Диаграмма сохранена: {pie_chart_path}")
    
            except Exception as e:
                print(f"Ошибка при создании диаграммы для дисциплины {discipline}: {e}")
                plt.close()  # Закрываем фигуру в случае ошибки
                continue
    
        # Сохранение результатов HHI в текстовый файл
        hhi_results_path = os.path.join(output_dir, "herfindahl_hirshman_index_results.txt")
        try:
            with open(hhi_results_path, "w", encoding="utf-8") as f:
                f.write("=== РЕЗУЛЬТАТЫ АНАЛИЗА ИНДЕКСА ХЕРФИНДАЛЯ-ХИРШМАНА ===\n\n")
                f.write("Индекс HHI по дисциплинам:\n")
                f.write("-" * 50 + "\n")
                f.write(hhi.to_string(index=False))
                f.write("\n\n")
    
                # Добавляем интерпретацию результатов
                f.write("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:\n")
                f.write("-" * 30 + "\n")
                f.write("HHI < 1500  - Низкая концентрация (конкурентный рынок)\n")
                f.write("1500 <= HHI < 2500 - Умеренная концентрация\n")
                f.write("HHI >= 2500 - Высокая концентрация (олигополия/монополия)\n\n")
    
                if "hhi_index" in hhi.columns:
                    for _, row in hhi.iterrows():
                        discipline = row.get("discipline", "Неизвестно")
                        hhi_val = row.get("hhi_index", 0)
    
                        if hhi_val < 1500:
                            interpretation = "Конкурентный рынок"
                        elif hhi_val < 2500:
                            interpretation = "Умеренная концентрация"
                        else:
                            interpretation = "Высокая концентрация"
    
                        f.write(f"{discipline}: {hhi_val:.2f} - {interpretation}\n")
    
            print(f"HHI результаты сохранены: {hhi_results_path}")
        except Exception as e:
            print(f"Ошибка при сохранении HHI результатов: {e}")
    
        # Сохранение крупных поставщиков в Excel
        if not all_major_suppliers.empty:
            major_suppliers_path = os.path.join(output_dir, "all_major_suppliers.xlsx")
            try:
                all_major_suppliers.to_excel(major_suppliers_path, index=False)
                print(f"Крупные поставщики сохранены: {major_suppliers_path}")
            except Exception as e:
                print(f"Ошибка при сохранении крупных поставщиков: {e}")
    
        # Создаем сводную диаграмму HHI по дисциплинам
        try:
            if "discipline" in hhi.columns and "hhi_index" in hhi.columns:
                plt.figure(figsize=(12, 6))
    
                bars = plt.bar(
                    hhi["discipline"],
                    hhi["hhi_index"],
                    color=[
                        "red" if x >= 2500 else "orange" if x >= 1500 else "green"
                        for x in hhi["hhi_index"]
                    ],
                )
    
                plt.axhline(
                    y=1500,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="Умеренная концентрация (1500)",
                )
                plt.axhline(
                    y=2500,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Высокая концентрация (2500)",
                )
    
                plt.title("Индекс Херфиндаля-Хиршмана по дисциплинам", fontsize=14)
                plt.xlabel("Дисциплина")
                plt.ylabel("Значение HHI")
                plt.xticks(rotation=45, ha="right")
                plt.legend()
                plt.tight_layout()
    
                # Добавляем значения на столбцы
                for bar, value in zip(bars, hhi["hhi_index"]):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 50,
                        f"{value:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
    
                hhi_chart_path = os.path.join(output_dir, "hhi_summary_chart.png")
                plt.savefig(hhi_chart_path, dpi=300, bbox_inches="tight")
                plt.close()
    
                print(f"Сводная диаграмма HHI сохранена: {hhi_chart_path}")
        except Exception as e:
            print(f"Ошибка при создании сводной диаграммы HHI: {e}")
    
        print("Результаты анализа Херфиндаля-Хиршмана успешно сохранены.")
        return True
    
    except Exception as e:
        print(f"Общая ошибка при сохранении результатов Херфиндаля-Хиршмана: {e}")
        return False


def cluster_results_visualize(df_clusters, efficiency_df):
    print("--- Дополнительная проверка данных для объединения ---")

    print(
        "\nТипы данных в df_clusters['cluster_label']:",
        df_clusters["cluster_label"].dtype,
    )
    print(
        "Типы данных в efficiency_df['original_cluster']:",
        efficiency_df["original_cluster"].dtype,
    )

    if df_clusters["cluster_label"].dtype != efficiency_df["original_cluster"].dtype:
        print("\nТипы данных различаются. Попытка преобразовать их в 'int'.")
        try:
            df_clusters["cluster_label"] = df_clusters["cluster_label"].astype(int)
            efficiency_df["original_cluster"] = efficiency_df[
                "original_cluster"
            ].astype(int)
            print("Преобразование типов данных выполнено успешно.")
        except ValueError as e:
            print(f"Ошибка при преобразовании типов данных: {e}")
            print(
                "Проверьте, нет ли нечисловых значений в 'cluster_label' или 'original_cluster'."
            )

    print("\nУникальные cluster_label в df_clusters (первые 20):")
    print(df_clusters["cluster_label"].unique()[:20])

    print("\nУникальные original_cluster в efficiency_df (первые 20):")
    print(efficiency_df["original_cluster"].unique()[:20])

    missing_in_efficiency = set(df_clusters["cluster_label"].unique()) - set(
        efficiency_df["original_cluster"].unique()
    )
    missing_in_clusters = set(efficiency_df["original_cluster"].unique()) - set(
        df_clusters["cluster_label"].unique()
    )

    if missing_in_efficiency:
        print(
            f"\nCluster_label из df_clusters, которых нет в efficiency_df['original_cluster']: {missing_in_efficiency}"
        )
    if missing_in_clusters:
        print(
            f"\nOriginal_cluster из efficiency_df, которых нет в df_clusters['cluster_label']: {missing_in_clusters}"
        )

    print("--- Конец дополнительной проверки ---")

    cluster_efficiency = pd.merge(
        df_clusters[["cluster_label", "actor_name"]],
        efficiency_df,
        left_on="cluster_label",
        right_on="original_cluster",
        how="inner",
    )

    print("\n--- Проверка объединенного DataFrame после исправления ---")
    print(
        "Первые 5 строк объединенного DataFrame (cluster_efficiency) после исправления:"
    )
    print(cluster_efficiency.head())
    print("Уникальные группы эффективности в объединенном DataFrame после исправления:")
    print(cluster_efficiency["efficiency_group"].unique())
    print("Количество NaN значений в efficiency_group после исправления:")
    print(cluster_efficiency["efficiency_group"].isnull().sum())
    print("Количество столбцов в cluster_efficiency_counts после исправления:")
    cluster_efficiency_counts = (
        cluster_efficiency.groupby(["cluster_label", "efficiency_group"])["actor_name"]
        .nunique()
        .unstack(fill_value=0)
    )
    print(cluster_efficiency_counts.shape[1])
    print("--- Конец проверки исправленного DataFrame ---")

    cluster_counts = (
        df_clusters.groupby("cluster_label")["actor_name"].nunique().sort_index()
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
    plt.title("Распределение количества исполнителей по кластерам")
    plt.xlabel("Номер кластера")
    plt.ylabel("Количество исполнителей")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "D:\\Analysis-Results\\clussification_analysis", "cluster_distribution.png"
        )
    )
    plt.close()

    if not cluster_efficiency_counts.empty and cluster_efficiency_counts.shape[1] > 1:
        cluster_efficiency_counts.plot(
            kind="bar", stacked=True, figsize=(12, 7), colormap="coolwarm"
        )
        plt.title("Распределение групп эффективности внутри кластеров")
        plt.xlabel("Номер кластера")
        plt.ylabel("Количество исполнителей")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Группа эффективности")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "D:\\Analysis-Results\\clussification_analysis",
                "cluster_efficiency_distribution.png",
            )
        )
        plt.close()
    else:
        print(
            "Недостаточно данных для стекированной столбчатой диаграммы по группам эффективности."
        )

    efficiency_group_counts = efficiency_df["efficiency_group"].value_counts()
    if not efficiency_group_counts.empty and len(efficiency_group_counts) > 1:
        plt.figure(figsize=(8, 8))
        plt.pie(
            efficiency_group_counts,
            labels=efficiency_group_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("pastel"),
        )
        plt.title("Распределение кластеров по группам эффективности")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "D:\\Analysis-Results\\clussification_analysis",
                "cluster_efficiency_pie_chart.png",
            )
        )
        plt.close()
    else:
        print("Недостаточно групп эффективности для круговой диаграммы.")

    cluster_distribution_by_efficiency = efficiency_df.groupby("efficiency_group")[
        "original_cluster"
    ].nunique()
    if (
        not cluster_distribution_by_efficiency.empty
        and len(cluster_distribution_by_efficiency) > 1
    ):
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=cluster_distribution_by_efficiency.index,
            y=cluster_distribution_by_efficiency.values,
            palette="viridis",
        )
        plt.title("Распределение кластеров по группам эффективности")
        plt.xlabel("Группа эффективности")
        plt.ylabel("Количество кластеров")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "D:\\Analysis-Results\\clussification_analysis",
                "cluster_distribution_by_efficiency.png",
            )
        )
        plt.close()
    else:
        print(
            "Недостаточно групп эффективности для столбчатой диаграммы распределения кластеров."
        )

    merged_df = pd.merge(
        df_clusters,
        efficiency_df,
        left_on="cluster_label",
        right_on="original_cluster",
        how="inner",
    )
    if not merged_df.empty:
        actors_per_cluster = (
            merged_df.groupby(["original_cluster", "efficiency_group"])["actor_name"]
            .nunique()
            .reset_index()
        )
        if not actors_per_cluster.empty:
            avg_actors_per_cluster_in_efficiency_group = actors_per_cluster.groupby(
                "efficiency_group"
            )["actor_name"].mean()
            if (
                not avg_actors_per_cluster_in_efficiency_group.empty
                and len(avg_actors_per_cluster_in_efficiency_group) > 1
            ):
                plt.figure(figsize=(8, 6))
                sns.barplot(
                    x=avg_actors_per_cluster_in_efficiency_group.index,
                    y=avg_actors_per_cluster_in_efficiency_group.values,
                    palette="plasma",
                )
                plt.title(
                    "Среднее количество исполнителей в кластерах по группам эффективности"
                )
                plt.xlabel("Группа эффективности")
                plt.ylabel("Среднее количество исполнителей")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        "D:\\Analysis-Results\\clussification_analysis",
                        "avg_actors_per_cluster.png",
                    )
                )
                plt.close()
            else:
                print(
                    "Недостаточно данных для столбчатой диаграммы среднего количества исполнителей по группам эффективности."
                )
        else:
            print("После агрегации 'actors_per_cluster' пуст.")
    else:
        print(
            "Объединенный DataFrame (merged_df) для среднего количества исполнителей пуст."
        )

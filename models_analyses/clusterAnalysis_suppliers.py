"""
    Версия v2 кластерного анализа. Увеличено количество позазателей для классификации
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")


class EnhancedSupplierClusterAnalyzer:
    def __init__(self, df, column_mapping=None):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.clusters = None
        self.cluster_centers = None

    def prepare_enhanced_supplier_features(self):
        """
        Расширенная подготовка признаков для кластеризации поставщиков
        """
        print(f"🔄 Обрабатываем {len(self.df)} записей...")
        print(f"📊 Уникальных поставщиков: {self.df['counterparty_name'].nunique()}")
        
        # создаем копию данных для обработки
        df_clean = self.df.copy()
        
        # преобразуем числовые колонки принудительно в правильный тип
        numeric_columns = [
            "total_contract_amount_eur", "unit_price_eur", "quantity", "delivery_time_days",
            "product_amount", "additional_expenses"
        ]
        # optional_numeric = ['product_amount', 'additional_expenses']

        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                print(f"✅ Преобразована опциональная колонка {col}, NaN значений: {df_clean[col].isna().sum()}")
                
    
        # Заполняем NaN значения нулями для численных колонок
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)
        if "product_amount" in df_clean.columns:
            df_clean["product_amount"] = df_clean["product_amount"].fillna(0)
        if "additional_expenses" in df_clean.columns:
            df_clean["additional_expenses"] = df_clean["additional_expenses"].fillna(0)
        
        # Создаем словарь агрегации динамически
        agg_dict = {
            # Финансовые показатели
            "total_contract_amount_eur": ["sum", "mean", "std", "count", "min", "max"],
            "unit_price_eur": ["mean", "std", "min", "max"],
            # Временные показатели
            "contract_signing_date": ["min", "max"],
            "delivery_time_days": ["mean", "std", "min", "max"],
            # Объемные показатели
            "quantity": ["sum", "mean", "std", "min", "max"],
            # Разнообразие
            "project_name": "nunique",
            "product_name": "nunique",
            "unit": "nunique",
        }
        
        # Добавляем в словарь опциональные колонки если они есть
        if "product_amount" in df_clean.columns:
            agg_dict["product_amount"] = ["sum", "mean", "std"]
        if "additional_expenses" in df_clean.columns:
            agg_dict["additional_expenses"] = ["sum", "mean"]
        if "discipline" in df_clean.columns:
            agg_dict["discipline"] = "nunique"
        if "contract_currency" in df_clean.columns:
            agg_dict["contract_currency"] = "nunique"


        # Основные агрегации по поставщикам
        supplier_stats = (df_clean.groupby('counterparty_name')
                    .agg(agg_dict).reset_index()
                          )
        # Добавим отладочную информацию:
        print("🔍 Отладка колонок после агрегации:")
        print(f"Фактическое количество колонок: {len(supplier_stats.columns)}")
        print(f"Названия колонок: {list(supplier_stats.columns)}")
        
        # Динамическое сглаживание названий колонок (из многоиндексного столбца делается столбец с одним наименованием)
        new_columns = []
        for col in supplier_stats.columns:
            if isinstance(col, tuple):
                # Проверяем, что кортеж содержит как минимум два элемента
                if len(col) > 1 and col[1] == "nunique":
                    new_columns.append(f"{col[0]}_count")
                elif len(col) > 1:
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    # Обработка кортежей с одним элементом, если такие есть
                    new_columns.append(col[0])
            else:
                new_columns.append(col)
        supplier_stats.columns = new_columns
        
        
        # Создаем дополнительные расчетные признаки
        
        # 1. Временные признаки
        supplier_stats["years_active"] = np.maximum(
            (pd.to_datetime(supplier_stats["contract_signing_date_max"])
             - pd.to_datetime(supplier_stats["contract_signing_date_min"])
        ).dt.days / 365.25, 0.1)
        
        supplier_stats["avg_contracts_per_year"] = supplier_stats["total_contract_amount_eur_count"] / (
            supplier_stats["years_active"] + 1
        )
        
        # 2. Волатильность и стабильность
        supplier_stats["price_volatility"] = (
            supplier_stats["unit_price_eur_std"]
            / supplier_stats["unit_price_eur_mean"].replace(0, np.nan)
        ).fillna(0)
        
        supplier_stats["contract_size_volatility"] = (
            supplier_stats["total_contract_amount_eur_std"]
            / supplier_stats["total_contract_amount_eur_sum"].replace(0, np.nan)
        ).fillna(0)
        
        supplier_stats["quantity_volatility"] = (
            supplier_stats["quantity_std"] / supplier_stats["quantity_mean"].replace(0, np.nan)
        ).fillna(0)
        
        supplier_stats["delivery_volatility"] = (
            supplier_stats["delivery_time_days_std"]
            / supplier_stats["delivery_time_days_mean"].replace(0, np.nan)
        ).fillna(0)
        
        # 3. Размер и масштаб операций
        supplier_stats["avg_contract_size_rank"] = supplier_stats["total_contract_amount_eur_sum"].rank(
            pct=True
        )
        supplier_stats["total_volume_rank"] = supplier_stats["total_contract_amount_eur_sum"].rank(pct=True)
        
        supplier_stats["contract_size_range"] = (
            supplier_stats["total_contract_amount_eur_max"] - supplier_stats["total_contract_amount_eur_min"]
        )
        
        supplier_stats["price_range"] = (
            supplier_stats["unit_price_eur_max"] - supplier_stats["unit_price_eur_min"]
        )
        
        # 4. Разнообразие и специализация
        supplier_stats["diversification_index"] = (
            np.log1p(supplier_stats["project_name_count"])
            * np.log1p(supplier_stats["product_name_count"])
            * np.log1p(supplier_stats["unit_count"])
            * np.log1p(supplier_stats.get("discipline_count", 1))
        ) ** 0.25  # Геометрическое среднее для сглаживания
        
        supplier_stats["specialization_ratio"] = supplier_stats[
            "total_contract_amount_eur_count"
        ] / supplier_stats["product_name_count"].replace(0, 1)
        
        # 5. Эффективность и надежность
        supplier_stats["delivery_efficiency"] = 1 / (
            supplier_stats["delivery_time_days_mean"] + 1
        )  # Чем меньше время, тем выше эффективность
        
        supplier_stats["contract_consistency"] = 1 / (
            supplier_stats["contract_size_volatility"] + 0.1
        )  # Чем меньше волатильность, тем выше консистентность
        
        # 6. Относительные показатели дополнительных расходов (только если колонка есть)
        if "additional_expenses_sum" in supplier_stats.columns:
            supplier_stats["additional_expenses_ratio"] = (
                supplier_stats["additional_expenses_sum"]
                / supplier_stats["total_contract_amount_eur_sum"].replace(0, np.nan)
            ).fillna(0)
        else:
            supplier_stats["additional_expenses_ratio"] = 0
        
        # 7. Интенсивность работы
        supplier_stats["avg_quantity_per_contract"] = (
            supplier_stats["quantity_sum"] / supplier_stats["total_contract_amount_eur_count"]
        )
        
        supplier_stats["revenue_per_project"] = supplier_stats["total_contract_amount_eur_sum"] / supplier_stats[
            "project_name_count"].replace(0, 1)
        
        # 8. Сезонность и регулярность (если есть данные по месяцам)
        if len(df_clean) > 100:  # Только если достаточно данных
            monthly_activity = df_clean.copy()
            monthly_activity["month"] = pd.to_datetime(
                monthly_activity["contract_signing_date"]
            ).dt.month
            monthly_var = (
                monthly_activity.groupby(["counterparty_name", "month"])
                .size()
                .unstack(fill_value=0)
                .var(axis=1)
            )
            supplier_stats.rename(columns={"counterparty_name_": "counterparty_name"}, inplace=True)
            supplier_stats = supplier_stats.merge(
                monthly_var.rename("seasonal_variance").reset_index(),
                on="counterparty_name",
                how="left",
            )
            supplier_stats["seasonal_variance"] = supplier_stats["seasonal_variance"].fillna(0)
        else:
            supplier_stats["seasonal_variance"] = 0
        
        # 9. Риск-показатели -
        supplier_stats["single_client_dependency"] = (
            1 / supplier_stats["project_name_count"]  # Чем меньше проектов, тем выше зависимость
        )
        
        supplier_stats["price_stability_score"] = 1 / (supplier_stats["price_volatility"] + 0.1)
        
        # Заполняем NaN значения
        supplier_stats = supplier_stats.fillna(0)
        
        # Заменяем inf на большие числа
        supplier_stats = supplier_stats.replace([np.inf, -np.inf], [999999, -999999])
        
        print(
            f"✅ Создано {len(supplier_stats.columns)} признаков для {len(supplier_stats)} поставщиков"
        )
        
        return supplier_stats
        
        
        
    def get_enhanced_feature_columns(self):
        """Возвращает словарь признаков для кластеризации"""
        
        rename_mapping = {
            "total_contract_amount_eur_sum": "total_volume",
            "total_contract_amount_eur_mean": "avg_contract_value",
            "total_contract_amount_eur_count": "contracts_count",
            "quantity_sum": "total_quantity",
            "quantity_mean": "avg_quantity",
            "unit_price_eur_mean": "avg_unit_price",
            "price_volatility": "price_volatility",
            "price_range": "price_range",
            "unit_price_eur_min": "min_unit_price",
            "unit_price_eur_max": "max_unit_price",
            "contract_size_volatility": "contract_size_volatility",
            "contract_size_range": "contract_size_range",
            "total_contract_amount_eur_min": "min_contract_value",
            "total_contract_amount_eur_max": "max_contract_value",
            "years_active": "years_active",
            "avg_contracts_per_year": "avg_contracts_per_year",
            "delivery_time_days_mean": "avg_delivery_time",
            "delivery_volatility": "delivery_volatility",
            "delivery_time_days_min": "min_delivery_time",
            "delivery_time_days_max": "max_delivery_time",
            "project_name_count": "projects_count",
            "product_name_count": "products_count",
            "unit_count": "units_count",
            "discipline_count": "disciplines_count",
            "diversification_index": "diversification_index",
            "specialization_ratio": "specialization_ratio",
            "delivery_efficiency": "delivery_efficiency",
            "contract_consistency": "contract_consistency",
            "price_stability_score": "price_stability_score",
            "quantity_volatility": "quantity_volatility",
            "additional_expenses_ratio": "additional_expenses_ratio",
            "avg_quantity_per_contract": "avg_quantity_per_contract",
            "revenue_per_project": "revenue_per_project",
            "avg_contract_size_rank": "avg_contract_size_rank",
            "total_volume_rank": "total_volume_rank",
            "single_client_dependency": "single_client_dependency",
            "seasonal_variance": "seasonal_variance",
        }
        return rename_mapping
        
    def find_optimal_clusters(self, features, max_clusters=10):
        """Поиск оптимального количества кластеров методом локтя и силуэта"""
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features, kmeans.labels_))

        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # График локтя
        ax1.plot(K_range, inertias, "bo-")
        ax1.set_xlabel("Количество кластеров")
        ax1.set_ylabel("Инерция")
        ax1.set_title("Метод локтя")
        ax1.grid(True)

        # График силуэта
        ax2.plot(K_range, silhouette_scores, "ro-")
        ax2.set_xlabel("Количество кластеров")
        ax2.set_ylabel("Силуэт-коэффициент")
        ax2.set_title("Анализ силуэта")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Рекомендуем оптимальное k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Рекомендуемое количество кластеров: {optimal_k}")
        print(f"Силуэт-коэффициент: {max(silhouette_scores):.3f}")

        return optimal_k

    def cluster_suppliers(self, n_clusters=None):
        """Кластеризация поставщиков с расширенным набором признаков"""
        # Подготовка данных
        supplier_stats = self.prepare_enhanced_supplier_features() # в этой функции все исправлено

        # Выбираем признаки для кластеризации
        rename_mapping = self.get_enhanced_feature_columns()

        # Переименуем столбцы датафрейма
        
        supplier_stats.rename(columns=rename_mapping, inplace=True)
        print(f"📊 Используется {len(supplier_stats.columns)} признаков")
        
        # Удаляем выбросы (поставщиков с экстремальными значениями)
        # Находим поставщиков, которые сильно отличаются по объему
        volume_q99 = supplier_stats["total_volume"].quantile(0.99)
        outliers_mask = supplier_stats["total_volume"] > volume_q99
        
        if outliers_mask.sum() > 0:
            print(f"⚠️ Найдено {outliers_mask.sum()} выбросов (супер-поставщиков):")
            outliers = supplier_stats[outliers_mask]["counterparty_name"].tolist()
            for outlier in outliers:
                volume = supplier_stats[
                    outliers_mask & (supplier_stats["counterparty_name"] == outlier)
                ]["total_volume"].iloc[0]
                print(f"   - {outlier}: {volume:,.0f} EUR")
        
        # Получаем данные без выбросов для дальнейшей кластеризации
        normal_suppliers = supplier_stats[~outliers_mask].copy()
        
        # Отфильтруем только числовые признаки
        numerical_features = normal_suppliers.select_dtypes(include=['number']).columns.tolist()
        
        # Определяем список признаков для кластеризации
        features_for_clustering = [col for col in numerical_features if col not in ['counterparty_name']]
        
        print(f"🎯 Кластеризуем {len(normal_suppliers)} обычных поставщиков")
        
        normal_features_to_scale = normal_suppliers[features_for_clustering]

        # Стандартизация только нормальных данных (без выбросов)
        normal_features_scaled = self.scaler.fit_transform(normal_features_to_scale)

        # Поиск оптимального количества кластеров (но не менее 3)
        if n_clusters is None:
            optimal_k = self.find_optimal_clusters(normal_features_scaled, max_clusters=8)
            n_clusters = max(optimal_k, 3) # минимум 3 кластера
        
        print(f"🎯 Используем {n_clusters} кластеров")

        # Кластеризация основной массы
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        normal_suppliers["cluster"] = kmeans.fit_predict(normal_features_scaled)
        
        # Добавляем выбросы как отдельные кластеры
        if outliers_mask.sum() > 0:
            outlier_suppliers = supplier_stats[outliers_mask].copy()
            # Присваиваем каждому выбросу свой кластер
            outlier_suppliers["cluster"] = range(
                n_clusters, n_clusters + len(outlier_suppliers)
            )
            # Объединяем все данные
            supplier_stats_final = pd.concat(
                [normal_suppliers, outlier_suppliers], ignore_index=True
            )
        else:
            supplier_stats_final = normal_suppliers

        self.clusters = supplier_stats_final
        self.cluster_centers = kmeans.cluster_centers_
        self.feature_columns = features_for_clustering

        return supplier_stats_final

    def analyze_enhanced_clusters(self):
        """Расширенный анализ характеристик кластеров"""
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return

        # Ключевые показатели для анализа
        key_metrics = [
            "total_volume",
            "avg_contract_value",
            "contracts_count",
            "price_volatility",
            "years_active",
            "projects_count",
            "diversification_index",
            "delivery_efficiency",
            "contract_consistency",
            "specialization_ratio",
        ]

        available_metrics = [col for col in key_metrics if col in self.clusters.columns and col != 'counterparty_name']

        # Статистика по кластерам
        cluster_summary = (
            self.clusters.groupby("cluster")
            .agg(
                {
                    "counterparty_name": "count",
                    **{col: "mean" for col in available_metrics},
                }
            )
            .round(3)
        )

        cluster_summary.columns = ["Количество"] + [
            col.replace("_", " ").title() for col in available_metrics
        ]

        print("=== РАСШИРЕННАЯ ХАРАКТЕРИСТИКА КЛАСТЕРОВ ===")
        print(cluster_summary)

        # Детальная интерпретация кластеров
        print("\n=== ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ ===")
        for cluster_id in sorted(self.clusters["cluster"].unique()):
            cluster_data = self.clusters[self.clusters["cluster"] == cluster_id]
            # cluster_data - все данные по кластеру № cluster_id (в будущем вынести в Excel)
            # Расчет характеристик кластера
            avg_volume = cluster_data["total_volume"].mean()
            avg_volatility = (
                cluster_data["price_volatility"].mean()
                if "price_volatility" in cluster_data.columns
                else 0
            )
            avg_projects = cluster_data["projects_count"].mean()
            avg_contracts = cluster_data["contracts_count"].mean()
            avg_years = cluster_data["years_active"].mean()

            # Определение типа поставщика
            if avg_volume > self.clusters["total_volume"].quantile(0.8):
                if avg_volatility < 0.2 and avg_years > 2:
                    category = "🏆 PREMIUM (крупные, стабильные, опытные)"
                else:
                    category = "⚡ КРУПНЫЕ (высокий объем, но нестабильные)"
            elif avg_contracts > self.clusters["contracts_count"].quantile(0.7):
                if avg_projects > 3:
                    category = "🔄 АКТИВНЫЕ УНИВЕРСАЛЫ (частые заказы, много проектов)"
                else:
                    category = (
                        "🔄 АКТИВНЫЕ СПЕЦИАЛИСТЫ (частые заказы, узкая специализация)"
                    )
            elif avg_years < 1:
                category = "🆕 НОВЫЕ (недавно начали работать)"
            elif avg_projects == 1:
                category = "🎯 УЗКОСПЕЦИАЛИЗИРОВАННЫЕ (работают в одном проекте)"
            else:
                category = "📉 РЕДКИЕ (нерегулярные поставщики)"

            print(f"\nКластер {cluster_id}: {category}")
            print(f"  Поставщиков: {len(cluster_data)}")
            print(f"  Средний объем: {avg_volume:,.0f} EUR")
            print(f"  Волатильность цен: {avg_volatility:.3f}")
            print(f"  Среднее кол-во проектов: {avg_projects:.1f}")
            print(f"  Среднее кол-во контрактов: {avg_contracts:.1f}")
            print(f"  Средний опыт работы: {avg_years:.1f} лет")

            # Показываем топ- поставщиков в кластере
            top_suppliers = cluster_data.nlargest(5, "total_volume")[
                "counterparty_name"
            ].tolist()
            print(
                f"  Топ поставщики: {', '.join(top_suppliers[:4])}{'...' if len(top_suppliers) > 4 else ''}"
            )

        return cluster_summary
    
    def visualize_enhanced_clusters(self, output_folder):
        import os
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        """
        Расширенная визуализация кластеров.
        Сохраняет графики в указанную папку.
        """
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return
    
        # 1. Создание папки, если она не существует
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"✅ Создана папка для результатов: {output_folder}")
    
        # Подготовка данных для PCA
        features = self.clusters[self.feature_columns]
        features_scaled = self.scaler.transform(features)
    
        # PCA для визуализации
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
    
        # Создаем DataFrame для удобства
        plot_data = pd.DataFrame(
            {
                "PC1": features_pca[:, 0],
                "PC2": features_pca[:, 1],
                "cluster": self.clusters["cluster"],
                "supplier": self.clusters["counterparty_name"],
                "total_volume": self.clusters["total_volume"],
                "contracts_count": self.clusters["contracts_count"],
                "diversification": self.clusters.get("diversification_index", 1),
            }
        )
    
        # Создаем комплексную визуализацию
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
        # 1. Основной PCA scatter plot
        scatter1 = ax1.scatter(
            plot_data["PC1"],
            plot_data["PC2"],
            c=plot_data["cluster"],
            s=np.sqrt(plot_data["total_volume"]) / 50,
            alpha=0.7,
            cmap="tab10",
        )
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)")
        ax1.set_title("PCA кластеров (размер = объем закупок)")
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label="Кластер")
    
        # 2. Объем vs Количество контрактов
        scatter2 = ax2.scatter(
            self.clusters["total_volume"],
            self.clusters["contracts_count"],
            c=self.clusters["cluster"],
            alpha=0.7,
            cmap="tab10",
        )
        ax2.set_xlabel("Общий объем (EUR)")
        ax2.set_ylabel("Количество контрактов")
        ax2.set_title("Объем vs Активность")
        ax2.set_xscale("log")
        ax2.grid(True, alpha=0.3)
    
        # 3. Волатильность vs Размер
        if "price_volatility" in self.clusters.columns:
            scatter3 = ax3.scatter(
                self.clusters["price_volatility"],
                self.clusters["avg_contract_value"],
                c=self.clusters["cluster"],
                alpha=0.7,
                cmap="tab10",
            )
            ax3.set_xlabel("Волатильность цен")
            ax3.set_ylabel("Средняя стоимость контракта (EUR)")
            ax3.set_title("Стабильность vs Размер контракта")
            ax3.set_yscale("log")
            ax3.grid(True, alpha=0.3)
    
        # 4. Диверсификация vs Специализация
        if (
            "diversification_index" in self.clusters.columns
            and "specialization_ratio" in self.clusters.columns
        ):
            scatter4 = ax4.scatter(
                self.clusters["diversification_index"],
                self.clusters["specialization_ratio"],
                c=self.clusters["cluster"],
                alpha=0.7,
                cmap="tab10",
            )
            ax4.set_xlabel("Индекс диверсификации")
            ax4.set_ylabel("Коэффициент специализации")
            ax4.set_title("Диверсификация vs Специализация")
            ax4.grid(True, alpha=0.3)
    
        plt.tight_layout()
    
        # 2. Создание полного пути и сохранение графика
        full_path_visualization = os.path.join(output_folder, "cluster_visualization.png")
        plt.savefig(full_path_visualization)
    
        plt.show()
    
        print(f"✅ Графики визуализации сохранены в: {full_path_visualization}")
        print(f"PCA объясняет {pca.explained_variance_ratio_.sum():.1%} дисперсии")
        print(f"Использовано {len(self.feature_columns)} признаков для кластеризации")

    def get_enhanced_recommendations(self):
        """Расширенные рекомендации по работе с кластерами"""
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return

        recommendations = {
            "🏆 PREMIUM": [
                "Развивать долгосрочное стратегическое партнерство",
                "Предоставлять приоритет в новых тендерах",
                "Заключать рамочные соглашения на выгодных условиях",
                "Регулярные встречи для стратегического планирования",
                "Совместная разработка инновационных решений",
            ],
            "⚡ КРУПНЫЕ": [
                "Работать над стабилизацией цен",
                "Заключать договоры с фиксированными ценами",
                "Усиленный контроль качества",
                "Диверсификация рисков через других поставщиков",
                "Регулярный мониторинг финансового состояния",
            ],
            "🔄 АКТИВНЫЕ": [
                "Автоматизировать процессы заказа",
                "Внедрить EDI-системы",
                "Оптимизировать логистику",
                "Создать каталоги стандартных позиций",
                "Упростить процедуры согласования",
            ],
            "🎯 СПЕЦИАЛИЗИРОВАННЫЕ": [
                "Углублять экспертизу в их области",
                "Привлекать к консультациям по техническим решениям",
                "Развивать эксклюзивные партнерства",
                "Совместное участие в выставках и конференциях",
            ],
            "🆕 НОВЫЕ": [
                "Тщательная проверка надежности",
                "Начинать с небольших заказов",
                "Регулярный мониторинг выполнения",
                "Обучение корпоративным стандартам",
                "Постепенное увеличение объемов при успешной работе",
            ],
            "📉 РЕДКИЕ": [
                "Пересмотреть необходимость сотрудничества",
                "Найти более активных поставщиков",
                "Минимизировать административные затраты",
                "Использовать только для экстренных случаев",
                "Рассмотреть консолидацию с другими поставщиками",
            ],
        }

        print("=== РАСШИРЕННЫЕ РЕКОМЕНДАЦИИ ПО КЛАСТЕРАМ ===")
        for category, recs in recommendations.items():
            print(f"\n{category}:")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")

    def save_results_to_excel(self, output_folder):
        """
        Сохраняет результаты кластеризации и сводную статистику в Excel-файлы.
        """
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return

        # Создание папки, если она не существует
        import os
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"✅ Создана папка для результатов: {output_folder}")

        # 1. Сохранение всех данных с кластерами
        full_path_clusters = os.path.join(output_folder, "supplier_clusters.xlsx")
        self.clusters.to_excel(full_path_clusters, index=False)
        print(f"✅ Результаты кластеризации сохранены в: {full_path_clusters}")

        # 2. Сохранение сводной статистики по кластерам
        cluster_summary = self.analyze_enhanced_clusters()
        full_path_summary = os.path.join(output_folder, "cluster_summary.xlsx")
        cluster_summary.to_excel(full_path_summary)
        print(f"✅ Сводная статистика по кластерам сохранена в: {full_path_summary}")

        # 3. Сохранение выбросов в отдельный файл (если они есть)
        outliers_mask = (self.clusters['cluster'] >= self.clusters['cluster'].nunique() -
                         len(self.clusters[self.clusters['cluster'].duplicated(keep=False) == False]))
        if outliers_mask.any():
            outliers_df = self.clusters[outliers_mask]
            full_path_outliers = os.path.join(output_folder, "outlier_suppliers.xlsx")
            outliers_df.to_excel(full_path_outliers, index=False)
            print(f"✅ Обнаруженные выбросы сохранены в: {full_path_outliers}")
    
    def save_cluster_interpretation_to_excel(self, output_folder):
        """
        Сохраняет детальную интерпретацию кластеров в Excel.
        """
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return
    
        import os
        import pandas as pd
    
        # Создание папки, если она не существует
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"✅ Создана папка для результатов: {output_folder}")
    
        # Подготовка данных для сохранения
        interpretation_data = []
    
        # Цикл по кластерам
        for cluster_id in sorted(self.clusters["cluster"].unique()):
            cluster_data = self.clusters[self.clusters["cluster"] == cluster_id]
    
            # Расчет характеристик кластера
            avg_volume = cluster_data["total_volume"].mean()
            avg_volatility = (
                cluster_data["price_volatility"].mean()
                if "price_volatility" in cluster_data.columns
                else 0
            )
            avg_projects = cluster_data["projects_count"].mean()
            avg_contracts = cluster_data["contracts_count"].mean()
            avg_years = cluster_data["years_active"].mean()
    
            # Определение категории
            if avg_volume > self.clusters["total_volume"].quantile(0.8):
                if avg_volatility < 0.2 and avg_years > 2:
                    category = "🏆 PREMIUM (крупные, стабильные, опытные)"
                else:
                    category = "⚡ КРУПНЫЕ (высокий объем, но нестабильные)"
            elif avg_contracts > self.clusters["contracts_count"].quantile(0.7):
                if avg_projects > 3:
                    category = "🔄 АКТИВНЫЕ УНИВЕРСАЛЫ (частые заказы, много проектов)"
                else:
                    category = (
                        "🔄 АКТИВНЫЕ СПЕЦИАЛИСТЫ (частые заказы, узкая специализация)"
                    )
            elif avg_years < 1:
                category = "🆕 НОВЫЕ (недавно начали работать)"
            elif avg_projects == 1:
                category = "🎯 УЗКОСПЕЦИАЛИЗИРОВАННЫЕ (работают в одном проекте)"
            else:
                category = "📉 РЕДКИЕ (нерегулярные поставщики)"
    
            # Получение топ-поставщиков
            top_suppliers = cluster_data.nlargest(5, "total_volume")[
                "counterparty_name"
            ].tolist()
    
            # Добавление данных в список
            interpretation_data.append(
                {
                    "Cluster ID": cluster_id,
                    "Category": category,
                    "Number of Suppliers": len(cluster_data),
                    "Average Volume (EUR)": round(avg_volume, 2),
                    "Price Volatility": round(avg_volatility, 3),
                    "Average Projects": round(avg_projects, 1),
                    "Average Contracts": round(avg_contracts, 1),
                    "Average Years Active": round(avg_years, 1),
                    "Top Suppliers": ", ".join(top_suppliers),
                }
            )
    
        # Создание DataFrame из собранных данных
        interpretation_df = pd.DataFrame(interpretation_data)
    
        # Сохранение в Excel
        full_path_interpretation = os.path.join(
            output_folder, "cluster_interpretation.xlsx"
        )
        interpretation_df.to_excel(full_path_interpretation, index=False)
    
        print(
            f"✅ Детальная интерпретация кластеров сохранена в: {full_path_interpretation}"
        )

def run_enhanced_supplier_clustering(df, output_folder=r'D:\Analysis-Results\Cluster_Analysis'):
    """
    Запуск полного расширенного анализа кластеризации поставщиков
    """
    print("📊 Доступные колонки:", list(df.columns))

    analyzer = EnhancedSupplierClusterAnalyzer(df)

    # Выполняем кластеризацию
    supplier_clusters = analyzer.cluster_suppliers()

    # Анализируем результаты
    cluster_summary = analyzer.analyze_enhanced_clusters()

    # Визуализируем
    analyzer.visualize_enhanced_clusters(output_folder)

    # Получаем рекомендации
    analyzer.get_enhanced_recommendations()

    # Сохраняем результаты в Excel
    analyzer.save_results_to_excel(output_folder)
    analyzer.save_cluster_interpretation_to_excel(output_folder)

    return supplier_clusters, analyzer

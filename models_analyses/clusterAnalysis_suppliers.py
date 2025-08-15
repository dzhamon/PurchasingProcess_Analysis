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

class SupplierClusterAnalyzer:
    def __init__(self, df, column_mapping=None):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.clusters = None
        self.cluster_centers = None
        
        # Маппинг колонок по умолчанию
        default_mapping = {
            'counterparty_name': 'counterparty_name',
            'contract_signing_date': 'contract_signing_date',
            'total_contract_amount_eur': 'total_contract_amount_eur',
            'unit_price_eur': 'unit_price_eur',
            'project_name': 'project_name'
        }
        
        self.column_mapping = column_mapping or default_mapping
        self._validate_columns()
        
    def _validate_columns(self):
        """ Проверяем наличие необходимых колонок"""
        missing = []
        for standard_name, actual_name in self.column_mapping.items():
            if actual_name not in self.df.columns:
                missing.append(actual_name)
        
        if missing:
            print(f"❌ Отсутствуют колонки: {missing}")
            print(f"📋 Доступные колонки: {list(self.df.columns)}")
            raise ValueError(f"Отсутствуют колонки: {missing}")
        
        print(f"✅ Все необходимые колонки найдены!")
        print(f"📊 Записей в DataFrame: {len(self.df)}")
        print(
            f"👥 Уникальных поставщиков: {self.df[self.column_mapping['counterparty_name']].nunique()}"
        )


    def prepare_supplier_features(self):
        """
        Подготовка признаков для кластеризации поставщиков
        """
        # Проверяем наличие необходимых колонок
        required_columns = [
            "counterparty_name",
            "total_contract_amount_eur",
            "unit_price_eur",
            "contract_signing_date",
            "project_name",
        ]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"❌ Отсутствуют колонки: {missing_columns}")
            print(f"📋 Доступные колонки: {list(self.df.columns)}")
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
        
        print(f"✅ Обрабатываем {len(self.df)} записей...")
        print(f"📊 Уникальных поставщиков: {self.df['counterparty_name'].nunique()}")
        
        # Агрегируем данные по поставщикам
        supplier_stats = (
            self.df.groupby(self.column_mapping['counterparty_name'])
            .agg(
                {
                    self.column_mapping['total_contract_amount_eur']: ['sum', 'mean', 'std', 'count'],
                    self.column_mapping['unit_price_eur']: ['mean', 'std'],
                    self.column_mapping['contract_signing_date']: ['min', 'max'],
                    self.column_mapping['project_name']: 'nunique',
                }
            )
            .reset_index()
        )

        # Сглаживаем названия колонок
        supplier_stats.columns = [
            "counterparty_name",
            "total_volume",
            "avg_contract_value",
            "contract_value_std",
            "contracts_count",
            "avg_unit_price",
            "unit_price_std",
            "first_contract",
            "last_contract",
            "projects_count",
        ]

        # Создаем дополнительные признаки
        supplier_stats["years_active"] = (
            pd.to_datetime(supplier_stats["last_contract"])
            - pd.to_datetime(supplier_stats["first_contract"])
        ).dt.days / 365.25
        
        # Защищаемся от деления на ноль(0)
        supplier_stats["price_volatility"] = (
            supplier_stats["unit_price_std"] / supplier_stats["avg_unit_price"].replace(0, np.nan)
        ).fillna(0)

        supplier_stats["avg_contracts_per_year"] = supplier_stats["contracts_count"] / (
            supplier_stats["years_active"] + 1
        )

        # Заполняем NaN значения
        supplier_stats = supplier_stats.fillna(0)

        return supplier_stats

    def find_optimal_clusters(self, features, max_clusters=10):
        """
        Поиск оптимального количества кластеров методом локтя и силуэта
        """
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
        """
        Кластеризация поставщиков
        """
        # Подготовка данных
        supplier_stats = self.prepare_supplier_features()

        # Выбираем признаки для кластеризации
        feature_columns = [
            "total_volume",
            "avg_contract_value",
            "contracts_count",
            "avg_unit_price",
            "price_volatility",
            "projects_count",
            "years_active",
            "avg_contracts_per_year",
        ]

        features = supplier_stats[feature_columns]

        # Стандартизация
        features_scaled = self.scaler.fit_transform(features)

        # Поиск оптимального количества кластеров
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(features_scaled)

        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        supplier_stats["cluster"] = kmeans.fit_predict(features_scaled)

        self.clusters = supplier_stats
        self.cluster_centers = kmeans.cluster_centers_

        return supplier_stats

    def analyze_clusters(self):
        """
        Анализ характеристик кластеров
        """
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return

        # Статистика по кластерам
        cluster_summary = (
            self.clusters.groupby("cluster")
            .agg(
                {
                    "counterparty_name": "count",
                    "total_volume": "mean",
                    "avg_contract_value": "mean",
                    "contracts_count": "mean",
                    "avg_unit_price": "mean",
                    "price_volatility": "mean",
                    "projects_count": "mean",
                    "years_active": "mean",
                }
            )
            .round(2)
        )

        cluster_summary.columns = [
            "Количество",
            "Средний объем",
            "Средняя стоимость контракта",
            "Среднее кол-во контрактов",
            "Средняя цена за единицу",
            "Волатильность цен",
            "Количество проектов",
            "Лет активности",
        ]

        print("=== ХАРАКТЕРИСТИКИ КЛАСТЕРОВ ===")
        print(cluster_summary)

        # Интерпретация кластеров
        print("\n=== ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ ===")
        for cluster_id in sorted(self.clusters["cluster"].unique()):
            cluster_data = cluster_summary.loc[cluster_id]

            # Простая логика классификации
            if (
                cluster_data["Средний объем"]
                > cluster_summary["Средний объем"].median()
            ):
                if cluster_data["Волатильность цен"] < 0.2:
                    category = "🏆 ПРЕМИУМ (крупные, стабильные)"
                else:
                    category = "⚡ КРУПНЫЕ (высокий объем, нестабильные цены)"
            elif cluster_data["Среднее кол-во контрактов"] > 5:
                category = "🔄 АКТИВНЫЕ (частые небольшие закупки)"
            else:
                if cluster_data["Лет активности"] < 1:
                    category = "🆕 НОВЫЕ (недавно начали работать)"
                else:
                    category = "📉 РЕДКИЕ (нерегулярные поставщики)"

            print(f"Кластер {cluster_id}: {category}")
            print(f"  Поставщиков: {cluster_data['Количество']}")
            print(f"  Средний объем: {cluster_data['Средний объем']:,.0f} EUR")
            print(f"  Волатильность: {cluster_data['Волатильность цен']:.2f}")
            print()

        return cluster_summary

    def visualize_clusters(self):
        """
        Визуализация кластеров
        """
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return

        # Подготовка данных для PCA
        feature_columns = [
            "total_volume",
            "avg_contract_value",
            "contracts_count",
            "avg_unit_price",
            "price_volatility",
            "projects_count",
            "years_active",
            "avg_contracts_per_year",
        ]

        features = self.clusters[feature_columns]
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
            }
        )

        # Визуализация
        plt.figure(figsize=(12, 8))

        # Основной scatter plot
        scatter = plt.scatter(
            plot_data["PC1"],
            plot_data["PC2"],
            c=plot_data["cluster"],
            s=np.sqrt(plot_data["total_volume"]) / 100,  # размер по объему
            alpha=0.7,
            cmap="tab10",
        )

        # Добавляем подписи для крупнейших поставщиков
        top_suppliers = plot_data.nlargest(10, "total_volume")
        for _, row in top_suppliers.iterrows():
            plt.annotate(
                (
                    row["supplier"][:20] + "..."
                    if len(row["supplier"]) > 20
                    else row["supplier"]
                ),
                (row["PC1"], row["PC2"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        plt.colorbar(scatter, label="Кластер")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)")
        plt.title("Кластеры поставщиков (размер точки = объем закупок)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"PCA объясняет {pca.explained_variance_ratio_.sum():.1%} дисперсии")

    def get_cluster_recommendations(self):
        """
        Рекомендации по работе с кластерами
        """
        if self.clusters is None:
            print("Сначала выполните кластеризацию!")
            return

        recommendations = {
            "🏆 ПРЕМИУМ": [
                "Развивать долгосрочное партнерство",
                "Предоставлять приоритет в новых тендерах",
                "Регулярные встречи и стратегическое планирование",
            ],
            "⚡ КРУПНЫЕ": [
                "Работать над стабилизацией цен",
                "Заключать рамочные соглашения",
                "Усиленный контроль качества",
            ],
            "🔄 АКТИВНЫЕ": [
                "Автоматизировать процессы заказа",
                "Внедрить EDI-системы",
                "Оптимизировать логистику",
            ],
            "🆕 НОВЫЕ": [
                "Тщательная проверка надежности",
                "Начинать с небольших заказов",
                "Регулярный мониторинг выполнения",
            ],
            "📉 РЕДКИЕ": [
                "Пересмотреть необходимость сотрудничества",
                "Найти альтернативных поставщиков",
                "Минимизировать административные затраты",
            ],
        }

        print("=== РЕКОМЕНДАЦИИ ПО КЛАСТЕРАМ ===")
        for category, recs in recommendations.items():
            print(f"\n{category}:")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")


def  run_supplier_clustering(df):
    """
    Запуск полного анализа кластеризации поставщиков
    """
    print("Колонки DataFrame :", df.columns)
    analyzer = SupplierClusterAnalyzer(df)

    # Выполняем кластеризацию
    supplier_clusters = analyzer.cluster_suppliers()

    # Анализируем результаты
    cluster_summary = analyzer.analyze_clusters()

    # Визуализируем
    analyzer.visualize_clusters()

    # Получаем рекомендации
    analyzer.get_cluster_recommendations()

    return supplier_clusters, analyzer


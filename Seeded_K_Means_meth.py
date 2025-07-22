import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)
n_samples = 300

# Создаем три группы клиентов
data_vip = np.random.normal(loc=[10, 10], scale=1, size=(100, 2))
print(data_vip)
print('----------------------------------------')
data_new = np.random.normal(loc=[0, 0], scale=1, size=(100, 2))
print(data_new)
print('----------------------------------------')
data_regular = np.random.normal(loc=[5, 5], scale=1, size=(100, 2))
print(data_regular)

# Объединяем данные
data = np.vstack((data_vip, data_new, data_regular))

# Задаем семена (метки) для части данных
seed_data = np.vstack((data_vip[0], data_new[0], data_regular[0])).reshape(3, 2)
seed_labels = np.array([0, 1, 2])  # Метки для VIP, Новых и Постоянных клиентов

# Обучаем модель K-means, начиная с семян
kmeans = KMeans(n_clusters=3, init=seed_data, n_init=1)
kmeans.fit(data)

# Получаем результаты
labels = kmeans.labels_

# Оценка результатов кластеризации
silhouette_avg = silhouette_score(data, labels)
print(f'Silhouette Score: {silhouette_avg}')

# Визуализация
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(seed_data[:, 0], seed_data[:, 1], c=seed_labels, marker='*', s=200, label='Seeds')
plt.legend()
plt.title('Seeded K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

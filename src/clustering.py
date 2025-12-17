from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def kmeans_elbow_plot(X_train_pca, k_range=range(2, 8)):

   inertia = []

   for k in k_range:

      km = KMeans(n_clusters=k, random_state=42)

      km.fit(X_train_pca)

      inertia.append(km.inertia_)

   plt.figure(figsize=(6,4))

   plt.plot(k_range, inertia, marker='o')

   plt.xlabel("Number of Clusters (k)")

   plt.ylabel("Inertia")

   plt.title("KMeans Elbow Plot (PCA space)")

   plt.show()

   return inertia

def apply_kmeans(X_train_pca, X_test_pca, n_clusters=3):

   kmeans = KMeans(n_clusters=n_clusters, random_state=42)

   train_clusters = kmeans.fit_predict(X_train_pca)

   test_clusters = kmeans.predict(X_test_pca)

   return train_clusters, test_clusters
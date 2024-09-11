# -*- coding: utf-8 -*-
"""
使用KMeans对PDF文件名进行聚类预处理
"""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from config import FORMATED_PDF_NAME_FOLDER
from load_pdf import load_pdf_names


def find_optimal_clusters(data, max_k):
    """
    使用肘部法则找到最佳聚类数量

    Args:
        data (array-like): 输入数据。
        max_k (int): 最大聚类数量。

    Returns:
        int: 最佳聚类数量。
    """
    iters = range(1, max_k + 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    # 绘制肘部图
    plt.figure(figsize=(10, 8))
    plt.plot(iters, sse, marker="o")
    plt.xlabel("Cluster Centers")
    plt.ylabel("SSE")
    plt.title("Elbow Method For Optimal k")
    plt.savefig("elbow_method.png")
    plt.close()

    return sse.index(min(sse)) + 1


def preprocess_with_kmeans(
    max_clusters=10, output_file="pdf_names_clustered_results.json"
):
    """
    使用KMeans对PDF文件名进行聚类预处理

    Args:
        max_clusters (int): 最大聚类数量。
        output_file (str): 聚类结果输出文件名。

    Returns:
        None
    """
    pdf_names = load_pdf_names(FORMATED_PDF_NAME_FOLDER)
    # 使用TF-IDF向量化文件名
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    x = vectorizer.fit_transform(pdf_names)

    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(x, max_clusters)

    # 应用KMeans聚类
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(x)

    # 准备返回结果
    clustered_files = {i: [] for i in range(optimal_clusters)}
    for file, label in zip(pdf_names, cluster_labels):
        clustered_files[label].append(file)

    # 将聚类结果转换为JSON格式并保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clustered_files, f, ensure_ascii=False, indent=4)

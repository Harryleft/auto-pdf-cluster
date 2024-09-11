# -*- coding: utf-8 -*-
"""结合大语言模型（LLM）开展PDF文件自动分类工作流
主要思路：
1. 提取文件名：读取指定文件目录的下的PDF文件名
2. 文件名规范化：利用LLM对PDF文件名进行规范化
3. 预处理聚类：使用KMeans对PDF文件名进行聚类
4. 主题分类：借助LLM参考聚类结果进行主题分类
5. 文件整理：根据LLM分类结果将相应的本地PDF文件移动到相应的主题分类文件夹中
"""
from add_prefix_to_pdf import add_prefix_to_pdf
from config import SOURCE_PDF_FOLDER, FORMATED_PDF_NAME_FOLDER
from pdf_classify import process_pdfs_cluster, scan_and_move_pdfs_back
from pdf_name_normalize import rename_pdf_files
from preprocess_title_with_kmeans import preprocess_with_kmeans

if __name__ == '__main__':
    # 1. 规范化命名PDF文件
    rename_pdf_files(SOURCE_PDF_FOLDER, FORMATED_PDF_NAME_FOLDER)

    # 2. 为PDF文件添加序号前缀
    add_prefix_to_pdf(FORMATED_PDF_NAME_FOLDER)

    # 3. 借助KMeans对PDF文件名进行初步聚类
    preprocess_with_kmeans()

    # 4. 借助LLM参考聚类结果进行主题分类，并将PDF文件移动到相应的文件夹
    process_pdfs_cluster()

    # 5.如果对分类结果不满意，可以调用以下函数将PDF文件移回原始文件夹
    # scan_and_move_pdfs_back(FORMATED_PDF_NAME_FOLDER, PDF_CLASSIFICATION_DIR)

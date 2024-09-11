# -*- coding: utf-8 -*-
"""
加载PDF文件的名称或内容模块

该模块提供了从指定目录中加载PDF文件名和内容的功能。主要功能包括：

1. 使用正则表达式处理文件名。
2. 读取指定文件夹下的PDF文件名。
"""

import glob
import os
import re


def get_paper_title_with_regx(filename):
    """
    使用正则表达式处理文件名。

    Args:
        filename (str): 原始文件名。

    Returns:
        str: 处理后的文件名。如果文件名符合规范，返回原始文件名；
             如果需要使用大语言模型处理，返回None。
    """
    name_without_ext = os.path.splitext(filename)[0]
    # 检查文件名是否为正常的标题
    if re.match(r"^[\u4e00-\u9fa5A-Za-z0-9\s]+$", name_without_ext):
        return name_without_ext  # 返回原始文件名，表示跳过该文件

    # 检查是否需要使用大语言模型处理
    if "..." in name_without_ext:
        return None  # 返回None表示需要使用大语言模型处理

    # 移除末尾的作者姓名（假设格式为 "_作者姓名"）
    return re.sub(r"_[^_]+$", "", name_without_ext)


def load_pdf_names(directory):
    """
    读取指定文件夹下的PDF文件名。

    Args:
        directory (str): 文件夹路径。

    Returns:
        list: 包含文件名（不含扩展名）的列表。
    """
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    pdf_files.sort()
    pdf_names = [
        os.path.splitext(os.path.basename(file_path))[0] for file_path in pdf_files
    ]
    return pdf_names

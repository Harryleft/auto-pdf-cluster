# -*- coding: utf-8 -*-
"""PDF文件名规范化处理

该模块提供了从PDF文件中提取文本并重命名文件的功能。主要功能包括：

1. 从PDF文件中提取文本内容。
2. 使用大语言模型（DeepSeek）从文本中提取并补充论文标题。
3. 根据提取的标题重命名PDF文件。
4. 检查输出目录中是否已经存在符合命名要求的PDF文件，避免重复处理。

主要函数：
- split_title(title): 将标题拆分为四个部分。
- load_document(file_path): 解析PDF文档格式，返回文档内容字符串。
- get_paper_title_with_deepseek(text, original_title): 使用DeepSeek模型从文本中提取并补充论文标题。
- sanitize_filename(filename): 清理文件名，移除非法字符。
- process_filename(filename): 处理文件名，根据不同情况进行相应的处理。
- rename_pdf_files(folder_path, output_path): 重命名指定文件夹中的PDF文件。

运行效果：
    人物传记资料本体构建与可视...馆学家彭斐章九十自述》为例_司莉.pdf -> 人物传记资料本体构建与可视化——以《图书馆学家彭斐章九十自述》为例.pdf
    农业科学数据集的本体构建与...以“棉花病害防治”领域为例_刘桂锋.pdf -> 农业科学数据集的本体构建与可视化研究以“棉花病害防治”领域为例.pdf
    古籍中人物史料的关联组织研...文志》中西汉经学家群体为例_程结晶.pdf -> 古籍中人物史料的关联组织研究——以《汉书·艺文志》中西汉经学家群体为例.pdf
"""
import os
import re
import logging
import shutil

from langchain_community.document_loaders import (
    PDFPlumberLoader,
)

from custom_exception import CopyException, MoveException
from fix_pdf_title_with_llm import get_paper_title_with_deepseek
from load_pdf import get_paper_title_with_regx

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_pdf_content(file_path):
    """
    解析多种文档格式的文件，返回文档内容字符串
    :param file_path: 文档文件路径
    :return: 返回文档内容的字符串
    """

    # 定义文档解析加载器字典，根据文档类型选择对应的文档解析加载器类和输入参数
    document_loader_mapping = {
        ".pdf": (PDFPlumberLoader, {}),  # 暂时只对PDF文档进行处理
    }

    ext = os.path.splitext(file_path)[1]  # 获取文件扩展名，确定文档类型
    loader_tuple = document_loader_mapping.get(
        ext
    )  # 获取文档对应的文档解析加载器类和参数元组

    if loader_tuple:  # 判断文档格式是否在加载器支持范围
        loader_class, loader_args = loader_tuple  # 解包元组，获取文档解析加载器类和参数
        loader = loader_class(
            file_path, **loader_args
        )  # 创建文档解析加载器实例，并传入文档文件路径
        documents = loader.load()  # 加载文档
        content = "\n".join(
            [doc.page_content for doc in documents]
        )  # 多页文档内容组合为字符串
        return content[:250]  # 返回文档内容的字符串

    print(file_path + f"，不支持的文档类型: '{ext}'")
    return ""

def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    return re.sub(r'[<>:"/\\|?*]', "", filename)


def create_output_directory(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def is_valid_pdf(filename):
    return filename.lower().endswith(".pdf")

def is_filename_valid(filename):
    return re.match(r"^\d+_.*\.pdf$", filename)

def process_filename(filename, file_path):
    try:
        processed_name = get_paper_title_with_regx(filename)
        if processed_name is None:
            pdf_text = load_pdf_content(file_path)
            original_title = os.path.splitext(filename)[0]
            paper_title = get_paper_title_with_deepseek(pdf_text, original_title)
            if paper_title:
                return sanitize_filename(paper_title) + ".pdf"
            logging.warning("无法提取标题 %s", filename)
            return None
        elif processed_name == os.path.splitext(filename)[0]:
            logging.info("跳过: %s", filename)
            return None
        else:
            return processed_name + ".pdf"
    except Exception as e:
        logging.error("处理文件名时出错 %s: %s", filename, str(e))
        return None

def rename_pdf_files(folder_path, output_path):
    create_output_directory(output_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if is_valid_pdf(filename):
            try:
                if is_filename_valid(filename):
                    logging.info("文件名已符合要求，直接移动: %s", filename)
                    new_file_path = os.path.join(output_path, filename)
                    move_file(file_path, new_file_path)
                else:
                    new_filename = process_filename(filename, file_path)
                    if new_filename:
                        new_file_path = os.path.join(output_path, new_filename)
                        copy_file(file_path, new_file_path)
                    else:
                        logging.warning("无法处理文件: %s", filename)
            except Exception as e:
                logging.error("处理文件时出错 %s: %s", filename, str(e))

def move_file(file_path, new_file_path):
    try:
        if not os.path.exists(new_file_path):
            shutil.move(file_path, new_file_path)
            logging.info("成功移动: %s -> %s", file_path, new_file_path)
        else:
            logging.warning("目标文件已存在，跳过移动: %s", new_file_path)
    except MoveException as e:
        logging.error("移动文件时出错 %s: %s", file_path, str(e))

def copy_file(file_path, new_file_path):
    try:
        if not os.path.exists(new_file_path):
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))
            shutil.copy2(file_path, new_file_path)
            logging.info("成功复制: %s -> %s", file_path, new_file_path)
        else:
            logging.warning("目标文件已存在，跳过复制: %s", new_file_path)
    except CopyException as e:
        logging.error("复制文件时出错 %s: %s", file_path, str(e))

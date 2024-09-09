# -*- coding: utf-8 -*-
"""
pdf_rename 模块

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

import json
import os
import re
import logging
import shutil
from functools import lru_cache

from openai import OpenAI

from langchain_community.document_loaders import (
    PDFPlumberLoader,
)

from custom_exception import APIException, CopyException

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# DeepSeek配置
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com",
)


def split_title(title):
    """将标题拆分为四个部分"""
    match = re.match(r"^(.*?)(\.\.\.)(.*?)(_.*)$", title)
    if match:
        part1 = match.group(1)
        part2 = match.group(2)
        part3 = match.group(3)
        part4 = match.group(4)
        return part1, part2, part3, part4

    return title, "", "", ""


def load_document(file_path):
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
        return content[:500]  # 返回文档内容的字符串

    print(file_path + f"，不支持的文档类型: '{ext}'")
    return ""


def get_paper_title_with_regx(filename):
    """
    使用正则表达式处理文件名
    :param filename:
    :return:
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


@lru_cache(maxsize=1000)
def get_paper_title_with_deepseek(text, original_title):
    """
    使用LLM模型从文本中提取并补充论文标题

    :param text: 从PDF中提取的文本内容
    :param original_title: 原始文件名中的标题部分
    :return: 补充完整的论文标题
    """

    part1, part2, part3, _ = split_title(original_title)
    part5 = ""
    system_prompt = f"""
        **背景**：  
        你是一名文件命名助手，需要根据输入的论文文本内容，将标题补充完整。
                
        >>>>>>>>>>>>>>>>>>>>>  
        **规则：**  
        请根据以下规则从文本中补充论文的准确标题：  
        - 【只返回】最终的论文标题，【不得包含】其他任何内容。  
        - 完整标题与输入标题相似，但可能存在【省略】或【不完整】的情况。
        - 【完整提取】标题，若语义相近的标题跨越多行，说明可能存在【副标题】，请一并提取，
        使用【冒号】分隔主副标题。
        - 【不得包含】作者名、机构名、期刊名等内容。 
        - 根据从文本内容中识别到的标题，更新{part2}，更新后的{part2}中不得包含...符号。
        - 将更新后的{part2}内容放入{part5}中。
        - 注意{part3}内容输出的完整，不要忽略该部分的输出整合。
        - 最终输出标题中不得包含空格字符。
        - 输出的论文标题必须为中文。
                
        **输出标题：**  
        - 以JSON格式输出: ["title": "{part1}{part5}{part3}"]
        
    """

    user_prompt = f"""
    文本内容：
    {text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=messages,
            response_format={"type": "json_object"},
        )

        renamed_title = json.loads(response.choices[0].message.content)
        return renamed_title.get("title", "")

    except APIException as e:
        logging.error("Error calling API: %s", str(e))
        return None


def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    return re.sub(r'[<>:"/\\|?*]', "", filename)


def rename_pdf_files(folder_path, output_path):
    """
    重命名指定文件夹中的PDF文件
    :param folder_path: 待处理的文件夹路径
    :param output_path: 处理后的文件夹路径
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            processed_name = get_paper_title_with_regx(filename)

            # 使用大语言模型处理非正常的文件名
            if processed_name is None:
                try:
                    # 从PDF文件中提取文本
                    pdf_text = load_document(file_path)
                    # 提取文件名
                    original_title = os.path.splitext(filename)[0]
                    # 使用大语言模型提取标题
                    paper_title = get_paper_title_with_deepseek(
                        pdf_text, original_title
                    )
                    if paper_title:
                        new_filename = sanitize_filename(paper_title) + ".pdf"
                    else:
                        logging.warning("Could not extract title for %s", filename)
                        continue
                except APIException as e:
                    logging.error("Error processing %s: %s", filename, str(e))
                    continue

            elif processed_name == os.path.splitext(filename)[0]:
                # 跳过正常标题的文件
                logging.info("Skipping: %s (already a valid title)", filename)
                continue
            else:
                # 直接使用处理后的文件名
                new_filename = processed_name + ".pdf"

            new_file_path = os.path.join(output_path, new_filename)

            # 检查输出目录中是否已经存在符合命名要求的PDF文件
            if os.path.exists(new_file_path):
                logging.info(
                    "File already exists and is correctly named: %s", new_filename
                )
                continue

            # 确保输出目录存在
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))

            # 将新命名的文件复制到新文件夹中
            try:
                shutil.copy2(file_path, new_file_path)
                logging.info("Copied: %s -> %s", filename, new_filename)
            except CopyException as e:
                logging.error("Error copying %s: %s", filename, str(e))

            # logging.info("Renamed: %s -> %s", filename, new_filename)


def main():
    """
    主函数
    :return:
    """
    # 使用示例
    folder_path = "test_pdf_file"
    output_path = "renamed_pdf_files"
    rename_pdf_files(folder_path, output_path)


if __name__ == "__main__":
    main()

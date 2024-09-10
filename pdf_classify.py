# -*- coding: utf-8 -*-
"""
PDF文件分类设计思路：
1. 读取指定文件夹下的PDF文件。
2. 读取PDF的文件名，使用数组存储文件名。
3. 使用大语言模型（DeepSeek）根据文件名对各PDF文件进行初步分类，使用JSON格式存储分类结果。
4. 根据分类结果，将PDF文件复制移动到对应的文件夹中。

TODO：添加多轮对话的支持，一次性数据输入太多，需要分批输入。
TODO：让大语言模型反思自己的分类结果，提高分类准确性。
TODO：处理异常情况：
"""
import glob
import json
import os
import shutil
from functools import lru_cache

from langchain_community.document_loaders import PDFPlumberLoader
from openai import OpenAI

# DeepSeek配置
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com",
)

CACHE_FILE = "pdf_data_cache.json"

def load_pdfs_name_and_content(directory):
    """
    读取指定文件夹下的PDF文件名和内容
    :param directory: 文件夹路径
    :return: 包含文件名和内容的列表
    """
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    pdf_files.sort()
    pdf_data = []

    for file_path in pdf_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        content = load_document(file_path)
        pdf_data.append({"filename": file_name, "content": content})

    return pdf_data

def load_document(file_path):
    """
    解析PDF文件，返回文档内容字符串
    :param file_path: 文档文件路径
    :return: 返回文档内容的字符串
    """
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    content = "\n".join([doc.page_content for doc in documents])
    return content[:500]  # 返回文档内容的前1000个字符


def classify_pdfs_with_llm(pdf_data):
    """
    使用大语言模型根据文件名对PDF文件进行初步分类
    :param pdf_files_names: PDF文件名列表
    :return: 分类结果
    """
    # print(pdf_data)
    # pdf_data = list(pdf_data_tuple)
    print("开始分类PDF文件...")
    system_prompt = """
任务描述:
- 你将收到一组同一主题的学术论文PDF文件名以及每篇论文的标题和前1000个字符内容(JSON格式)。
- 你的任务是自动发现这些论文涉及的具体研究对象，然后根据研究对象进行研究主题分类。
- 研究主题分类不能直接从文件名中推断出来，需要通过深入分析每篇论文的摘要、标题、关键词来发现。
- 研究主题分类不能太泛，需要让用户能够清晰地了解每个主题类别的研究内容。
- 通过深入分析每篇论文的内容(摘要、关键词)和标题，找出其所属的可能研究主题，并将所有文件归类到相应的主题类别中。
---
输入:
[{'filename':'xxx', 'contents':'xxx'}...]
---
输出:
输出应为一个JSON格式的对象, 包含以下内容:
- 主题分类: 一个包含所有论文分类的对象，每个对象的键是主题类别，值是一个包含该类别下所有论文的列表。
    
---    
要求:
- 分析内容：深入分析每篇论文的标题和前1000个字符的文本，自主发现隐藏的主题类别。无需依赖任何预设的分类标签。
- 分类：将所有输入文件准确归类到它们所属的主题类别中。每篇论文标题(filename)必须归类到唯一的主题类别中。
- 输出格式：输出结果必须为JSON格式。
"""

    user_prompt = f"""
PDF文件名和前500个字符内容：
{pdf_data}    
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
    )

    print(response.choices[0].message.content)
    # classified_results = json.loads(response.choices[0].message.content)
    # return classified_results

def save_cache(data, cache_file):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def move_pdfs_based_on_classification(data,
                                      source_folder,
                                      destination_folder):
    for category, titles in data['主题分类'].items():
        category_path = os.path.join(destination_folder, category)
        os.makedirs(category_path, exist_ok=True)

        for title in titles:
            sanitized_title = title + '.pdf'
            source_file = os.path.join(source_folder, sanitized_title)
            destination_file = os.path.join(category_path, sanitized_title)
            if os.path.exists(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {sanitized_title} to {category_path}")
            else:
                print(f"File not found: {sanitized_title}")


if __name__ == "__main__":
    # 尝试从缓存文件中加载pdf_data
    source_folder = r"C:\Users\afkxw\Desktop\看论文\本体\本体构建"
    target_folder = r"C:\Users\afkxw\Desktop\看论文\本体\本体构建\classified_pdf_files"

    pdf_data = load_cache(CACHE_FILE)
    if pdf_data is None:
        print("缓存文件不存在，从文件夹加载pdf_data")
        # 如果缓存不存在，则读取指定文件夹下的PDF文件名和内容
        pdf_data = load_pdfs_name_and_content(f"{source_folder}")

        # 将pdf_data缓存到文件中
        save_cache(pdf_data, CACHE_FILE)
    else:
        print("从缓存中加载pdf_data")

    # 将pdf_data转换为元组以便使用lru_cache
    # pdf_data_tuple = tuple(tuple(item.items()) for item in pdf_data)

    # 使用大语言模型根据文件名对PDF文件进行初步分类
    classify_data = classify_pdfs_with_llm(pdf_data)

    move_pdfs_based_on_classification(classify_data,
                                      f"{source_folder}",
                                      f"{target_folder}")

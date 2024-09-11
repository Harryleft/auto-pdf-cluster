# -*- coding: utf-8 -*-
"""
基于大语言模型的PDF文件自动化分类设计思路：
1. 读取指定文件夹下的PDF文件。
2. 读取PDF的文件名，使用数组存储文件名。
3. 使用大语言模型（DeepSeek）根据文件名对各PDF文件进行初步分类，使用JSON格式存储分类结果。
4. 根据分类结果，将PDF文件复制移动到对应的文件夹中。
"""
import glob
import json

import os
import shutil

from langchain_community.document_loaders import PDFPlumberLoader
from openai import OpenAI

import preprocess_title_with_kmeans

# DeepSeek配置
deepseek_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/",
)

CACHE_FILE = "pdf_article_names.json"
PDF_SOURCE_DIR = r"C:\Users\afkxw\Desktop\看论文\本体\本体构建"
PDF_TARGET_DIR = r"C:\Users\afkxw\Desktop\看论文\本体\本体构建\classified_pdf_files"


def load_pdf_names_and_content(directory):
    """
    读取指定文件夹下的PDF文件名和内容
    :param directory: 文件夹路径
    :return: 包含文件名和内容的列表
    """
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    pdf_files.sort()
    pdf_names = [
        os.path.splitext(os.path.basename(file_path))[0] for file_path in pdf_files
    ]
    return pdf_names


def load_document_content(file_path):
    """
    解析PDF文件，返回文档内容字符串
    :param file_path: 文档文件路径
    :return: 返回文档内容的字符串
    """
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    content = "\n".join([doc.page_content for doc in documents])
    return content[:500]


def classify_pdfs_with_llm(pdf_names):
    """
    使用大语言模型根据文件名对PDF文件进行多轮分类
    :param pdf_names: PDF文件名列表
    :return: 分类结果
    """
    system_prompt = """
    任务描述:
    - 你是一位科研助理，将收到一组已经通过初步聚类的学术论文。
    - 你的任务是基于这个初步聚类，逐步逐步(Step by Step)地结合每篇论文的所属聚类和标题内容，进行更精确的主题分类。
    - 请考虑聚类的结果，但不要完全依赖它。如果你认为某篇论文应该属于不同的类别，请进行相应的调整。
    - 研究主题分类应该具体且有信息量，让用户能清晰地了解每个类别的研究内容。
    - 不要限制于示例，你可以根据需要添加更多的主题类别，如应用研究、理论研究等。
    - 每个主题分类下的文件数量不低于3篇，同时主题分类数量不多于10个，不少于5个。
    - 重要：如果某篇论文的主题与其他论文显著不同，或者你无法确定其合适的分类，请将其放入"未分类"类别。不要强行将所有论文都分类。

    输入:
    - KMeans聚类结果使用"0"、"1"、"2"等数字表示每个类别。
    - 每篇论文的标题，使用数组存储。例如: {"0":['filename1', 'filename2'],  "1":['filename3', 'filename4']...}

    输出:
    输出应为一个JSON格式的对象, 包含以下内容:
    - 主题分类: 一个包含所有论文分类的对象，每个对象的键是主题类别，值是一个包含该类别下所有论文的列表。
    - 论文的文件名需要保持输入时的原样，不需要对其进行修改。
    - 例如: {'主题分类': {'类别1': ['filename1', 'filename2'], '类别2': ['filename3', 'filename4']},
            '未分类': ['filename5', 'filename6']}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"文件名的聚类结果信息：{json.dumps(pdf_names, ensure_ascii=False)}",
        },
    ]
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
    )
    initial_classification = json.loads(response.choices[0].message.content)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(initial_classification, ensure_ascii=False),
        }
    )

    # 第二轮：根据第一轮分类结果反思
    messages.append(
        {
            "role": "user",
            "content": "请反思（Reflection）上述主题分类结果的合理性。"
                       "确保每个有效主题类别至少包含3个标题，并且有效主题类别数量在5到10个之间。"
                       "同时，仔细考虑是否有标题应该被归类为'未分类'。确保所有输入的文件名都得到处理。",
        }
    )
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
    )
    second_classification = json.loads(response.choices[0].message.content)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(second_classification, ensure_ascii=False),
        }
    )

    # 第三轮优化分类结果
    messages.append(
        {
            "role": "user",
            "content": "请再次检查一次分类结果，确保所有要求都被满足。特别注意：1) 是否有论文被强行分类到不太合适的类别中？2) 是否有论文的主题与其他论文显著不同？如果有，请将这些论文移至'未分类'列表。确保输出包含'主题分类'和'未分类'两个顶级键，即使'未分类'为空。确保所有输入的文件名都得到处理。",
        }
    )
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
    )
    final_classification = json.loads(response.choices[0].message.content)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(final_classification, ensure_ascii=False),
        }
    )

    # 处理"未分类"文献
    unclassified_papers = final_classification.get("未分类")

    print("未分类文献有：", unclassified_papers)

    if unclassified_papers:
        messages.append(
            {
                "role": "user",
                "content": f"请对未分类的文献逐步逐步地开展归类，尝试将它们加入到现有主题分类中或创建新的主题分类。未分类文献：{json.dumps(unclassified_papers, ensure_ascii=False)}",
            }
        )
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
        )
        unclassified_classification = json.loads(response.choices[0].message.content)
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(unclassified_classification, ensure_ascii=False),
            }
        )

        # 优化未分类文献的分类
        messages.append(
            {
                "role": "user",
                "content": "请优化上述未分类文献的分类结果，尽量将它们整合到现有类别中，或在必要时创建新的合适类别。",
            }
        )
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
        )
        optimized_unclassified_classification = json.loads(
            response.choices[0].message.content
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    optimized_unclassified_classification, ensure_ascii=False
                ),
            }
        )

        # 最终确认未分类文献的分类
        messages.append(
            {
                "role": "user",
                "content": "请最后检查一次未分类文献的分类结果，确保它们被合理地分类或整合到现有类别中。如果仍有无法分类的文献，请将它们保留在'未分类'类别中。",
            }
        )
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
        )
        final_unclassified_result = json.loads(response.choices[0].message.content)

        # 整合未分类文献的分类结果到最终分类中
        for category, papers in final_unclassified_result.get("主题分类", {}).items():
            if category in final_classification["主题分类"]:
                final_classification["主题分类"][category].extend(papers)
            else:
                final_classification["主题分类"][category] = papers

        # 更新"未分类"类别
        final_classification["未分类"] = final_unclassified_result.get("未分类", [])

    if "未分类" not in final_classification:
        final_classification["未分类"] = []

    return final_classification


def save_to_cache(data, cache_file):
    """将预处理数据保存至本地"""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_from_cache(cache_file):
    """从本地加载预处理数据"""
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def move_pdfs_to_classified_folders(
    classification_data, source_folder, destination_folder
):
    """
    根据分类结果移动PDF文件到对应的文件夹
    :param classification_data: 分类结果数据
    :param source_folder: 源文件夹路径
    :param destination_folder: 目标文件夹路径
    """
    for category, titles in classification_data["主题分类"].items():
        category_path = os.path.join(destination_folder, category)
        os.makedirs(category_path, exist_ok=True)

        for title in titles:
            pdf_filename = title + ".pdf"
            source_file = os.path.join(source_folder, pdf_filename)
            destination_file = os.path.join(category_path, pdf_filename)
            if os.path.exists(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {pdf_filename} to {category_path}")
            else:
                print(f"File not found: {pdf_filename}")

    if "未分类" in classification_data:
        unclassified_path = os.path.join(destination_folder, "未分类")
        os.makedirs(unclassified_path, exist_ok=True)
        for title in classification_data["未分类"]:
            pdf_filename = title + ".pdf"
            source_file = os.path.join(source_folder, pdf_filename)
            destination_file = os.path.join(unclassified_path, pdf_filename)
            if os.path.exists(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {pdf_filename} to {unclassified_path}")
            else:
                print(f"File not found: {pdf_filename}")


def scan_and_move_pdfs_back(source_folder, destination_folder):
    """
    扫描各分类文件夹中的所有PDF文件，然后将其移动到原文件夹
    :param source_folder: 原文件夹路径
    :param destination_folder: 分类文件夹路径
    """
    for root, dirs, files in os.walk(destination_folder):
        for file in files:
            if file.endswith(".pdf"):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(source_folder, file)
                if os.path.exists(source_file):
                    shutil.move(source_file, destination_file)
                    print(f"Moved: {file} back to {source_folder}")
                else:
                    print(f"File not found: {file}")
    print("移动回去完成！")

    # 移动完毕后删除空文件夹
    delete_empty_folders(destination_folder)


def delete_empty_folders(directory):
    """
    删除指定目录下的所有空文件夹
    :param directory: 目录路径
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Deleted empty folder: {dir_path}")


def process_pdfs_cluster():
    # 查看本地是否有缓存文件，如果有则加载，否则重新加载PDF文件名
    if not os.path.exists(CACHE_FILE):
        print("1. 开始加载PDF文件名")
        pdf_names = load_pdf_names_and_content(PDF_SOURCE_DIR)
        print(pdf_names)
        save_to_cache(pdf_names, CACHE_FILE)
    else:
        print("1. 开始加载PDF文件名")
        pdf_names = load_pdf_names_and_content(PDF_SOURCE_DIR)
        print(pdf_names)

    # 检查本地是否存在聚类结果的JSON文件：clustered_files.json
    # 如果存在，则直接加载，否则重新聚类
    if not os.path.exists("clustered_files.json"):
        print("2. 开始聚类。。。")
        preprocess_title_with_kmeans.preprocess_with_kmeans(pdf_names)
        print("聚类完成！")
    else:
        print("2. 开始加载聚类结果。。。")
        kmeans_results = load_from_cache("clustered_files.json")

    print("开始利用大语言模型分类。。。")
    llm_classification_results = classify_pdfs_with_llm(kmeans_results)

    print(llm_classification_results)
    print("分类完成！")

    print("开始移动PDF文件。。。")
    move_pdfs_to_classified_folders(
        llm_classification_results, PDF_SOURCE_DIR, PDF_TARGET_DIR
    )
    print("移动完成！")


if __name__ == "__main__":
    # 开始分类
    # process_pdfs_cluster()

    # 分类结果不理想，想把文件移回去
    scan_and_move_pdfs_back(PDF_SOURCE_DIR, PDF_TARGET_DIR)

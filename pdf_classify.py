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
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)

CACHE_FILE = "pdf_article_names.json"
PDF_SOURCE_PATH = r"C:\Users\afkxw\Desktop\看论文\本体\本体构建"
PDF_TARGET_PATH = r"C:\Users\afkxw\Desktop\看论文\本体\本体构建\classified_pdf_files"


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
        # content = load_document(file_path)
        # pdf_data.append({"filename": file_name, "content": content})
        # pdf_data.append({"filename": file_name})
        pdf_data.append(file_name)

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
    return content[:500]


def classify_pdfs_with_llm(pdf_data):
    """
    使用大语言模型根据文件名对PDF文件进行多轮分类
    :param pdf_data: PDF文件名和初步聚类结果
    :return: 分类结果
    """
    print("开始分类PDF文件...")

    system_prompt = """
    任务描述:
    - 你将收到一组已经通过KMeans算法初步聚类的学术论文。
    - 你的任务是基于这个初步聚类，结合每篇论文的所属聚类和标题内容，进行更精确的主题分类。
    - 请考虑KMeans聚类的结果，但不要完全依赖它。如果你认为某篇论文应该属于不同的类别，请进行相应的调整。
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

    要求:
    - 深入分析每篇论文的标题，结合KMeans聚类结果进行更精确的分类。
    - 如果发现KMeans聚类结果不准确，请根据论文内容进行调整。
    - 确保每篇论文都被分类到最合适的主题类别中，或放入未分类列表。
    - 不要强行将难以分类的论文归入某个类别，应该将其放入"未分类"列表。
    - 输出结果必须为JSON格式，包含"主题分类"和"未分类"两个顶级键。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"文件名的聚类结果信息：{json.dumps(pdf_data, ensure_ascii=False)}",
        },
    ]

    # 第一轮：初步分类
    print("开始第一轮分类...")
    response = client.chat.completions.create(
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
    print("初步分类结果：", initial_classification)
    print("==================================")



    # 第二轮：优化分类
    print("开始第二轮分类...")
    messages.append(
        {
            "role": "user",
            "content": "请检查并优化上述分类结果，确保每个类别至少包含3篇论文，并且类别数量在5到10个之间。如有需要，请调整类别名称使其更具体和信息量。同时，仔细考虑是否有论文应该被归类为'未分类'。",
        }
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
        stream=False,
    )
    optimized_classification = json.loads(response.choices[0].message.content)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(optimized_classification, ensure_ascii=False),
        }
    )
    print("第二轮优化分类结果：", optimized_classification)
    print("==================================")

    # 第三轮：最终确认和调整
    print("开始最终确认和调整...")
    messages.append(
        {
            "role": "user",
            "content": "请最后检查一次分类结果，确保所有要求都被满足。特别注意：1) 是否有论文被强行分类到不太合适的类别中？2) 是否有论文的主题与其他论文显著不同？如果有，请将这些论文移至'未分类'列表。确保输出包含'主题分类'和'未分类'两个顶级键，即使'未分类'为空。",
        }
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
        stream=False,
    )
    final_classification = json.loads(response.choices[0].message.content)
    print("最终分类结果：", final_classification)
    print("==================================")

    # 第四轮：处理"未分类"文献
    print("开始处理未分类文献...")
    unclassified = final_classification.get("未分类", [])
    print("未分类的文献有：", unclassified)
    if unclassified:
        messages.append({"role": "user",
                         "content": f"请对以下未分类的文献step by step开展归类，"
                                    f"尝试将它们加入到现有主题分类中或创建新的主题分类。"
                                    f"未分类文献：{json.dumps(unclassified, ensure_ascii=False)}"})

        # 子轮次1：初步分类未分类文献
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
        )
        subround_1 = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant",
                         "content": json.dumps(subround_1, ensure_ascii=False)})
        print("初步对未分类文献的分类结果：", subround_1)

        # 子轮次2：优化未分类文献的分类
        print("开始优化未分类文献的分类...")
        messages.append({"role": "user",
                         "content": "请优化上述未分类文献的分类结果，尽量将它们整合到现有类别中，或在必要时创建新的合适类别。"})
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
        )
        subround_2 = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant",
                         "content": json.dumps(subround_2, ensure_ascii=False)})

        print("初步对未分类文献的分类结果：", subround_2)

        # 子轮次3：最终确认未分类文献的分类
        print("开始最终确认未分类文献的分类...")
        messages.append({"role": "user",
                         "content": "请最后检查一次未分类文献的分类结果，确保它们被合理地分类或整合到现有类别中。如果仍有无法分类的文献，请将它们保留在'未分类'类别中。"})
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
        )
        final_unclassified_result = json.loads(
            response.choices[0].message.content)

        print("对未分类文献的最终分类结果：", final_unclassified_result)

        # 整合未分类文献的分类结果到最终分类中
        # 整合未分类文献的分类结果到最终分类中
        for category, papers in final_unclassified_result.get("主题分类",
                                                              {}).items():
            if category in final_classification["主题分类"]:
                final_classification["主题分类"][category].extend(papers)
            else:
                final_classification["主题分类"][category] = papers

        # 更新"未分类"类别
        final_classification["未分类"] = final_unclassified_result.get("未分类", [])

    return final_classification

def save_cache(data, cache_file):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def move_pdfs_based_on_classification(data, source_folder, destination_folder):
    # 移动分类好的文件
    for category, titles in data["主题分类"].items():
        category_path = os.path.join(destination_folder, category)
        os.makedirs(category_path, exist_ok=True)

        for title in titles:
            sanitized_title = title + ".pdf"
            source_file = os.path.join(source_folder, sanitized_title)
            destination_file = os.path.join(category_path, sanitized_title)
            if os.path.exists(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {sanitized_title} to {category_path}")
            else:
                print(f"File not found: {sanitized_title}")

    # 移动未分类的文件
    if "未分类" in data:
        unclassified_path = os.path.join(destination_folder, "未分类")
        os.makedirs(unclassified_path, exist_ok=True)
        for title in data["未分类"]:
            sanitized_title = title + ".pdf"
            source_file = os.path.join(source_folder, sanitized_title)
            destination_file = os.path.join(unclassified_path, sanitized_title)
            if os.path.exists(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {sanitized_title} to {unclassified_path}")
            else:
                print(f"File not found: {sanitized_title}")


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
        pdf_names = load_pdfs_name_and_content(PDF_SOURCE_PATH)
        print(pdf_names)
        save_cache(pdf_names, CACHE_FILE)
    else:
        print("1. 开始加载PDF文件名")
        pdf_names = load_pdfs_name_and_content(PDF_SOURCE_PATH)
        print(pdf_names)

    # 检查本地是否存在聚类结果的JSON文件：clustered_files.json
    # 如果存在，则直接加载，否则重新聚类
    if not os.path.exists("clustered_files.json"):
        print("2. 开始聚类。。。")
        preprocess_title_with_kmeans.preprocess_with_kmeans(pdf_names)
        print("聚类完成！")
    else:
        print("2. 开始加载聚类结果。。。")
        kmeans_results = load_cache("clustered_files.json")

    print("开始利用大语言模型分类。。。")
    llm_cluster_results = classify_pdfs_with_llm(kmeans_results)

    print(llm_cluster_results)
    print("分类完成！")

    print("开始移动PDF文件。。。")
    # move_pdfs_based_on_classification(
    #     llm_cluster_results, PDF_SOURCE_PATH, PDF_TARGET_PATH
    # )
    print("移动完成！")


if __name__ == "__main__":
    # 开始分类
    process_pdfs_cluster()

    # 分类结果不理想，想把文件移回去
    # scan_and_move_pdfs_back(PDF_SOURCE_PATH, PDF_TARGET_PATH)

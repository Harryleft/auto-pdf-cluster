# -*- coding: utf-8 -*-
"""对PDF文件进行自动化分类

基于大语言模型的PDF文件自动化分类设计思路：
1. 读取指定文件夹下的PDF文件。
2. 读取PDF的文件名，使用数组存储文件名。
3. 使用大语言模型（DeepSeek）根据文件名对各PDF文件进行初步分类，使用JSON格式存储分类结果。
4. 根据分类结果，将PDF文件复制移动到对应的文件夹中。
"""
import json

import os
import shutil

from openai import OpenAI

import preprocess_title_with_kmeans
from config import FORMATED_PDF_NAME_FOLDER, PDF_NAME_CACHE_FILE, \
    PDF_CLASSIFICATION_DIR
from load_pdf import load_pdf_names

# DeepSeek配置
deepseek_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/",
)

def classify_pdfs_with_llm(pdf_names):
    """
    使用大语言模型根据文件名对PDF文件进行多轮分类
    :param pdf_names: PDF文件名列表
    :return: 分类结果
    """
    system_prompt = """
    任务描述:
    - 你是一位科研助理，将收到一组已经通过初步聚类的学术论文题目。\n
    - 你的任务是基于这个数据集，逐步(Step by Step)地开展适合人类阅读和理解的主题分类。\n
    - 请考虑聚类的结果，但不要完全依赖它。如果你认为某篇论文应该属于不同的类别，请进行相应的调整。\n
    - 研究主题分类应该具体且有信息量，让用户能清晰地了解每个类别的研究内容。\n
    - 不要限制于示例，你可以根据需要添加更多的主题类别，如应用研究、理论研究等。\n
    - 每个主题分类下的文件数量不低于3篇，同时主题分类数量不多于10个，不少于5个。\n
    - 检查序号的连续性，确保没有遗漏的文件，如有遗漏，则先放入未分类中，然后再进行处理。\n
    - 重要：如果某篇论文的主题与其他论文显著不同，或者你无法确定其合适的分类，请将其放入"未分类"类别。不要强行将所有论文都分类。\n

    输入:\n
    - KMeans聚类结果使用"0"、"1"、"2"等数字表示每个类别。\n
    - 每篇论文的题名结构为：[[序号]]_[[文献题名]]，例如："01_The Title of the Paper"。\n
    - 输入的数据集样式为： {"0":['01_filename1', '02_filename2'],  "1":['03_filename3', '04_filename4']...}\n

    输出:\n
    输出应为一个JSON格式的对象, 包含以下内容:\n
    - 主题分类: 一个包含所有论文分类的对象，每个对象的键是主题类别，值是一个包含该类别下所有论文的列表。\n
    - 论文的文件名需要保持输入时的原样，不需要对其进行修改。\n
    - 输出样式: {'主题分类': {'类别1': ['filename1', 'filename2'], '类别2': ['filename3', 'filename4']},
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
    print("第一轮LLM分类结果：\n", initial_classification)

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
    print("第二轮LLM分类结果：\n", second_classification)

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
    print("第三轮LLM分类结果：\n", final_classification)

    # 处理"未分类"文献
    unclassified_papers = final_classification.get("未分类")

    print("开始处理当前LLM中的未分类文献：", unclassified_papers)

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
        print("继续优化未分类文献的分类")
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
        print("未分类文献的优化分类结果：\n", optimized_unclassified_classification)

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
        print("最终未分类文献的分类结果：\n", final_unclassified_result)
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

    print("最终分类结果：\n", final_classification)

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
    for root, _, files in os.walk(destination_folder):
        for file in files:
            if file.endswith(".pdf"):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(source_folder, file)
                if os.path.exists(source_file):
                    shutil.move(source_file, destination_file)
                    print(f"Moved: {file} back to {source_folder}")
                else:
                    print(f"File not found: {file}")
    print("所有PDF文件已经移动回原文件夹")

    # 移动完毕后删除空文件夹
    delete_empty_folders(destination_folder)


def delete_empty_folders(directory):
    """
    删除指定目录下的所有空文件夹
    :param directory: 目录路径
    """
    for root, dirs, _ in os.walk(directory, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
    print("原分类的空文件夹已经被删除")


def process_pdfs_cluster():
    """
    处理PDF文件的分类
    :return:
    """
    # 查看本地是否有缓存文件，如果有则加载，否则重新加载PDF文件名
    if not os.path.exists(PDF_NAME_CACHE_FILE):
        print(">> 开始加载PDF文件名")
        pdf_names = load_pdf_names(FORMATED_PDF_NAME_FOLDER)
        print(pdf_names)
        save_to_cache(pdf_names, PDF_NAME_CACHE_FILE)
        print(">> 已将PDF文件名缓存至本地")
    else:
        print(">> 检测到本地存在可利用的PDF文件名JSON文件，尝试加载")
        pdf_names = load_pdf_names(PDF_NAME_CACHE_FILE)
        print(pdf_names)

    # 使用KMeans方法对文献题名进行初步聚类
    if not os.path.exists("clustered_files.json"):
        print(">> 预处理：开始使用KMeans方法对文献题名进行初步")
        preprocess_title_with_kmeans.preprocess_with_kmeans(pdf_names)
        print(">> 预处理：结束")
    else:
        print(">> 预处理：检测到本地存在可利用的聚类结果，尝试加载")
        kmeans_results = load_from_cache("clustered_files.json")
        print(">> 预处理：加载聚类结果完成")

    # 利用LLM分类文献题名
    print(">> LLM处理：开始利用LLM分类文献题名")
    llm_classification_results = classify_pdfs_with_llm(kmeans_results)
    print(">> LLM处理：利用LLM分类文献题名任务完成！")

    # 移动PDF文件到分类文件夹
    print(">> 移动操作：根据LLM分类结果移动本体PDF文件")
    move_pdfs_to_classified_folders(
        llm_classification_results, FORMATED_PDF_NAME_FOLDER, PDF_CLASSIFICATION_DIR
    )
    print(">> 移动操作：完成")

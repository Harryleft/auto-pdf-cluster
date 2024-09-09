# -*- coding: utf-8 -*-
import os
import re
from PyPDF2 import PdfReader
import dashscope

# 设置DashScope API密钥
dashscope.api_key = "sk-279ed50b460648aab17c93307f17627a"
qwen_model = "qwen-turbo"

def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本"""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_paper_title(text):
    """使用Qwen-Turbo模型从文本中提取论文标题"""
    prompt = f"""
    请从以下文本中提取论文的准确标题。注意以下几点：
    1. 只返回标题本身，不要有其他任何内容。
    2. 标题通常出现在文章的开头，是字体最大的文字。
    3. 标题可能跨越多行，请完整提取。
    4. 不要包含作者名、机构名等信息，只提取标题。
    5. 如果有副标题，请一并提取，用冒号分隔主副标题。

    文本内容：
    {text[:500]}
    """

    response = dashscope.Generation.call(
        model=qwen_model,
        prompt=prompt,
        max_tokens=200
    )

    if response.status_code == 200:
        return response.output.text.strip()
    else:
        print(f"Error: {response.code}, {response.message}")
        return None


def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    return re.sub(r'[<>:"/\\|?*]', '', filename)


def process_filename(filename):
    """处理文件名，根据不同情况进行相应的处理"""
    # 移除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]

    # 检查是否符合特定模式（以《开头的文件名）
    if name_without_ext.startswith('《'):
        return None  # 返回None表示需要使用大语言模型处理
    else:
        # 移除末尾的作者姓名（假设格式为 "_作者姓名"）
        return re.sub(r'_[^_]+$', '', name_without_ext)


def rename_pdf_files(folder_path):
    """重命名指定文件夹中的PDF文件"""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)

            # 处理文件名
            processed_name = process_filename(filename)

            if processed_name is None:
                # 需要使用大语言模型处理
                pdf_text = extract_text_from_pdf(file_path)
                paper_title = get_paper_title(pdf_text)
                if paper_title:
                    new_filename = sanitize_filename(paper_title) + '.pdf'
                else:
                    print(f"Could not extract title for {filename}")
                    continue
            else:
                # 直接使用处理后的文件名
                new_filename = processed_name + '.pdf'

            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {str(e)}")

# 使用示例
folder_path = "test_pdf_file"
rename_pdf_files(folder_path)

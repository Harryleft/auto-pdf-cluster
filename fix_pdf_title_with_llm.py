# -*- coding: utf-8 -*-
"""
利用LLM对PDF文件名进行完整补充
"""
import json
import logging
import os
import re

from openai import OpenAI

from custom_exception import APIException


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

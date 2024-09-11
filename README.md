# AutoPDFCluster

---

## 1. 项目简介
利用大语言模型(LLM)的强大能力,为中文学术论文提供智能化的命名和分类解决方案。

主要特性:
- 流程自动化: 批量处理大量论文,节省时间和人力
- 命名规范化: 统一论文命名格式,提高后续阅读效率
- 智能语义分类: 结合[DeepSeek大模型](https://platform.deepseek.com/)，对文献根据主题自动分类,便于后续阅读和管理

## 2. 实现思路

1. 提取文件名：读取指定文件目录下的PDF文件名
2. 文件名规范化：利用LLM对PDF文件名进行规范化处理
3. 文件名预处理：使用KMeans对PDF文件名进行初步的聚类处理
4. LLM主题分类：借助LLM参考聚类结果进行主题分类
5. 本地文件整理：根据LLM的主题分类结果对本地的PDF文件进行整理

## 3. 使用方法
1. 修改`config.py`中的PDF文件目录（注意使用绝对路径，如`D:/PDF`）
```python
# 主要涉及如下几个路径的修改：
# 修改：PDF文件存放的原始文件夹
# 例如：D:\PDF
SOURCE_PDF_FOLDER = r"[添加你的PDF原始文件夹绝对路径]"

# 修改：命名规范化后的 PDF 文件夹
# 例如：D:\PDF\formated_pdf
FORMATED_PDF_NAME_FOLDER = r"[添加规范化处理后的PDF文件夹绝对路径]"

# 修改PDF文件主题分类文件夹
# 例如：D:\PDF\classification_pdf
PDF_CLASSIFICATION_DIR = r"[添加PDF分类文件夹绝对路径]"
```
2. 运行`main.py`文件
3. 如果对聚类结果不满意，可以运行`main.py`文件中的`scan_and_move_pdfs_back`函数，可将PDF文件移回原格式化目录`FORMATED_PDF_NAME_FOLDER`中
- 注意：在使用`scan_and_move_pdfs_back`时，需要将其他函数注释掉

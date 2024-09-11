# AutoPDFCluster使用说明文档

## 1. 使用场景

从中国知网下载的部分PDF文件中，文件名包含`...`符号，导致无法完整显示论文题目。为了解决这个问题，`AutoPDFCluster`结合大语言模型（LLM）的优势，用于规范化命名和聚类PDF学术论文。
## 2. 总体思路

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

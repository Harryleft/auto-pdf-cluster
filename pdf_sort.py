# -*- coding: utf-8 -*-
"""
为指定文件夹下的PDF文件添加序号前缀
"""
import os
import glob


def batch_rename_pdfs(directory):
    """
    为PDF文件添加序号前缀
    :param directory:
    :return:
    """
    print(f">> Renaming PDF files in {directory}...")
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    pdf_files.sort()
    for index, file_path in enumerate(pdf_files, start=1):
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        new_name = f"{str(index).zfill(2)}_{base_name}"

        new_path = os.path.join(dir_name, new_name)

        os.rename(file_path, new_path)
        print(f">> Renamed {file_path} to {new_path}")

    print(">> Successfully Renamed ALL PDF Files.")


if __name__ == "__main__":
    # 使用示例
    batch_rename_pdfs(r"[你的文件夹绝对路径]")

# -*- coding: utf-8 -*-
"""为指PDF文件添加前缀序号
"""
import os
import glob
import re
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def is_filename_valid(filename):
    """检查文件名是否已符合规范"""
    return re.match(r"^\d+_.*\.pdf$", filename) is not None


def add_prefix_to_pdf(directory):
    """
    为PDF文件添加序号前缀，如果文件名已符合规范则不添加
    :param directory: 要处理的目录路径
    """
    logging.info(f"开始处理文件夹: {directory}")

    try:
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        if not pdf_files:
            logging.warning(f"在 {directory} 中没有找到PDF文件")
            return

        pdf_files.sort()
        renamed_count = 0
        skipped_count = 0

        for index, file_path in enumerate(pdf_files, start=1):
            try:
                dir_name = os.path.dirname(file_path)
                base_name = os.path.basename(file_path)

                if is_filename_valid(base_name):
                    logging.info(f"文件已符合规范，跳过: {base_name}")
                    skipped_count += 1
                    continue

                new_name = f"{str(index).zfill(2)}_{base_name}"
                new_path = os.path.join(dir_name, new_name)

                if os.path.exists(new_path):
                    logging.warning(f"目标文件已存在，跳过: {new_path}")
                    skipped_count += 1
                    continue

                os.rename(file_path, new_path)
                logging.info(f"重命名: {file_path} 为 {new_path}")
                renamed_count += 1

            except OSError as e:
                logging.error(f"重命名文件时出错: {file_path}. 错误: {str(e)}")

        logging.info(
            f"处理完成. 重命名: {renamed_count} 个文件, 跳过: {skipped_count} 个文件."
        )

    except Exception as e:
        logging.error(f"处理目录时出错: {directory}. 错误: {str(e)}")

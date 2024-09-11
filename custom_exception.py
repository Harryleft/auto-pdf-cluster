# -*- coding: utf-8 -*-
"""
自定义异常类
"""
class APIException(Exception):
    """自定义异常类，用于处理无法获取API的异常"""
    def __init__(self, message="无法获取API"):
        self.message = message
        super().__init__(self.message)

class CopyException(Exception):
    """自定义异常类，用于处理复制错误的异常"""
    def __init__(self, message="复制文件时发生错误"):
        self.message = message
        super().__init__(self.message)

class MoveException(Exception):
    """自定义异常类，用于处理移动错误的异常"""
    def __init__(self, message="移动文件时发生错误"):
        self.message = message
        super().__init__(self.message)

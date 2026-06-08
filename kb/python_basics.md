# Python 基础知识

## 数据类型
- str（字符串）：引号包裹，如 "hello"
- int（整数）：如 42
- float（浮点数）：如 3.14
- list（列表）：有序可变，如 [1, 2, 3]
- tuple（元组）：有序不可变，如 (1, 2, 3)
- dict（字典）：键值对，如 {"name": "张三", "age": 20}
- set（集合）：无序去重，如 {1, 2, 3}

## 控制流
- if/elif/else：条件判断
- for：遍历可迭代对象
- while：条件循环
- try/except：异常处理

## 函数
使用 def 关键字，支持参数和默认值。
返回值用 return。

## 文件操作
```python
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()
```
模式：r（读）、w（写）、a（追加）。

## 模块和包
import 导入模块，pip 安装第三方包。
虚拟环境：venv 或 conda 管理依赖。

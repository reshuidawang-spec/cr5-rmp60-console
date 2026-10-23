"""
通用几何工具函数集
====================

本模块提供整个代码库中共享的基础三维几何运算函数，包括：

  - normalize： 三维向量归一化（单位向量转换）
  - add：       两个等长向量的逐元素加法
  - scale：     向量与标量的数乘

这些函数不依赖 NumPy，仅使用 Python 标准库 math，适合在需要轻量级
几何运算的场景中使用（如计算逼近方向、构造偏移矢量等）。
"""
import math


def normalize(vector, label="vector"):
    """
    三维向量归一化。

    将任意非零三维向量转换为单位向量（长度为 1）。
    若输入向量长度接近零（≤ 1e-12），视为零向量并抛出 ValueError。

    数学原理
    --------
    给定向量 v = (vx, vy, vz)，其长度为 ||v|| = sqrt(vx² + vy² + vz²)。
    单位向量 u = v / ||v|| = (vx/||v||, vy/||v||, vz/||v||)。

    参数
    ----
    vector : tuple/list 或类似
        三维向量 (x, y, z)。
    label : str
        向量的名称/标签，在抛出异常时用于提示。

    返回
    ----
    list[float] 长度为 3
        单位向量。

    异常
    ----
    ValueError : 当输入向量为零向量时抛出。
    """
    length = math.sqrt(sum(v * v for v in vector))
    if length <= 1e-12:
        raise ValueError(f"{label} cannot be zero")
    return [v / length for v in vector]


def add(a, b):
    """
    两个等长向量的逐元素加法。

    返回一个新向量 c，其中 c[i] = a[i] + b[i]。

    常用于将法兰盘位置与旋转后的偏移量相加，或将多个位移矢量叠加。

    参数
    ----
    a, b : list[float]
        两个等长向量。

    返回
    ----
    list[float]
        逐元素和。
    """
    return [x + y for x, y in zip(a, b)]


def scale(vector, value):
    """
    向量数乘（标量乘法）。

    将向量 v 的每个分量乘以标量 s，得到 s * v = (s*vx, s*vy, s*vz)。

    常用于将方向向量缩放至指定长度，或将球半径乘以逼近方向的
    负值以计算偏移量。

    参数
    ----
    vector : list[float]
        输入向量。
    value : float
        缩放系数。

    返回
    ----
    list[float]
        缩放后的向量。
    """
    return [v * value for v in vector]

import math

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plot
import numpy as np


def creat_data(nums_of_data):
    """
    # 生成数据函数
    :param nums_of_data:
    :return:
    """
    # 利用np.random.randn(1, 2)生成一个满足标准正太分布的坐标点
    # 这里将单条数据设计为一行三列，colum_1、colum_2分别为散点的x、y坐标，colum_3为预留的打标数据
    coordinates = np.random.randn(nums_of_data, 3)
    for row in coordinates:
        tag_entry(row)
    return coordinates


def tag_entry(array):
    """
    打标，对训练数据分类
    :param x:
    :param y:
    :return:
    """
    x_axis = array[0]
    y_axis = array[1]
    # 根据划分标准要求，以圆内外进行分类，所及计算散点距离圆心的距离
    # 如果距离超过1则划分为A类，距离小于1划分为B类
    radius = math.sqrt(x_axis ** 2 + y_axis ** 2)
    if radius < 1:
        array[2] = 0
    else:
        array[2] = 1


def plot_data(data, title):
    """
    将最终结果以散点图的形式呈现出来
    :param data:
    :param title:
    :return:
    """
    color = []
    for i in data:
        if i[2] == 0:
            color.append("blue")
        else:
            color.append("red")

    plot.scatter(data[:, 0], data[:, 1], c=color)
    plot.title(title)
    plot.show()


if __name__ == "__main__":
    data = creat_data(100)
    plot_data(data, "demo")
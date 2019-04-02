#!/usr/bin/python
# -*- coding: utf-8 -*-
# 图像处理工具箱
import sys
import os
import math
import itertools
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class OpencvIo:
    # 数据IO操作
    def __init__(self):
        self.__util = Util()  # 初始化工具类

    def imread(self, path, option=1):
        # 读入图像
        try:
            if not os.path.isfile(os.path.join(os.getcwd(), path)):
                # os.getcwd()获取当前的路经
                raise IOError('File is not exist')
            src = cv.imread(path, option)  # 读数据
            # flag=-1按解码得到的方式读入图像
            # flag=0按照单通道的方式读入图像，即灰白图像
            # flag=1按照三通道方式读入图像，即彩色图像
        except IOError:
            raise
        except:
            print('Arugment Error : Something wrong')
            sys.exit()
        return src

    def imshow(self, src, name='a image'):
        # 采用pyplot显示图片
        plt.figure(name)
        plt.imshow(src,cmap="gray")
        plt.pause(20)
        plt.close()

    def imshow_array(self, images):
        i=0  # 计数
        for x in images:
            plt.figure(str(i))
            img=np.uint8(self.__util.normalize_range(x))  # 0～255规范化
            # cv2读取图片格式问题
            plt.imshow(img,cmap="gray")
            plt.pause(5)
            plt.close()
            i=i+1


class Util:
    def normalize_range(self, src, begin=0, end=255):
        # 对数据表示范围大小进行规范化
        dst = np.zeros((len(src), len(src[0])))
        amin, amax = np.amin(src), np.amax(src)
        for y, x in itertools.product(range(len(src)), range(len(src[0]))):
            if amin != amax:
                dst[y][x] = (src[y][x] - amin) * (end - begin) / (amax - amin) + begin
            else:
                dst[y][x] = (end + begin) / 2
        return dst

    def normalize(self, src):
        # 根据数据情况进行拉伸或者平滑
        src = self.normalize_range(src, 0., 1.)  # [0,1]规范化
        amax = np.amax(src)  # amax：每一个元素中的最大值找到，然后把这些最大值找出来组成一个数组
        maxs = []
        for y in range(1, len(src) - 1):
            for x in range(1, len(src[0]) - 1):
                val = src[y][x]
                if val == amax:
                    continue
                if val > src[y - 1][x] and val > src[y + 1][x] and val > src[y][x - 1] and val > src[y][x + 1]:
                    # 当前点亮度大于上下左右
                    maxs.append(val)

        if len(maxs) != 0:
            src *= math.pow(amax - (np.sum(maxs) / np.float64(len(maxs))), 2.)
        return src
#!/usr/bin/python
# -*- coding: utf-8 -*-
# 显著图

import math
import itertools
import cv2 as cv
import numpy as np
from utils import Util


class GaussianPyramid:
    # 高斯金字塔
    def __init__(self, src):
        self.maps = self.__make_gaussian_pyramid(src)  # 生成高斯金字塔

    def __make_gaussian_pyramid(self, src):
        # 高斯金字塔，输出一系列特征图
        maps = {'intensity': [],
                'colors': {'b': [], 'g': [], 'r': [], 'y': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}
        amax = np.amax(src)  # 最大值
        b, g, r = cv.split(src)
        for x in range(1, 9):
            b, g, r = map(cv.pyrDown, [b, g, r])  # 先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
            if x < 2:
                continue
            buf_its = np.zeros(b.shape)
            buf_colors = list(map(lambda _: np.zeros(b.shape), range(4)))  # b, g, r, y
            for y, x in itertools.product(range(len(b)), range(len(b[0]))):
                buf_its[y][x] = self.__get_intensity(b[y][x], g[y][x], r[y][x])  # 亮度图
                # 生成4个色差图
                buf_colors[0][y][x],buf_colors[1][y][x],buf_colors[2][y][x],buf_colors[3][y][x]=self.__get_colors(b[y][x], g[y][x], r[y][x], buf_its[y][x], amax)
            maps['intensity'].append(buf_its)
            for (color, index) in zip(sorted(maps['colors'].keys()), range(4)):
                maps['colors'][color].append(buf_colors[index])
            for (orientation, index) in zip(sorted(maps['orientations'].keys()), range(4)):
                maps['orientations'][orientation].append(self.__conv_gabor(buf_its, np.pi * index / 4))  # 方向特征图
        return maps

    def __get_intensity(self, b, g, r):
        # 获取亮度
        return (np.float64(b) + np.float64(g) + np.float64(r)) / 3.

    def __get_colors(self, b, g, r, i, amax):
        # 获取色差图
        b, g, r = list(map(lambda x: np.float64(x) if (x > 0.1 * amax) else 0., [b, g, r]))  # 将小于最大值的十分之一的数据置零
        nb, ng, nr = list(map(lambda x, y, z: max(x - (y + z) / 2., 0.), [b, g, r], [r, r, g], [g, b, b]))  # 两两相减形成3个色差图
        ny = max(((r + g) / 2. - math.fabs(r - g) / 2. - b), 0.)  # 第四个色差图
        if i != 0.0:
            return list(map(lambda x: x / np.float64(i), [nb, ng, nr, ny]))
        else:
            return nb, ng, nr, ny

    def __conv_gabor(self, src, theta):
        # Gabor是一个用于边缘提取的线性滤波器
        # 其频率和方向表达与人类视觉系统类似
        # 能够提供良好的方向选择和尺度选择特性，而且对于光照变化不敏感，因此十分适合纹理分析。
        kernel = cv.getGaborKernel((8, 8), 4, theta, 8, 1)
        return cv.filter2D(src, cv.CV_32F, kernel)


class FeatureMap:
    # 获取差分特征图
    def __init__(self,srcs):
        self.maps = self.__make_feature_map(srcs)

    def __make_feature_map(self, srcs):
        cs_index = ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6))  # 用来计算center-surround的index值（不同的层代表着中心或者周边）
        maps = {'intensity': [],
                'colors': {'bg': [], 'ry': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}
        for c, s in cs_index:
            maps['intensity'].append(self.__scale_diff(srcs['intensity'][c], srcs['intensity'][s]))  # 差分操作
            for key in maps['orientations'].keys():
                maps['orientations'][key].append(self.__scale_diff(srcs['orientations'][key][c], srcs['orientations'][key][s]))  # 差分操作
            for key in maps['colors'].keys():
                maps['colors'][key].append(self.__scale_color_diff(
                    srcs['colors'][key[0]][c], srcs['colors'][key[0]][s],
                    srcs['colors'][key[1]][c], srcs['colors'][key[1]][s]
                ))  # 进行差分
        return maps

    def __scale_diff(self, c, s):
        # 差分操作
        c_size = tuple(reversed(c.shape))
        return cv.absdiff(c, cv.resize(s, c_size, None, 0, 0, cv.INTER_NEAREST))

    def __scale_color_diff(self,c1,s1,c2,s2):
        # 差分操作
        c_size = tuple(reversed(c1.shape))
        return cv.absdiff(c1 - c2, cv.resize(s2 - s1, c_size, None, 0, 0, cv.INTER_NEAREST))


class ConspicuityMap:
    # 显著图
    def __init__(self, srcs):
        self.maps = self.__make_conspicuity_map(srcs)

    def __make_conspicuity_map(self, srcs):
        util = Util()
        intensity = self.__scale_add(list(map(util.normalize, srcs['intensity'])))  # 亮度累加
        for key in srcs['colors'].keys():
            srcs['colors'][key] = list(map(util.normalize, srcs['colors'][key]))
        color = self.__scale_add([srcs['colors']['bg'][x] + srcs['colors']['ry'][x] for x in range(len(srcs['colors']['bg']))])  # 颜色累加
        orientation = np.zeros(intensity.shape)
        for key in srcs['orientations'].keys():
            orientation += self.__scale_add(list(map(util.normalize, srcs['orientations'][key])))  # 方向累加爱
        return {'intensity': intensity, 'color': color, 'orientation': orientation}  #三个属性下的显著图

    def __scale_add(self, srcs):
        # 同一个属性差分特征图进行累加
        buf = np.zeros(srcs[0].shape)
        for x in srcs:
            buf += cv.resize(x, tuple(reversed(buf.shape)))
        return buf


class SaliencyMap:
    # 显著图
    def __init__(self, src):
        self.gp = GaussianPyramid(src)
        self.fm = FeatureMap(self.gp.maps)
        self.cm = ConspicuityMap(self.fm.maps)
        self.map = cv.resize(self.__make_saliency_map(self.cm.maps), tuple(reversed(src.shape[0:2])))

    def __make_saliency_map(self, srcs):
        util = Util()
        srcs = list(map(util.normalize, [srcs[key] for key in srcs.keys()]))
        # 将各个属性下的显著图等比例相加
        return srcs[0] / 3. + srcs[1] / 3. + srcs[2] / 3.

#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from saliency_map import SaliencyMap
from utils import OpencvIo

filename="/home/yhq/Desktop/saliency/images/test2.jpg"
oi = OpencvIo()
src = oi.imread(filename)
sm = SaliencyMap(src)
oi.imshow_array([sm.map])
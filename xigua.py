#!/usr/bin/python
#coding:utf8
import numpy as np
import os
import sys
from mltoolkits import *
import logging
import mygmm

logger = logging.getLogger(sys.argv[0])
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
  logger.debug('start...')

  params = {}
  params['cluster_num'] = 3
  #2元高斯混合模型
  params['attr_num'] = 2
  #迭代次数
  params['iters'] = 200


  #加载数据
  train_file = './data/xiguashuju.txt'
  train_data = np.genfromtxt(train_file, delimiter = ',', dtype = np.float)
  logger.debug('训练样本:%s', train_data)

  #训练样本x
  train_x = train_data

  #创建模型
  gmm = mygmm.MyGMM(params)

  #自己加载初始mu点
 # mu0 = [0.403, 0.237]
 # mu1 = [0.714, 0.346]
 # mu2 = [0.532, 0.472]
  #gmm.LoadMu({0:mu0, 1:mu1, 2:mu2})
  #开始训练模型
  gmm.Train(train_x)
  #预测样本聚类
  pred_y = gmm.Predict(train_x)
  logger.debug('pred:\n%s', pred_y)

  #显示
  import matplotlib.pyplot as plt
  markers = ['o', '*', '.', '+']
  colors = ['red', 'yellow', 'blue', 'black']
  for j in range(0, len(train_x)):
    plt.plot(train_x[j][0], train_x[j][1], color = colors[pred_y[j]], marker = markers[pred_y[j]])
  plt.show()



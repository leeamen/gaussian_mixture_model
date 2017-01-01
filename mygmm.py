#!/usr/bin/python
#coding:utf8
import numpy as np
import os
import sys
from mltoolkits import *
import logging

logger = logging.getLogger(sys.argv[0])
logger.setLevel(logging.DEBUG)

class MyGMM(object):
  def __init__(self, params = {}):
    #高斯成分个数
    self.k = params['cluster_num']
    #n维样本空间
    self.n = params['attr_num']
    logger.debug('混合成分个数:%d', self.k)
    logger.debug('高斯维度:%d', self.n)

    self.iters = params['iters']
    logger.debug('迭代次数:%d', self.iters)

    #初始化参数
    #混合系数
    self.alphas = {}
    #协方差矩阵,对角线为方差
    self.sigmas = {}
    #均值向量
    self.mus = {}
    self.__init_alpha_sigma(self.k)

  #初始化模型参数 alpha, sigma, mu是随机选取样本作为均值向量，因此需要在train时候初始化
  def __init_alpha_sigma(self, k):
    alpha = np.array([1./k] * k, dtype = np.float)
    sigma = np.empty((0, self.n))
    #初始化协方差矩阵
    for i in range(0, self.n):
      v = np.zeros(self.n)
      v[i] = 0.1
      sigma = np.vstack((sigma, v))
    logger.debug('初始化的参数sigma:\n%s', sigma)
    #k个高斯
    for i in range(0, k):
      self.sigmas[i] = sigma
      self.alphas[i] = alpha[i]

  def __init_mu(self, x):
    mus = myrandom.RandomSampleList(x, self.k)
    logger.debug('随机选择mu:\n%s', mus)
    for i in range(0, self.k):
      self.mus[i] = mus[i]

  def __calc_gaussian(self, x, n, mu, sigma):
    return myequation.GaussFunc(x, n, mu, sigma)

  def __cacl_posterior(self, x):
    m = len(x)
    k = self.k
    n = self.n
    '''计算后延概率
      posterior[j][i] : p(X[j]|mu[i],sigma[i])
    '''
    posterior = np.zeros((m, k))
    for j in range(0, m):
      for i in range(0, k):
        mu = self.mus[i]
        sigma = self.sigmas[i]
        alpha = self.alphas[i]
        posterior[j][i] = alpha * self.__calc_gaussian(x[j], n, mu, sigma)
      posterior[j] = posterior[j] / np.sum(posterior[j])
#    logger.debug('后验概率:\n%s', posterior)
    return posterior

  '''
    计算log似然函数值
  '''
  def __calc_log_like(self, x):
    m = len(x)
    k = self.k
    n = self.n

    ll = 0.
    for j in range(0, m):
      s = 0.
      for i in range(0, k):
        mu = self.mus[i]
        sigma = self.sigmas[i]
        alpha = self.alphas[i]
        g = self.__calc_gaussian(x[j], n, mu, sigma)
        s += g * alpha
      ll += np.log(s)
    return ll
  def __calc_new_mu(self, x, posterior):
    k = self.k
    m = len(x)
    n = self.n
    #计算新的mu
    for i in range(0, k):
      weight_all = np.sum(posterior[:,i])
      new_x = posterior[:,i].reshape((m, 1)) * x
      mu = np.sum(new_x, 0)
      self.mus[i] = mu / weight_all

  def __calc_new_sigma(self, x, posterior):
    k = self.k
    m = len(x)
    n = self.n
    for i in range(0, k):
      weight_all = np.sum(posterior[:,i])
      pos1 = posterior[:,i].reshape((m, 1)) * (x - self.mus[i])
      pos2 = x - self.mus[i]
      sigma = np.dot(pos1.T, pos2) / weight_all
      self.sigmas[i] = sigma
  def __calc_new_alpha(self, x, posterior):
    k = self.k
    m = len(x)
    n = self.n
    for i in range(0, k):
      weight_all = np.sum(posterior[:,i])
      alpha = 1./m * weight_all
      self.alphas[i] = alpha

  '''迭代
  '''
  def __iterate(self, x):
    iters = self.iters
    for i in range(0, iters):
      #后验概率 m x k
      posterior = self.__cacl_posterior(x)
      self.__calc_new_mu(x, posterior)
      self.__calc_new_sigma(x, posterior)
      self.__calc_new_alpha(x, posterior)
      #计算似然性
      ll = self.__calc_log_like(x)
      logger.info('Iteration %d: Log-likelihood:%f', i+1, ll)

  def LoadMu(self, mus = {}):
    k = self.k
    for i in range(0, k):
      self.mus[i] = mus[i]

  '''训练模型
  '''
  def Train(self, x):
    #x必须为二维矩阵
    self.__init_mu(x)
    #迭代
    self.__iterate(x)

  '''预测
  '''
  def Predict(self, x):
    posterior = self.__cacl_posterior(x)
    logger.debug('posterior:\n%s', posterior)
    pred = myfunction.HArgmax(posterior, len(posterior))
    return pred

if __name__ == '__main__':
  logger.debug('start...')

  params = {}
  params['cluster_num'] = 3
  #2元高斯混合模型
  params['attr_num'] = 2
  #迭代次数
  params['iters'] = 100


  #加载数据
  train_file = './data/xiguashuju.txt'
  train_data = np.genfromtxt(train_file, delimiter = ',', dtype = np.float)
  logger.debug('训练样本:%s', train_data)

  #训练样本x
  train_x = train_data

  #创建模型
  gmm = MyGMM(params)

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

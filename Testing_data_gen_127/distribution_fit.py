# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 20:21:04 2022

@author: zidonghua_30
"""
from distfit import distfit
import matplotlib.pyplot as plt
import numpy as np
X = np.random.normal(0, 2, [100,10])
dist=distfit(distr='norm')
dist.fit_transform(X)
dist.plot()
plt.show()





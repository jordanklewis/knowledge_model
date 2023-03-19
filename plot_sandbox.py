# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:42:27 2023

@author: Lewis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
from collections import Counter
import seaborn as sns
import pandas as pd

#####################################
# figure 1 in report
#####################################
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-20, 20, 0.01)
  
# Calculating mean and standard deviation
mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)

plt.figure(figsize=(5,3))
plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
plt.xlabel('Knowledge Categories')
plt.ylabel('Knowledge Quantity')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.show()

##########
# figure 2
##########
mean_std = [(0, 6), (15,5), (-20,4)]
bw_adjust = .2
dist1 = []
for m in mean_std:
    dist1 += list(np.random.normal(m[0], m[1], 1000))
    
plt.figure(figsize=(5,3))
sns.kdeplot(data=dist1, bw_adjust=bw_adjust)
plt.xlabel('Knowledge Categories')
plt.ylabel('Knowledge Quantity')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.show()


##########
# figure 3
##########
mean_std2 = [(5, 4), (20,5), (-15,6)]
plt.figure(figsize=(5,3))
sns.kdeplot(data=dist1, bw_adjust=bw_adjust, label='person 1')
dist2 = []
for m in mean_std2:
    dist2 += list(np.random.normal(m[0], m[1], 1000))
sns.kdeplot(data=dist2, bw_adjust=.25, color='red', label='person 2')
plt.legend()
plt.xlabel('Knowledge Categories')
plt.ylabel('Knowledge Quantity')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.show()

##########
# figure 4
##########
mean_std = [(0, 6), (15,5), (-20,4)]
mean_std2 = [(5, 4), (20,5), (-15,6)]
plt.figure(figsize=(5,3))
sns.kdeplot(data=dist1, bw_adjust=bw_adjust, label='person 1', color='black')
sns.kdeplot(data=dist2, bw_adjust=bw_adjust, color='black', label='person 2')
plt.legend()
plt.xlabel('Knowledge Categories')
plt.ylabel('Knowledge Quantity')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.show()

##########
# figure 5 and 6
##########
# systems engineering department distribution

max_know = 100000
know_cat_ct = int(max_know/10)
know_cats = list(np.arange(know_cat_ct).astype(int)
                      - int(know_cat_ct/2))
se_know = np.random.normal(0,know_cat_ct/5,max_know).astype(int)

# wrap to +/- self.know_cat_ct/2 to create department overlap
se_know %= know_cat_ct
se_know[se_know>=know_cat_ct/2] -= know_cat_ct
se_know[se_know<-know_cat_ct/2] += know_cat_ct
se_know = Counter(se_know)

plt.figure(figsize=(10,6))
plt.bar(se_know.keys(), se_know.values(), width=1)
plt.xlim([min(know_cats)-0.5, max(know_cats)+0.5])
plt.minorticks_on()
plt.xticks(np.arange(min(know_cats), max(know_cats)+1, know_cat_ct/10))
plt.tick_params(axis='x', which='minor', length=5, width=1)
plt.tick_params(axis='x', which='major', length=7, width=2)
plt.xlabel('Knowledge Categories')
plt.ylabel('Knowledge Quantity')
# plt.grid()
plt.show()
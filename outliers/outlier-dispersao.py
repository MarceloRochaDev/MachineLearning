# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:38:48 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
base = base.dropna() # apaga registros não preenchidos
base.loc[base.age<0,'age'] = 40.92
#income x age
import matplotlib.pyplot as plt
plt.scatter(base.iloc[:,1], base.iloc[:,2]) # em 1 ta o income, e em 2 ta o age

# income x loan
plt.scatter(base.iloc[:,1],base.iloc[:,3])

# age x loan
plt.scatter(base.iloc[:,2],base.iloc[:,3])


base_census = pd.read_csv('census.csv')
# age x final weight
plt.scatter(base_census.iloc[:,0],base_census.iloc[:,2])
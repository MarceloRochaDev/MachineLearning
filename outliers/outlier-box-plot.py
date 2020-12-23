# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:38:48 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
base = base.dropna() # apaga registros não preenchidos

import matplotlib.pyplot as plt

#outlisers age
plt.boxplot(base.iloc[:,2],showfliers=True) 
outliers_age = base[(base.age<-20)] # age<20 foi tirado a partir da visualização dos outliers no boxplot

#outliers loan
plt.boxplot(base.iloc[:,3])
outlisers_loan = base[(base.loan>13400)]
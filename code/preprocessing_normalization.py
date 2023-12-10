#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openpyxl
from sklearn.preprocessing import MinMaxScaler
import numpy as np

workbook = openpyxl.load_workbook('.../Cancer.xlsx')

sheet = workbook.active

data = []

for row in sheet.iter_rows(min_row=2, values_only=True):
    data.append([row[1]])

data = np.array(data)

scaler = MinMaxScaler(feature_range=(1, 10))

normalized_data = scaler.fit_transform(data)

normalized_data = normalized_data.round(0).astype(int)

for element in normalized_data:
    print(element[0])


'''
  A script converts log files to one excel file.
'''

import os
import pandas as pd
import numpy as np

path = './'
files= os.listdir(path)
item = ''
dic = []
for file_name in files:
    id = file_name[-13:-4]
    fp = open(file_name)
    try:
        data = fp.readlines()
        for d in data:
            if "test score: {" in d:  # may one log exists two experimental results
                items = d.replace('test score:', '').split(', ')
                dic.append([id, float(items[0][9:15]), float(items[1][9:15]), float(items[2][10:16]),
                            float(items[3][11:17]), float(items[4][11:17]), float(items[5][10:16]),
                            float(items[8][13:19]), float(items[11][7:13])])
    except:
        print('error id: {}'.format(file_name))
dic = sorted(dic, key=(lambda x:x[0])) # sort by id

writer = pd.ExcelWriter('read result.xlsx')
data_pd = pd.DataFrame(dic)
data_pd.to_excel(writer, 'sheet1', float_format='%.4f', header=False, index=False)
writer.save()
writer.close()

print("finish")
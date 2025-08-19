import pandas
import numpy as np
from os import path

with open(path.join('data','key.txt'), 'r') as k:
    key = int(k.read())

data = {'start_x' : list(range(key)), 'start_y' : list(range(key)),
        'finish_x' : list(range(key)), 'finish_y' : list(range(key)),
        'true_beginning' : list(range(key)), 'true_finish' : list(range(key))}
for i in range(key):
    df = pandas.read_csv(path.join('data','Record{!s}.csv'.format(i)),index_col=0,dtype=np.float64)
    start_x = df.at[0,'x_pos']
    start_y = df.at[0,'y_pos']
    last = df.last_valid_index()
    end_x = df.at[last,'x_pos']
    end_y = df.at[last,'y_pos']

    data['start_x'][i] = start_x
    data['start_y'][i] = start_y
    data['finish_x'][i] = end_x
    data['finish_y'][i] = end_y
    data['true_beginning'][i] = min(df['x_pos'].where(df['x_pos']!=start_x).first_valid_index(),
                                    df['y_pos'].where(df['y_pos']!=start_y).first_valid_index())-1
    data['true_finish'][i] = max(df['x_pos'].where(df['x_pos']!=end_x).last_valid_index(),
                                 df['y_pos'].where(df['y_pos']!=end_y).last_valid_index())+1

real_position = pandas.DataFrame(data=data,dtype=int)
real_position.to_csv(path.join('data','real_position.csv'))
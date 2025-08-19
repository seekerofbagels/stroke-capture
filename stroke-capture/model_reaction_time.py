import pandas
from keras.models import Sequential
from keras.layers import Dense
from os import path

with open(path.join('data','key.txt'),'r') as k:
    key = int(k.read())

test_size = round(key/10)
data = pandas.read_csv(path.join('data','real_position.csv'),index_col=0)
x = data.loc[0:key-test_size, ['start_x', 'start_y', 'finish_x', 'finish_y']]
y = data.loc[0:key-test_size,'true_beginning']
test_x = data.loc[key-test_size:key, ['start_x', 'start_y', 'finish_x', 'finish_y']]
test_y = data.loc[key-test_size:key,'true_beginning']

model = Sequential()
model.add(Dense(8, input_shape=(4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
model.fit(x,y, epochs=50,batch_size=10)
model.save(path.join('models','reaction.keras'))

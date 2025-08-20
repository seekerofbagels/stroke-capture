import pandas
import numpy as np
import keras
import tensorflow as tf
from os import path


class DataImporter:

    def __init__(self,forward_direction=True):
        with open(path.join('data','key.txt'),'r') as k:
            self.key = int(k.read())
        self.real_position = pandas.read_csv(path.join('data','real_position.csv'),index_col=0)
        self.import_columns = ['x_pos','y_pos']
        self.forward_direction = forward_direction
    
    def read_record(self,record_index)->tf.Tensor:
        data = pandas.read_csv(path.join('data','Record{!s}.csv'.format(record_index)),index_col=0,dtype=np.float64)
        start_index = self.real_position.at[record_index,'true_beginning']
        end_index = self.real_position.at[record_index,'true_finish']
        data = data.loc[start_index:end_index,self.import_columns]
        if not self.forward_direction:
            data = data[::-1]
        end_point = data.iloc[-1]
        return tf.constant(data - end_point)


class InputProcessor:

    def __init__(self,input_width,output_width,offset,data_importer=None):
        self.input_width = input_width
        self.output_width = output_width
        self.offset = offset
        self.total_window_size = input_width + offset
        self.input_slice = slice(0,input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.output_start = self.total_window_size - self.output_width
        self.output_slice = slice(self.output_start,None)
        self.output_indices = np.arange(self.total_window_size)[self.output_slice]
        self.data_split_proportion = {'train' : 0.64, 'validate' : 0.16, 'test' : 0.2}
        self.data_importer = data_importer
        if data_importer == None:
            self.data_importer = DataImporter()
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Output indices: {self.output_indices}',
        ])
    
    def split_series(self,dataset):
        inputs = dataset[:,self.input_slice,:]
        outputs = dataset[:,self.output_slice,:]
        inputs.set_shape([1,self.input_width,2])
        outputs.set_shape([1,self.output_width,2])
        return inputs, outputs
    
    def window_generator(self,data):
        index = 0
        data_length = data.shape[0]
        while index <= data_length - self.total_window_size:
            yield tf.expand_dims(data[index:index+self.total_window_size],axis=0)
            index += 1
    
    def model_input(self,records,number_of_columns):
        def gen():
            for record in map(self.window_generator,records):
                for window in record:
                    yield window
        dataset = tf.data.Dataset.from_generator(generator=gen, output_signature=tf.TensorSpec(shape=(1,self.total_window_size,number_of_columns),dtype=np.float64))
        dataset = dataset.map(self.split_series)
        return dataset
    
    @property
    def train(self):
        data_index_begin = 0
        data_index_end = round(self.data_importer.key*self.data_split_proportion['train'])
        records = map(self.data_importer.read_record,range(data_index_begin,data_index_end))
        return self.model_input(records,len(self.data_importer.import_columns))
    
    @property
    def validate(self):
        data_index_begin = round(self.data_importer.key*self.data_split_proportion['train'])
        data_index_end = round(self.data_importer.key*(self.data_split_proportion['train'] + self.data_split_proportion['validate']))
        records = map(self.data_importer.read_record,range(data_index_begin,data_index_end))
        return self.model_input(records,len(self.data_importer.import_columns))
    
    @property
    def test(self):
        data_index_begin = round(self.data_importer.key*(self.data_split_proportion['train'] + self.data_split_proportion['validate']))
        data_index_end = self.data_importer.key
        records = map(self.data_importer.read_record,range(data_index_begin,data_index_end))
        return self.model_input(records,len(self.data_importer.import_columns))


input_width = 4
output_width = 1
offset = 1
forward_importer = DataImporter(forward_direction=True)
backward_importer = DataImporter(forward_direction=False)
ip = InputProcessor(data_importer=forward_importer,input_width=input_width,output_width=output_width,offset=offset,)

epochs = 20
def compile_and_fit(model,input_process):
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanAbsoluteError()])
    train = input_process.train.cache()
    validate = input_process.validate.cache()
    history = model.fit(train,epochs=epochs,validation_data=validate)
    return

naive_model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(units=8,activation='relu'),
    keras.layers.Dense(units=2),
    keras.layers.Reshape([1,-1])
])

#compile_and_fit(naive_model,ip)
#naive_model.save('models/naive.keras')

#loaded_model = keras.models.load_model('models/naive.keras')
#test = ip.test
#loaded_model.evaluate(test)

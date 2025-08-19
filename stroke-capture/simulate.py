import pyglet
import pandas
import numpy as np
import keras
import tensorflow as tf
import random
from os import path

input_width = 4

class Instance:
    
    def __init__(self,color=(255,255,255)):
        self.color = color
        self.reset()
    
    def reset(self):
        self.batch = pyglet.graphics.Batch()
        self.line_list = []

    def generate_next_position(self):
        raise NotImplementedError()

    def generate_lines(self,next_position_gen):
        g = next_position_gen()
        previous_position = g.__next__()
        for position in g:
            yield pyglet.shapes.Line(previous_position[0],previous_position[1],position[0],position[1],color=self.color)
            previous_position = position
    
    def compile_line_generator(self):
        self.gen = self.generate_lines(self.generate_next_position)

    def construct_lines(self,dt):
        try:
            self.line_list.append(self.gen.__next__())
            self.line_list[-1].batch = self.batch
        except StopIteration:
            pass


class RecordReplay(Instance):

    def read(self,index):
        p = path.join('data','Record{!s}.csv'.format(index))
        self.data = pandas.read_csv(p,index_col=0)
        self.data = np.array(self.data[['x_pos','y_pos']])
    
    def generate_next_position(self):
        index = 0
        for i in range(self.data.shape[0]):
            yield self.data[i]
        index += 1


class Forecaster(Instance):

    def __init__(self,reaction_model_path,model_path,forward_direction=True,color=(255,255,255)):
        Instance.__init__(self,color)
        self.reaction = keras.models.load_model(reaction_model_path,compile=False)
        self.model = keras.models.load_model(model_path,compile=False)
        self.forward_direction = forward_direction
    
    def assign(self,start_position,end_position):
        self.start_position = start_position
        self.end_position = end_position
    
    def relative(self,absolute_position):
        if self.forward_direction:
            return np.subtract(absolute_position,self.end_position)
        else:
            return np.subtract(absolute_position,self.start_position)
    
    def absolute(self,relative_position):
        if self.forward_direction:
            return np.add(relative_position,self.end_position)
        else:
            return np.add(relative_position,self.start_position)
    
    def generate_next_position(self):
        if self.forward_direction:
            first_position = self.relative(self.start_position)
        else:
            first_position = self.relative(self.end_position)
        reaction_frames = self.reaction(tf.reshape(tf.constant([self.start_position,self.end_position],dtype=tf.float32),shape=[1,-1])).numpy()[0][0]
        for i in range(round(reaction_frames)):
            yield self.absolute(first_position)
        start_prediction = tf.tile(np.reshape(first_position,shape=[1,1,-1]),[1,input_width,1])
        
        input = tf.cast(start_prediction,tf.float32)
        for i in range(70):
            new = self.model(input)
            yield self.absolute(tf.reshape(new,shape=[-1]).numpy())
            if tf.math.reduce_sum(tf.math.abs(new))<30:
                break
            t = input[:,1:,:]
            input = tf.concat([t,new],axis=1)


class SeekingForecaster(Instance):

    def __init__(self,reaction_model_path,model_path,color=(255,255,255)):
        Instance.__init__(self,color)
        self.reaction = keras.models.load_model(reaction_model_path,compile=False)
        self.model = keras.models.load_model(model_path,compile=False)
    
    def assign(self,start_position,end_position):
        self.start_position = start_position
        self.end_position = end_position
    
    def relative(self,absolute_position):
        return np.subtract(absolute_position,self.end_position)
    
    def absolute(self,relative_position):
        return np.add(relative_position,self.end_position)
    
    def generate_next_position(self):
        first_position = self.relative(self.start_position)
        reaction_frames = self.reaction(tf.reshape(tf.constant([self.start_position,self.end_position],dtype=tf.float32),shape=[1,-1])).numpy()[0][0]
        for i in range(round(reaction_frames)):
            yield self.absolute(first_position)
        start_prediction = tf.tile(np.reshape(first_position,shape=[1,1,-1]),[1,input_width,1])
        
        input = tf.cast(start_prediction,tf.float32)
        lambd = 4
        r = random.expovariate(lambd/2)
        for i in range(70):
            last = tf.reshape(input[:,-1,:],shape=[-1]).numpy()
            new = tf.reshape(self.model(input),shape=[-1]).numpy()
            vec = np.subtract(new,last)
            norm = - np.linalg.vector_norm(vec)/np.linalg.vector_norm(last)
            centre_pointing = last * norm
            r = max(r*0.6,random.expovariate(lambd))
            if r > 1:
                r = 1
            vec = (1-r)*vec + r*centre_pointing
            new = np.add(last,vec)
            new = tf.reshape(tf.constant(new,dtype=tf.float32),shape=[1,1,-1])
            yield self.absolute(tf.reshape(new,shape=[-1]).numpy())
            if np.linalg.vector_norm(new)<30:
                break
            t = input[:,1:,:]
            input = tf.concat([t,new],axis=1)


with open(path.join('data','key.txt'), 'r') as k:
    key = int(k.read())
iter_index = iter(range(key))
position_dataframe = pandas.read_csv(path.join('data','real_position.csv'),index_col = 0)

r = RecordReplay()
f = Forecaster(reaction_model_path=path.join('models','reaction.keras'),model_path=path.join('models','naive.keras'),color=(255,0,0))
v = Forecaster(reaction_model_path=path.join('models','reaction.keras'),model_path=path.join('models','naive_velocity.keras'),color=(0,255,0))
b = Forecaster(reaction_model_path=path.join('models','reaction.keras'),model_path=path.join('models','naive_backward.keras'),forward_direction=False,color=(0,0,255))
s = SeekingForecaster(reaction_model_path=path.join('models','reaction.keras'),model_path=path.join('models','naive.keras'),color=(255,255,0))
active_instances = [r,f,s]

window = pyglet.window.Window(fullscreen=True)

@window.event
def on_key_press(symbol,modifiers):
    if symbol & pyglet.window.key.SPACE:
        index = iter_index.__next__()
        for instance in active_instances:
            pyglet.clock.unschedule(instance.construct_lines)
            if type(instance) == RecordReplay:
                instance.read(index)
            elif type(instance) == Forecaster or type(instance) == SeekingForecaster:
                instance.assign(start_position=np.array(position_dataframe.loc[index,['start_x','start_y']]),
                                end_position=np.array(position_dataframe.loc[index,['finish_x','finish_y']]))
            
            instance.reset()
            instance.compile_line_generator()
            pyglet.clock.schedule_interval(instance.construct_lines,1/60)

@window.event
def on_draw():
    window.clear()
    for instance in active_instances:
        instance.batch.draw()
    
pyglet.app.run(1/60)
import pyglet
from pyglet.shapes import Circle
import pandas
import numpy as np
import time
from os import path

rng = np.random.default_rng()

window = pyglet.window.Window(fullscreen=True)


class Recording:

    def __init__(self, init_x, init_y, end_x, end_y):
        self.position_x = []
        self.position_y = []
        self.elapsed = []
        self.start_time = time.perf_counter()
        self.init_x = init_x
        self.init_y = init_y
        self.end_x = end_x
        self.end_y = end_y

    def record_data(self,x,y):
        self.position_x.append(x)
        self.position_y.append(y)
        self.elapsed.append(time.perf_counter()-self.start_time)

    def export_data(self):
        with open(path.join('data','key.txt'), 'r+') as k:
            key = int(k.read())
            k.seek(0)
            k.write(str(key+1))
        data = {'time' : self.elapsed, 'x_pos' : self.position_x, 'y_pos' : self.position_y}
        pandas.DataFrame(data).to_csv(path.join('data','Record{!s}.csv'.format(key)))
        menu_data = {'init_x' : self.init_x, 'init_y' : self.init_y, 'end_x' : self.end_x, 'end_y' : self.end_y}
        pandas.DataFrame(menu_data,index=[key]).to_csv(path.join('data','menu.csv'), mode='a', header=False)

    def reset_data(self):
        self.position_x = []
        self.position_y = []
        self.elapsed = []
        self.start_time = time.perf_counter()

    def assign_target(self, end_x, end_y):
        self.init_x = self.end_x
        self.init_y = self.end_y
        self.end_x = end_x
        self.end_y = end_y


sprite = Circle(x=window.width//2, y=window.height//2, radius=25,color=(255,255,255))
started = False
count = 0
mouse_x = 0
mouse_y = 0
recording = Recording(sprite.x, sprite.y, sprite.x, sprite.y)

def start_record():
    pyglet.clock.schedule_once(end_record,60.0)
    global started
    started = True

def end_record(dt):
    pyglet.app.exit()

@window.event
def on_draw():
    window.clear()
    sprite.draw()
    if started == True:
        recording.record_data(mouse_x, mouse_y)

@window.event
def on_mouse_motion(x,y,dx,dy):
    global mouse_x
    global mouse_y
    mouse_x = x
    mouse_y = y

@window.event
def on_mouse_press(x,y,buttons,modifiers):
    if buttons & pyglet.window.mouse.LEFT & ((x,y) in sprite):
        if started == False:
            start_record()
        else:
            global count
            recording.export_data()
            count += 1
        new_x = round(rng.random()*window.width)
        new_y = round(rng.random()*window.height)
        sprite.x = new_x
        sprite.y = new_y
        recording.assign_target(new_x, new_y)
        recording.reset_data()

pyglet.app.run(1/60)
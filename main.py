import tensorflow as tf
from tensorflow.keras import layers, models
from tkinter import *
from tkinter.colorchooser import askcolor
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import random


class Paint(object):
    DEFAULT_PEN_SIZE = 10.0
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.ai = AI()
        self.root = Tk()

        self.pen_button = Button(self.root, text='Pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='Brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='Eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=3)

        self.plot_button = Button(self.root, text='Plot', command=self.plot)
        self.plot_button.grid(row=0, column=5)

        self.c = Canvas(self.root, bg='Black', width=600, height=600)
        self.c.grid(row=1, column=0, columnspan=4)

        self.b = Canvas(self.root, bg='Grey', width=450, height=600)
        self.b.grid(row=1, column=4, columnspan=2)

        self.image = np.zeros(shape=(28, 28))

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 10
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'black' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        image_y = int(np.floor(event.x/600 * 28))
        image_x = int(np.floor(event.y/600 * 28))
        
        if self.old_y:
            image_old_y = int(np.floor(self.old_x/600 * 28))
            image_old_x = int(np.floor(self.old_y/600 * 28))
            
            x_diff = np.abs(image_old_x - image_x)
            y_diff = np.abs(image_old_y - image_y)
            
            if 1 < x_diff <= 3 or 1 < y_diff <= 3:
                self.image[image_old_x : image_x][image_old_y : image_y] = 150

            self.image[image_x+1][image_y+1] = random.randint(100,200)
            self.image[image_x-1][image_y-1] = random.randint(100,200)
        
        self.image[image_x][image_y] = random.randint(155,255)
            
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.c.create_line(0, 0, 600, 600, width=900, fill='black')

        self.image = np.zeros(shape=(28, 28))

    def plot(self):
        fig = Figure(figsize=(4, 4))
        plot1 = fig.add_subplot(111)
        prediction = self.ai.prediction(self.image)
        predicted_label = np.argmax(prediction)
        plot1.grid(False)
        plt.xticks(range(10))
        plt.yticks(range(0, 101, 10))
        plt.xlabel('Prediction')
        print(prediction)
        print(predicted_label)
        thisplot = plt.bar(range(10), 10 * prediction, color="#777777")
        plt.ylim([0, 100])
        
        thisplot[predicted_label].set_color('green')

        canvas = FigureCanvasTkAgg(fig, master=self.b)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, self.b)
        toolbar.update()
        canvas.get_tk_widget().pack()


class AI():

    def __init__(self):
        self.model = 0

        self.mnist = np.load('/Users/mahadhassan/Documents/Coding/Projects/Digit Recognizer/mnist.npz')

        self.train_images = self.mnist['x_train']
        self.train_labels = self.mnist['y_train']

        self.test_images = self.mnist['x_test']
        self.test_labels = self.mnist['y_test']

        self.train_images, self.test_images = self.train_images / 255, self.test_images / 255

        self.create_model()
        self.training()

    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(10))

    def training(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        history = self.model.fit(self.train_images, self.train_labels, epochs=1,
                                 validation_data=(self.test_images, self.test_labels))

        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)

    def prediction(self, test_image):
        test_image = test_image / 255

        test_image = np.expand_dims(test_image, axis=0)

        probability_model = tf.keras.Sequential([self.model,
                                                 tf.keras.layers.Softmax()])

        prediction = probability_model.predict(test_image)

        return prediction


if __name__ == '__main__':
    Paint()

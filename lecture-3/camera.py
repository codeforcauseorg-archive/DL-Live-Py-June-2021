import cv2
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.graphics import Color, Line
from kivy.core.window import Window
from tensorflow.keras.models import load_model
import numpy as np
import os

model = load_model("mnist_model")

class PaintWindow(Label):

    def __init__(self, **kwargs):
        super(PaintWindow, self).__init__(**kwargs)
        self.prev = None
        self.to_crop = True
        self.size_hint = (None, None)  # without this, the size below has no effect
        self.size = (600, 600)

    def on_touch_down(self, touch):
        self.prev = touch.x, touch.y

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos) and self.to_crop:
            with self.canvas:
                Line(points=[self.prev[0], self.prev[1], touch.x, touch.y], width=15)
                self.prev = touch.x, touch.y

            return True


class ControlPanel(GridLayout):

    def clear_screen(self, event):
        self.main_layout.image.canvas.clear()

    def predict_number(self, event):
        Window.screenshot("value.png")
        img = cv2.imread("value0001.png")
        cut = img[:600, :600]
        small = cv2.resize(cut, (28, 28))
        twoD = small.mean(axis=2)
        cv2.imwrite("myfile.png", twoD)
        output = model.predict(np.array([twoD]))
        self.output.text = str((output.argmax(axis=1))[0])

        if os.path.exists("value0001.png"):
            os.remove("value0001.png")

    def __init__(self, main_layout, **kwargs):
        super(ControlPanel, self).__init__(**kwargs)
        self.videowriter = None
        self.rows = 1
        self.main_layout = main_layout
        self.predict = Button(text="Predict")
        self.clear = Button(text="Clear")
        self.output = Label(text="")
        self.output.font_size=150

        self.add_widget(self.predict)
        self.add_widget(self.clear)
        self.add_widget(self.output)

        self.clear.bind(on_press=self.clear_screen)
        self.predict.bind(on_press=self.predict_number)


class MainLayout(GridLayout):

    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)
        self.frame = None
        self.rows = 2
        self.image = PaintWindow()
        self.control = ControlPanel(self)
        self.add_widget(self.image)
        self.add_widget(self.control)


class TestApp(App):

    def build(self):
        return MainLayout()


app = TestApp()
app.run()
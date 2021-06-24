from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout

class MainLayout(GridLayout):

    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)
        self.rows = 2
        self.add_widget(Button(text="I am a button"))
        self.add_widget(Button(text="I am a button"))
        self.add_widget(Button(text="I am a button"))
        self.add_widget(Button(text="I am a button"))
        self.add_widget(Button(text="I am a button"))
        self.add_widget(Button(text="I am a button"))


class TestApp(App):

    def handleClick(self, event):
        print("I am Clicked")

    def build(self):
        return MainLayout()


app = TestApp()
app.run()
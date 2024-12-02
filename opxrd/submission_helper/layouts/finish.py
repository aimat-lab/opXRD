from __future__ import annotations

import os
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

from submission_helper.elements.types import BlackLabel
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from submission_helper.configs import get_true_height, get_true_width
# -------------------------------------------

class FinishLayout(BoxLayout):
    def __init__(self, callback: callable, exit_funct : callable, **kwargs):

        l = 15 * get_true_height() / 900.
        super(FinishLayout, self).__init__(orientation='vertical', size_hint=(1, 0.085),padding=(0, l, 0, l),**kwargs)

        s = 10 * get_true_width() / 1600.
        upper_finish = BoxLayout(orientation='horizontal', size_hint=(1, 0.3),spacing=s)
        note = BlackLabel(text='Target folder:', size_hint=(0.2, 1), font_size=Window.width * 0.02, bold=True)
        self.target_path_input = TextInput(text=f'{os.getcwd()}',
                                           size_hint=(0.7, 1),
                                           font_size=get_true_width() * 0.02,
                                           multiline=False)

        buffer = BlackLabel(text='', size_hint=(0.075, 1))
        self.ok_button = self.get_ok_button()
        self.ok_button.bind(on_press=callback)

        upper_finish.add_widget(note)
        upper_finish.add_widget(self.target_path_input)
        upper_finish.add_widget(self.ok_button)
        upper_finish.add_widget(buffer)

        self.exit_funct = exit_funct
        self.default_font_size = get_true_width() * 0.018
        self.feedback_label = self.get_feedback_label(font_size=self.default_font_size)
        self.feedback_popup = self.get_feedback_popup()

        self.add_widget(upper_finish)


    def show(self):
        self.feedback_popup.open()

    def get_feedback_popup(self):
        container = BoxLayout(orientation='vertical')
        container.add_widget(self.feedback_label)

        button_layout = BoxLayout(size_hint_y=0.2)
        dismiss_button = Button(text='Dismiss', on_press=lambda instance: self.feedback_popup.dismiss())
        exit_button = Button(text='Exit', on_press=self.exit_funct)

        button_layout.add_widget(dismiss_button)
        button_layout.add_widget(exit_button)

        container.add_widget(button_layout)

        # Create and return the popup
        return Popup(title="Success!", content=container, size_hint=(0.95, 0.4),title_align='center', title_size=self.default_font_size)

    @staticmethod
    def get_feedback_label(font_size: float) -> Widget:
        text_label =  Label(size_hint=(1, 0.8),
                            valign='middle',
                            halign='left',
                            font_size=font_size)
        text_label.bind(size=text_label.setter('text_size'))
        return text_label

    @staticmethod
    def get_ok_button() -> Widget:
        ok_button = Button(text="bundle files", size_hint=(0.2, 1))
        ok_button.background_color = (0, 1, 0, 1)  # (R, G, B, A)
        return ok_button

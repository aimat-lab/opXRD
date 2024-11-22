from __future__ import annotations

import os

from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from submission_helper.configs import get_true_width, get_true_height
from submission_helper.elements.types import FocusTextInput
from submission_helper.filesystem import FsNode, get_initial_path
from submission_helper.resources.resource_manager import get_blended_logo_path, get_kivy_image
# -------------------------------------------

class InputDialog(Popup):
    font_size = get_true_width() * 0.0145

    def __init__(self, callback: callable, **kwargs):
        super(InputDialog, self).__init__(**kwargs)
        self.size_hint = (0.65, 0.825)
        self.title = 'XRD data collector'
        self.title_align = 'center'
        self.auto_dismiss = False
        self.callback = callback
        self.background_color = (100 / 255, 1, 1, 1)  # (R, G, B, A)

        h_space = 12 * get_true_width() / 1600.
        v_space = 12 * get_true_height() / 900.
        self.content = BoxLayout(orientation='vertical',
                                 padding=[h_space, v_space, h_space, v_space],
                                 spacing=v_space)

        logo_image = get_kivy_image(width=Window.width*0.4, imgPath=get_blended_logo_path(), size_hint =(1,0.25))
        first_hint = self.make_hint()
        self.format_input =self.make_format_input()
        second_hint = self.make_second_hint()

        self.notice = self.make_notice()
        self.input_folder_hint = self.make_input_folder_hint()
        self.path_input = self.make_path_input()
        confirm_button = self.make_confirm_button()

        for widget in [logo_image,first_hint,self.format_input,second_hint,self.notice,self.input_folder_hint,self.path_input,confirm_button]:
            self.content.add_widget(widget=widget)
        self.set_focus_chain()


    def set_focus_chain(self):
        self.format_input.focus_next = self.path_input
        self.format_input.focus_previous = self.path_input

        self.path_input.focus_next = self.format_input
        self.path_input.focus_previous = self.format_input


    # -------------------------------------------
    # Callbacks

    @staticmethod
    def _update_text_height(instance, texture_size):
        instance.height = texture_size[1]  # Set the height to the text height


    def on_confirm(self, instance):
        _ = instance
        FsNode.set_default_formats(self.format_input.text)
        user_input = self.path_input.text

        print(f"User input: {user_input}")
        if not os.path.isdir(user_input):
            self.print_warning()
            self.path_input.text = ''
            return
        else:
            self.print_wait()

        Clock.schedule_once(lambda dt: self.callback(user_input),0.1)

    def print_warning(self):
        self.notice.text = f'Given input \"{self.path_input.text}\" is not a path to a directory. Please try again'
        self.notice.opacity = 1

    def print_wait(self):
        self.notice.text = f'Generating selection. Please wait ...'
        self.notice.opacity = 1
        self.notice.color = [0,1,0,1]


    # -------------------------------------------
    # make widgets

    def make_hint(self) -> Widget:
        first_hint_text = (
            'This application scans a \"data folder\" and all its subfolders for files matching specified xrd formats. '
            'Upon confirmation selected files are collected files into a .zip file. '
            'You can edit this list of formats to your preferences:')

        hint = Label(text=first_hint_text,
                          size_hint=(1, 0.2),
                          font_size=InputDialog.font_size)
        hint.bind(width=lambda *x: setattr(hint, 'text_size', (hint.width, None)))
        hint.bind(texture_size=self._update_text_height)

        return hint


    def make_second_hint(self) -> Widget:
        second_hint_text = (
            'After the search of the data folder is complete you will be presented with a selection of all files that match any of the specified formats. '
            'You can then check boxes to indicate which folders or individual files you want to share.'
            '\n\nOn press of the \"bundle files\" button, the application will collect all selected into a .zip file ready for submission on '
            'xrd.aimat.science along with a corresponding .csv file intended for specifying material properties.'
        )

        second_hint = Label(text=second_hint_text,
                                         size_hint=(1, 0.35),
                                         font_size=InputDialog.font_size)
        second_hint.bind(width=lambda *x: setattr(second_hint, 'text_size', (second_hint.width, None)))
        second_hint.bind(texture_size=self._update_text_height)

        return second_hint


    @staticmethod
    def make_format_input() -> Widget:
        default_xrd_formats = FsNode.xrd_formats
        default_xrd_text = ''
        for xrd_format in default_xrd_formats:
            default_xrd_text += f'.{xrd_format},'

        format_input = FocusTextInput(text=f'{default_xrd_text}',
                                           size_hint=(1, 0.085),
                                           font_size=InputDialog.font_size,
                                           multiline=False)
        Clock.schedule_once(lambda dt: setattr(format_input, 'focus', True), 0.1)

        return format_input


    @staticmethod
    def make_notice() -> Widget:
        return Label(opacity=0,
                     size_hint=(1, 0.075),
                     font_size = InputDialog.font_size,
                     halign="left",
                     color=[1, 0, 0, 1])



    def make_input_folder_hint(self):
        hint = Label(text='Data folder path:',
                                         size_hint=(1, 0.1),
                                         font_size=InputDialog.font_size)
        hint.bind(width=lambda *x: setattr(hint, 'text_size', (hint.width, None)))
        hint.bind(texture_size=self._update_text_height)
        return hint


    def make_path_input(self) -> Widget:
        path_input = FocusTextInput(text=f'{get_initial_path()}',
                                         size_hint=(1, 0.085),
                                         font_size=InputDialog.font_size,
                                         multiline=False)

        path_input.bind(on_text_validate=self.on_confirm)
        return path_input


    def make_confirm_button(self) -> Widget:
        confirm = Button(text='start search', size_hint=(1, 0.08), font_size=InputDialog.font_size/1.2)
        confirm.background_color = (0, 1, 0, 1)  # (R, G, B, A)
        confirm.bind(on_press=self.on_confirm)
        return confirm

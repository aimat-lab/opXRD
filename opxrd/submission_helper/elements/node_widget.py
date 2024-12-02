from __future__ import annotations

from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle
from kivy.core.image import Image as CoreImage
from kivy.properties import BooleanProperty, ObjectProperty

from submission_helper.configs import get_line_height
from submission_helper.elements.types import BlackLabel

from submission_helper.resources import get_unchecked_box_path, get_checked_box_path, get_fileicon_path, \
    get_foldericon_path, get_collapsed_icon_path, get_expanded_icon_path, get_kivy_image, get_empty_path

# -------------------------------------------

class NodeWidget(BoxLayout):
    def __init__(self, callback : callable, height : int, labeled_checkbox : LabeledCheckBox, is_file : bool):
        super().__init__(orientation='horizontal', size_hint=(1, None), height=height)

        if not is_file:
            icon_toggle_button = IconToggleButton(size_hint=(None, None),
                                          size=(height,height))

            icon_toggle_button.btn.bind(state=callback)
            self.add_widget(icon_toggle_button)
        else:
            place_holder = Widget(size_hint=(None,None),size=(height,height))
            self.add_widget(place_holder)

        self.add_widget(labeled_checkbox)


class LabeledCheckBox(BoxLayout):
    indent = get_line_height()*1.5

    def __init__(self, height : int, text :str, check_callback, is_file : bool, indent=0, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = height

        self.check_box = ImageCheckBox(size_hint=(None, None), size=(self.height, self.height))
        self.check_box.bind(active=check_callback)

        icon_path = get_fileicon_path() if is_file else get_foldericon_path()
        self.icon = Image(source=icon_path, size_hint=(None, None), size=(self.height, self.height))
        self.add_widget(self.icon)


        self.label = BlackLabel(text=text,
                                font_size=Window.width * 0.018,
                                size_hint_x=None,
                                size_hint_y=1,
                                halign='left',
                                valign='middle')

        self.add_widget(self.check_box)
        self.add_widget(self.label)
        self.padding = [indent * LabeledCheckBox.indent, 0, 0, 0]  # Left padding for indentation


class IconToggleButton(RelativeLayout):
    def __init__(self, **kwargs):
        super(IconToggleButton, self).__init__(**kwargs)

        height = self.height
        self.collapsed = get_kivy_image(width=height, imgPath=get_collapsed_icon_path(),
                                        size_hint=(None, None),
                                        size=(height,height))

        self.expanded = get_kivy_image(width=height,
                                       imgPath=get_expanded_icon_path(),
                                       size_hint=(None, None),
                                       size=(height,height))

        self.btn = ToggleButton(size_hint=(1, 1), background_color=(0, 0, 0, 0))
        self.add_widget(self.btn)

        self.icon = self.collapsed if self.btn.state == 'down' else self.expanded
        self.add_widget(self.icon)

        self.btn.bind(state=self.toggle_image)


    def toggle_image(self,instance, value):
        _ = instance
        self.remove_widget(self.icon)
        if value == 'down':
            self.icon = self.collapsed
        else:
            self.icon = self.expanded
        self.add_widget(self.icon)


class ImageCheckBox(CheckBox):
    active = BooleanProperty(False)
    checked_image = ObjectProperty(None)
    unchecked_image = ObjectProperty(None)

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.checked_image = CoreImage(get_checked_box_path()).texture
        self.unchecked_image = CoreImage(get_unchecked_box_path()).texture

        self.background_checkbox_down = get_empty_path()
        self.background_checkbox_normal = get_empty_path()

        with self.canvas:
            self.rect = Rectangle(texture=self.unchecked_image, pos=self.pos, size=self.size)

        self.bind(pos=self.update_rect, size=self.update_rect, active=self.update_rect)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.active = not self.active
            return True
        return super().on_touch_down(touch)

    def update_rect(self, *args):
        _ = args
        self.rect.pos = self.pos
        self.rect.size = self.size
        self.rect.texture = self.checked_image if self.active else self.unchecked_image


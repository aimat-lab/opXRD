from __future__ import annotations

from typing import Optional
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock


from submission_helper.filesystem.fsnode import FsNode
from submission_helper.elements.node_element import NodeElement
from submission_helper.elements.types import ThickVerticalSlider, HeaderWidget, Placeholder

from submission_helper.configs import get_line_height, get_true_height
import intervals as I

# -------------------------------------------

class SelectionLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(SelectionLayout, self).__init__(orientation='horizontal', size_hint=(1, 0.9), **kwargs)

        # attributes
        self.root_checkbox : Optional[NodeElement] = None
        self.last_load_range = I.closed(float('-inf'),float('inf'))
        self.dismiss_popup = None

        # GUI elements
        self.slider = ThickVerticalSlider(orientation='vertical', min=0, max=1, value=1, size_hint=(0.1, 1))
        self.slider.bind(value=self.do_scroll)

        self.scroll_view = self.get_scroll_view()
        self.scroll_view.bind(scroll_y=self.do_scroll)

        self.header_layout = self.get_header_widget(num_elements=0)
        self.checkboxes_layout = self.get_checkboxes_layout(file_count_label=self.header_layout,
                                                            scroll_view=self.scroll_view)

        self.add_widget(self.checkboxes_layout)
        self.add_widget(self.slider)


    def calculate_content(self, path : str):
        self.root_checkbox: NodeElement = NodeElement(path=path, height=get_line_height(), scroll_view=self.scroll_view)
        self.root_checkbox.prepare_fsys(on_done=self.set_content)


    def set_content(self, *args, **kwargs):
        _, __ = args, kwargs
        print(f'Time taken to initialize filesystem: {self.root_checkbox.time_filesystem_searching}')
        print(f'Time taken to search relevancy: {self.root_checkbox.time_relevancy_sorting}')

        self.header_layout.update_number_of_elements(num_elements=len(self.root_checkbox.get_xrd_file_des()))
        self.scroll_view.add_widget(widget=self.get_scroll_view_content(root_checkbox=self.root_checkbox))

        update_rate = 0.2
        Clock.schedule_interval(self.update_node_population, update_rate)
        Clock.schedule_interval(self.check_selection_readiness, update_rate)

    def register_content_done_callback(self, callback : callable):
        self.dismiss_popup = callback

    # -------------------------------------------
    # logic

    def update_node_population(self, *args, **kwargs):
        _, __ = args, kwargs
        total_height = self.get_total_height()
        vp_height = self.get_vp_height()

        vp_ypos = (1-self.scroll_view.scroll_y)*(total_height-vp_height)
        buffer_range = vp_height/2.
        new_load_range = I.closed(lower=vp_ypos-buffer_range,upper=vp_ypos+vp_height+buffer_range)
        unload_range_list = self.last_load_range - new_load_range

        nodes_to_load = self.root_checkbox.get_nodes_in_interval(node=self.root_checkbox, interval=new_load_range)
        nodes_to_unload = []
        for unload_range in unload_range_list:
            nodes_to_unload += self.root_checkbox.get_nodes_in_interval(node=self.root_checkbox, interval=unload_range)

        for node in nodes_to_load:
            node.load()

        for node in nodes_to_unload:
            node.unload()

        self.last_load_range = new_load_range


    def do_scroll(self, initiator, value):
        self.scroll_view.do_scroll_y = self.get_total_height() > self.get_vp_height()

        if initiator == self.slider and self.scroll_view.scroll_y != value:
            self.scroll_view.scroll_y = value
        elif initiator == self.scroll_view and self.slider.value != value:
            self.slider.value = value


    def check_selection_readiness(self, *args, **kwargs):
        _, __ = args, kwargs
        dismiss_condition = all([node.is_initialized for node in self.root_checkbox.xrd_node_des])
        if dismiss_condition and self.dismiss_popup:
            self.dismiss_popup()

    # -------------------------------------------
    # get

    def get_total_height(self) -> int:
        return self.scroll_view.children[0].height

    def get_vp_height(self) -> int:
        return self.scroll_view.height


    @staticmethod
    def get_checkboxes_layout(file_count_label: Label, scroll_view: ScrollView):
        select_layout = BoxLayout(orientation='vertical', spacing = 15 * get_true_height() / 900.)
        select_layout.add_widget(widget=file_count_label)
        select_layout.add_widget(widget=scroll_view)

        return select_layout

    @staticmethod
    def get_scroll_view():
        scroll_view = ScrollView(size_hint=(1, 1), bar_width=0)
        scrol_unit = get_true_height()/10.
        scroll_view.scroll_distance = scrol_unit
        scroll_view.scroll_wheel_distance = scrol_unit

        return scroll_view

    def get_scroll_view_content(self, root_checkbox: NodeElement):
        scroll_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        scroll_layout.bind(minimum_height=scroll_layout.setter('height'))
        scroll_layout.add_widget(Placeholder(height=get_line_height()/4.))
        self.recursively_add_boxes(gui_parent=scroll_layout, root_box=root_checkbox, indent=0)

        return scroll_layout

    def recursively_add_boxes(self, gui_parent, root_box: NodeElement, indent: int):
        if not root_box.get_is_file() and len(root_box.xrd_node_des) == 0:
            return

        def init_gui(*args, **kwargs):
            _, __ = args, kwargs
            root_box.initialize_gui(gui_parent, level=indent)
            for child_box in root_box.potential_xrd_children:
                self.recursively_add_boxes(gui_parent=root_box.child_container,
                                           root_box=child_box,
                                           indent=indent + 1)

        Clock.schedule_once(init_gui)

    @staticmethod
    def get_header_widget(num_elements: int):
        return HeaderWidget(num_elements= num_elements, format_list=FsNode.xrd_formats)

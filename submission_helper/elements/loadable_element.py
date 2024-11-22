from __future__ import annotations

from submission_helper.elements.node_widget import LabeledCheckBox
from abc import abstractmethod
from intervals import Interval

from typing import Optional
from typing import List

# -------------------------------------------

class LoadableElem:

    def __init__(self, height : int):
        super().__init__()

        self.labeled_checkbox : Optional[LabeledCheckBox] = None

        self.child_container = None
        self.root_container = None
        self.height = height
        self.child_nodes : List[LoadableElem] = []
        self.parent : Optional[LoadableElem] = None

    # -------------------------------------------
    # callbacks

    @abstractmethod
    def unload(self):
        pass


    @abstractmethod
    def load(self):
        pass

    # -------------------------------------------
    # get

    @staticmethod
    def get_nodes_in_interval(node : LoadableElem, interval : Interval) -> list:
        start_y = interval.lower
        end_y = interval.upper

        selected_nodes = []

        def dfs(current_node):
            ypos = current_node.get_ypos()
            if start_y <= ypos <= end_y:
                selected_nodes.append(current_node)

            if ypos <= end_y:
                for child in current_node.get_gui_child_nodes():
                    dfs(child)

        dfs(node)
        return selected_nodes


    def get_ypos(self) -> int:
        if self.parent is None:
            return 0

        parent_gui_children = self.parent.get_gui_child_nodes()
        child_index = parent_gui_children.index(self)
        previous_children = parent_gui_children[:child_index]

        ypos = self.parent.get_ypos() + sum([child.root_container.height for child in previous_children]) + self.parent.height
        return ypos

    def get_gui_child_nodes(self):
        return [child for child in self.child_nodes if child.root_container]


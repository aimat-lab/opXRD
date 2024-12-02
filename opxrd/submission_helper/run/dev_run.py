from submission_helper.configs import set_configs
set_configs()

from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)

from kivy.modules import inspector
from submission_helper.app import DataCollectApp
# -------------------------------------------


def main():
    win_test_dir = 'C:\\Users\\daniel\\OneDrive\\Desktop\\test_folder'
    win_target_test = 'C:\\Users\\daniel\\OneDrive\\Desktop'

    linux_input_test = '/home/daniel/OneDrive'
    linux_target_test = f'/home/daniel/aimat/tool_target_folder'


    _, __ = win_test_dir, win_target_test
    _, __ = linux_input_test, linux_target_test

    app = DataCollectApp(override_input_folder=None,
                         override_target_folder=linux_target_test)
    inspector.create_inspector(Window, app)

    app.run()

main()
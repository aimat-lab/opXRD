from submission_helper.configs import set_configs
set_configs()

from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)
from submission_helper.app import DataCollectApp
# -------------------------------------------


def main():
    app = DataCollectApp(override_input_folder=None,
                         override_target_folder=None)

    app.run()

main()
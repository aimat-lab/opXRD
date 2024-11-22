import sys
import ctypes
from kivy.config import Config
from screeninfo import get_monitors
import logging

logging.basicConfig(level=logging.INFO)
# -------------------------------------------


def get_scaling_factor():
    ui_scale = 1
    if sys.platform == 'win32':
        try:
            ui_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
        except:
            pass


    return ui_scale

def set_configs():
    Config.set('graphics', 'fullscreen', 0)
    # Config.set('graphics', 'resizable', '1')

    app_width, app_height = get_scaled_down_app_width(), get_scaled_down_app_height()

    print(f'[INFO]: Unscaled width, height: {app_width},{app_height}')

    Config.set('graphics', 'width', str(app_width))
    Config.set('graphics', 'height', str(app_height))


def get_scaled_down_app_width() -> int:
    width = int(get_true_width() / get_scaling_factor())
    return width


def get_scaled_down_app_height() -> int:
    height = int(get_true_height() / get_scaling_factor())
    return height


def get_true_width() -> int:
    primary_monitor = get_primary_monitor()
    relative_width = 0.75

    app_width = int(primary_monitor.width * relative_width)
    return app_width


def get_true_height() -> int:
    primary_monitor = get_primary_monitor()
    relative_height = 0.8

    app_height = int(primary_monitor.height * relative_height)
    return app_height


def get_line_height():
    std_width = 1080
    std_line_height = 40
    line_height =int(std_line_height * get_primary_monitor_height()/std_width)
    print(line_height)

    return line_height


def get_primary_monitor():
    monitors = get_monitors()
    for monitor in monitors:
        if monitor.is_primary:
            return monitor
    return None


def get_primary_monitor_width():
    return get_primary_monitor().width

def get_primary_monitor_height():
    return get_primary_monitor().height
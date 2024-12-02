import platform, sys, os

def get_initial_path():
    if platform.system() == 'Windows':
        initial_path = os.path.splitdrive(sys.executable)[0] + '\\'
    else:
        initial_path = '/'

    return initial_path




def make_fsys_dict(root_dir : str) -> dict:
    file_structure = {}

    for root, dirs, files in os.walk(root_dir, followlinks=True):
        relative_path = root.replace(root_dir, '').lstrip(os.sep)
        current_level = file_structure

        parts = relative_path.split(os.sep)
        for part in parts:
            if part:
                current_level = current_level.setdefault(part, {})

        for file in files:
            current_level[file] = None

    return file_structure
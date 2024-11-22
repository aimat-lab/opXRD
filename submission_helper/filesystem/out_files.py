from submission_helper.resources import get_template_csv

import os
import zipfile, csv
from typing import List


# -------------------------------------------

def zip_file_list(path_list: List[str], zipfile_path : str, root_path : str):
    zip_path_dict = get_rel_path_dict(abs_path_list=path_list, root_path=root_path)

    if os.path.isfile(zipfile_path):
        print(f'Target path {zipfile_path} already exists. Aborting ...')
        return

    with zipfile.ZipFile(zipfile_path, 'a') as zipf:
        for file_path in path_list:
            zip_path = zip_path_dict[file_path]
            zipf.write(file_path, zip_path)

    print(f"ZIP archive created at {zipfile_path}")



def produce_csv_file(abs_path_list : List[str], target_path : str, root_path : str):
    rel_path_dict = get_rel_path_dict(abs_path_list=abs_path_list, root_path=root_path)
    rel_path_list = list(rel_path_dict.values())
    template_path = get_template_csv()

    if os.path.isfile(target_path):
        print(f'Target CSV path {target_path} already exists. Aborting ...')
        return

    with open(template_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    for i, string in enumerate(rel_path_list, start=2):
        if i < len(data):
            data[i][0] = string
        else:
            data.append([string])

    with open(target_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file created at {target_path}")


def get_rel_path_dict(abs_path_list: List[str], root_path : str) -> dict:
    rel_path_dict = {}

    for path in abs_path_list:
        try:
            rel_path = os.path.relpath(path, root_path)
            rel_path_dict[path] = rel_path

        except Exception as e:
            print(f"Error processing path '{path}': {e}")

    return rel_path_dict


import os


def parse_data_cfg(path=r"D:\\pythonProject\\AI_from_bili\\YOLOv3SPP\\net_cfg\\yolov3-spp.cfg"):
    # Parses the data configuration file
    # if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
    #     path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

if __name__ == "__main__":
    print(parse_data_cfg())

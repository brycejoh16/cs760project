
import os
def make_dir(point):
    name=str(point.__name__)
    if name not in os.listdir('./data/'):
        os.system(f'mkdir ./data/{name}')

    return f"./data/{name}"

def filename_ns(input):
    return make_dir(input['point']) + '/' + f"{input}.txt"

def filename_dg(input,neighbors):
    return make_dir(input['point']) + '/' + f"neigh={neighbors},{input}.png"

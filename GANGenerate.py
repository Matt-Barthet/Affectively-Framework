import sys,os
import glob

sys.path.append(os.path.dirname(sys.path[0]))

from GA.repair import *
from GAN.dcgan import Generator
from torch.autograd import Variable

from stable_baselines3 import PPO

import keyboard
import shutil

def get_level(noise, to_string, name, size):
    width = 16

    model_to_load = name
    batch_size = 1
    image_size = 32 * size
    ngf = 64
    nz = 32
    z_dims = 10  # number different titles
    generator = Generator(nz, ngf, image_size, z_dims)
    generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))
    latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    im = levels.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    im = little_level(im[0], size)
    if to_string:
        return arr_to_str(im[0:14, 0:width])
    else:
        return im[0:14, 0:width]
    
def get_random_long_level(values):
    lvs = []

    lvs.append(get_level(values, False, './GAN/generator.pth', 1))

    lv = np.concatenate(lvs, axis=-1)
    lv = addLine(lv)
    return lv

def repair(base, destroyed_folder):
    # print(", generate 3")
    # keyboard.wait("space")

    # net_name = rootpath + "//CNet//dict.pkl"
    src_net_name = os.path.join(rootpath, "CNet", "dict.pkl")

    # lv_name = rootpath + "//LevelGenerator//GAN//Destroyed//lv0.txt"
    # result_path = rootpath + "//GA//result"

    result_path = os.path.join(base, "GA", "result")
    # result_path = rootpath + "//GA//result"
    # net_name = os.path.join(base, "CNet", "dict.pkl")
    
    net_folder = os.path.join(base, "CNet")
    os.makedirs(net_folder, exist_ok=True)

    net_name = os.path.join(net_folder, "dict.pkl")

    if not os.path.exists(net_name):
        shutil.copy2(src_net_name, net_name)
       
    
    # net_name = rootpath + "//CNet//dict.pkl"
    lv_name = os.path.join(destroyed_folder, "lv0.txt")

    # print(", generate 4")
    # keyboard.wait("space")

    score, level = GA(net_name, lv_name, result_path, isfigure=True, isrepair=True)

    # print(", generate 5")
    # keyboard.wait("space")

    return level

def generateNewLevel(values, worker_id):
    base = os.path.join(os.path.dirname(__file__), "tmp_workers", f"worker_{worker_id}")
    
    destroyed_folder = os.path.join(base, "LevelGenerator", "GAN", "Destroyed")
    if os.path.exists(destroyed_folder):
        for file in glob.glob(os.path.join(destroyed_folder, "*")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Could not delete {file}: {e}")
    else:
        os.makedirs(destroyed_folder)

    result_folder = os.path.join(base, "GA", "result")
    target_files = ["result.txt", "start.txt"]
    if os.path.exists(result_folder):
        for file in target_files:
            file_path = os.path.join(result_folder, file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Could not delete {file}: {e}")
    else:
        os.makedirs(result_folder)

    lvs = []

    lv = get_random_long_level(values)
    cnt = calculate_broken_pipes(lv)
    lvs.append((cnt, lv))

    cnt_sum = 0

    lv_path = os.path.join(destroyed_folder, f'lv0.txt')

    # print(str(worker_id) + ", lv_path: " + str(lv_path))

    with open(lv_path, 'w') as f:
        f.write(arr_to_str(lvs[0][1]))

    # print(str(worker_id) + ", generate 1")
    # keyboard.wait("space")

    cnt_sum += lvs[0][0]

    # print(str(worker_id) + ", generate 2")
    # keyboard.wait("space")

    level = repair(base, destroyed_folder)

    # print(str(worker_id) + ", level: " + str(level))
    # keyboard.wait("space")
    return level

if __name__ == '__main__':
    values = json.loads(sys.argv[1])
    print("values")
    print(str(values))

    level = generateNewLevel(values, 0)
    print("final level")
    print(str(level))
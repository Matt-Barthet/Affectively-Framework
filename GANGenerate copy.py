# import sys,os
# import glob

# sys.path.append(os.path.dirname(sys.path[0]))

# from GA.repair import *
# from LevelGenerator.GAN.dcgan import Generator
# from torch.autograd import Variable

# # def get_level(noise, to_string, name, size):
# #     width = 14

# #     model_to_load = name
# #     batch_size = 1
# #     image_size = 32 * size
# #     ngf = 64
# #     nz = 32
# #     z_dims = 10  # number different titles
# #     generator = Generator(nz, ngf, image_size, z_dims)
# #     generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))
# #     latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)
# #     with torch.no_grad():
# #         levels = generator(Variable(latent_vector))
# #     im = levels.data.cpu().numpy()
# #     im = np.argmax(im, axis=1)
# #     im = little_level(im[0], size)
# #     if to_string:
# #         return arr_to_str(im[0:14, 0:width])
# #     else:
# #         return im[0:14, 0:width]

# def get_level(noise, to_string, name, size):
#     model_to_load = name
#     batch_size = 1
#     image_size = 32 * size
#     ngf = 64
#     nz = 32
#     z_dims = 10  # number different titles
#     generator = Generator(nz, ngf, image_size, z_dims)
#     generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))
#     latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)
#     with torch.no_grad():
#         levels = generator(Variable(latent_vector))
#     im = levels.data.cpu().numpy()
#     im = np.argmax(im, axis=1)
#     im = little_level(im[0], size)
#     if to_string:
#         return arr_to_str(im[0:14, 0:28])
#     else:
#         return im[0:14, 0:28]

# def get_random_long_level(values):
#     values = np.random.randn(1, 32)
#     lvs = []
#     # print("")
#     # print("values:")
#     # print(values)
#     for i in range(int(120/28)):
#         lvs.append(get_level(values, False, './LevelGenerator/GAN/generator.pth', 1))
#     lv = np.concatenate(lvs, axis=-1)
#     lv = addLine(lv)
#     return lv

# def repair():
#     net_name = rootpath + "//CNet//dict.pkl"
#     lv_name = rootpath + "//LevelGenerator//GAN//Destroyed//lv0.txt"
#     result_path = rootpath + "//GA//result"

#     score, level = GA(net_name, lv_name, result_path, isfigure=True, isrepair=True)

#     return level

# def generateNewLevel(values):
#     destroyed_folder = os.path.join(os.path.dirname(__file__), "LevelGenerator", "GAN", "Destroyed")
#     if os.path.exists(destroyed_folder):
#         for file in glob.glob(os.path.join(destroyed_folder, "*")):
#             try:
#                 os.remove(file)
#             except Exception as e:
#                 print(f"Could not delete {file}: {e}")
#     else:
#         os.makedirs(destroyed_folder)

#     result_folder = os.path.join(os.path.dirname(__file__), "GA", "result")
#     target_files = ["result.txt", "start.txt"]
#     if os.path.exists(result_folder):
#         for file in target_files:
#             file_path = os.path.join(result_folder, file)
#             if os.path.exists(file_path):
#                 try:
#                     os.remove(file_path)
#                 except Exception as e:
#                     print(f"Could not delete {file}: {e}")
#     else:
#         os.makedirs(result_folder)

#     lvs = []
#     # print('\rgenerate',0,end='')
#     lv = get_random_long_level(values)
#     cnt = calculate_broken_pipes(lv)
#     lvs.append((cnt, lv))
#     lvs.sort(key=lambda s:s[0], reverse=True)
#     cnt_sum = 0

#     # print()
    
#     lv_path = os.path.join(destroyed_folder, f'lv0.txt')
#     with open(lv_path, 'w') as f:
#         f.write(arr_to_str(lvs[0][1]))
#     # print('lv0: cnt=', str(lvs[0][0]))
#     cnt_sum += lvs[0][0]
#     # print('avg_broken_pipe_combinations=', cnt_sum)

#     level = repair()

#     return level

# if __name__ == '__main__':
#     values = [float(x) for x in sys.argv[1:]]
#     print("values")
#     print(str(values))
#     # values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#     #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#     #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#     #         1.0, 1.0]
#     # print("new values")
#     # print(str(values))
#     level = generateNewLevel(values)
#     print("final level")
#     print(str(level))
#     # print("end test")
















































import sys,os
import glob

sys.path.append(os.path.dirname(sys.path[0]))

from GA.repair import *
from GAN.dcgan import Generator
from torch.autograd import Variable

from stable_baselines3 import PPO

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

    # values = np.array(values)
    # for latent in values:
    #     latent = latent.reshape(1, 32)
    #     lvs.append(get_level(latent, False, './LevelGenerator/GAN/generator.pth', 1))

    lvs.append(get_level(values, False, './GAN/generator.pth', 1))

    lv = np.concatenate(lvs, axis=-1)
    lv = addLine(lv)
    return lv

def repair():
    net_name = rootpath + "//CNet//dict.pkl"
    lv_name = rootpath + "//LevelGenerator//GAN//Destroyed//lv0.txt"
    result_path = rootpath + "//GA//result"

    score, level = GA(net_name, lv_name, result_path, isfigure=True, isrepair=True)

    return level

def generateNewLevel(values):
    destroyed_folder = os.path.join(os.path.dirname(__file__), "LevelGenerator", "GAN", "Destroyed")
    if os.path.exists(destroyed_folder):
        for file in glob.glob(os.path.join(destroyed_folder, "*")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Could not delete {file}: {e}")
    else:
        os.makedirs(destroyed_folder)

    result_folder = os.path.join(os.path.dirname(__file__), "GA", "result")
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
    with open(lv_path, 'w') as f:
        f.write(arr_to_str(lvs[0][1]))

    cnt_sum += lvs[0][0]

    level = repair()

    return level

if __name__ == '__main__':
    # from GANEnv import GANLevelEnv
    # env = GANLevelEnv()
    # model = PPO.load("Agents/PPO/cnn_ppo_solid_optimize_1_extended.zip", env=env)
    # obs, info = env.reset()

    # action, _ = model.predict(obs, deterministic=True)

    # obs, reward, terminated, truncated, info = env.step(action)

    
    # print("Generated Level (obs):")
    # print(obs)
    # print("Reward:", reward)

    values = json.loads(sys.argv[1])
    print("values")
    print(str(values))
    # values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0]
    # print("new values")
    # print(str(values))
    level = generateNewLevel(values)
    print("final level")
    print(str(level))
    # print("end test")

































# import sys,os
# import glob

# sys.path.append(os.path.dirname(sys.path[0]))

# from GA.repair import *
# from LevelGenerator.GAN.dcgan import Generator
# from torch.autograd import Variable

# def get_level(noise, to_string, name, size):
#     model_to_load = name
#     batch_size = 1
#     image_size = 32 * size
#     ngf = 64
#     nz = 32
#     z_dims = 10  # number different titles
#     generator = Generator(nz, ngf, image_size, z_dims)
#     generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))
#     latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)
#     with torch.no_grad():
#         levels = generator(Variable(latent_vector))
#     im = levels.data.cpu().numpy()
#     im = np.argmax(im, axis=1)
#     im = little_level(im[0], size)
#     if to_string:
#         return arr_to_str(im[0:14, 0:28])
#     else:
#         return im[0:14, 0:28]
# def get_random_long_level():
#     lvs = []
#     for i in range(int(120/28)):
#         lvs.append(get_level(np.random.randn(1, 32), False, './LevelGenerator/GAN/generator.pth', 1))
#     lv = np.concatenate(lvs, axis=-1)
#     lv = addLine(lv)
#     return lv

# def repair():
#     net_name = rootpath + "//CNet//dict.pkl"
#     lv_name = rootpath + "//LevelGenerator//GAN//Destroyed//lv" + str(random.randint(0, 4)) + ".txt"
#     result_path = rootpath + "//GA//result"

#     GA(net_name, lv_name, result_path, isfigure=True, isrepair=True)

# def generateNewLevel():
#     destroyed_folder = os.path.join(os.path.dirname(__file__), "LevelGenerator", "GAN", "Destroyed")
#     if os.path.exists(destroyed_folder):
#         for file in glob.glob(os.path.join(destroyed_folder, "*")):
#             try:
#                 os.remove(file)
#             except Exception as e:
#                 print(f"Could not delete {file}: {e}")
#     else:
#         os.makedirs(destroyed_folder)

#     lvs = []
#     total = 100
#     select = 5
#     for i in range(total):
#         print('\rgenerate',i,end='')
#         lv = get_random_long_level()
#         cnt = calculate_broken_pipes(lv)
#         lvs.append((cnt, lv))
#     lvs.sort(key=lambda s:s[0], reverse=True)
#     cnt_sum = 0
#     print()
#     for i in range(select):
#         lv_path = os.path.join(destroyed_folder, f'lv{i}.txt')
#         with open(lv_path, 'w') as f:
#             f.write(arr_to_str(lvs[i][1]))
#         print('lv'+str(i)+': cnt=', str(lvs[i][0]))
#         cnt_sum += lvs[i][0]
#     print('avg_broken_pipe_combinations=', cnt_sum / total)

#     repair()

# if __name__ == '__main__':
#     generateNewLevel()
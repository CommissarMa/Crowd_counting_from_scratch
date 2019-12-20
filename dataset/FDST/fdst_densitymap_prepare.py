#%%
'''
Modify test_data/65/001.json first, or will raise error.
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm


def generate_densitymap(image,points,sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for Fudan-ShanghaiTech. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [x,y,width,height]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        # width = int(round(point[2]))
        # height = int(round(point[3]))
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
        # densitymap += gaussian_filter(point2density, sigma=(width,height), mode='constant')
        # densitymap += gaussian_filter(point2density, sigma=15, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap    

if __name__ == '__main__':
    phase_list = ['train','test']
    for phase in phase_list:
        temp_root = phase + '_data'
        print('Now process -', temp_root)
        for scene in os.listdir(temp_root):
            temp_path = os.path.join(temp_root,scene)
            '''process each image'''
            for name in tqdm(os.listdir(temp_path)):
                if (name[0] not in ['0','1']) or name.split('.')[-1] != 'jpg':
                    continue
                img_path = os.path.join(temp_path,name)
                json_path = img_path.replace('jpg','json')
                npy_path = img_path.replace('jpg','npy')
                img = plt.imread(img_path)
                # plt.imshow(img)
                with open(json_path,'r') as load_f:
                    anno = json.load(load_f)
                    keyname = list(anno.keys())[0]
                    anno = anno[keyname]['regions']
                    points = []
                    for head in anno:
                        head = head['shape_attributes']
                        x = head['x']
                        y = head['y']
                        width = head['width']
                        height = head['height']
                        points.append((x,y,width,height))
                    '''generate density map'''
                    densitymap = generate_densitymap(img,points)
                    np.save(npy_path,densitymap)
                    # plt.figure()
                    # plt.imshow(densitymap)

        #         break
        #     break
        # break

# %%

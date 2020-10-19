import torch
import torch.utils.data as data
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps, ImageChops
import random
import torchvision.transforms as transforms
import cv2
import numpy as np

def ImgOfffSet(Img,xoff,yoff):
    width, height = Img.size
    c = ImageChops.offset(Img,xoff,yoff)
    c.paste((0,0,0),(0,0,xoff,height))
    c.paste((0,0,0),(0,0,width,yoff))
    return c

def read_image(file_list,crop_size,input_height,scale,random_seed,is_mirror=True,option=False):

    #image= cv2.imread(file_list)
    image = Image.open(file_list)
    image = image.convert('RGB')
    if is_mirror and random_seed is 0:
        # print('mirror')
        image = ImageOps.mirror(image)
    # else:
    #     print('not mirror')
    a1 = 178 / 2 - crop_size / 2
    a2 = 178 / 2 + crop_size / 2
    b1 = 218 / 2 - crop_size / 2
    b2 = 218 / 2 + crop_size / 2
    ########
    #print('this is image',image)
    center_crop = ImageOps.crop(image, (a1, b1, a1, b1))


    #print('this is center crop',center_crop)
    #center_crop = image[int(b1):int(b2), int(a1):int(a2), :]

    hr = center_crop.resize((input_height, input_height),Image.BICUBIC)
    lr = center_crop.resize((int(input_height / scale), int(input_height / scale)),Image.BICUBIC)

##offset the pixels
    # if option==True:
    #
    #     lr=ImgOfffSet(lr,20,0)
###############


        #
        # lr.save('shift_10.png')
        # print('*********this is render ')

    #hr = cv2.resize(center_crop, dsize=(input_height, input_height), interpolation=cv2.INTER_LINEAR)
    #lr = cv2.resize(hr, dsize=(int(input_height / scale), int(input_height / scale)), interpolation=cv2.INTER_LINEAR)

    return hr,lr
def load_data(file_path,render_file,crop_size,input_height,scale,is_mirror):
    random_seed=random.randint(0, 1)
    hr,lr=read_image(file_path,crop_size,input_height,scale,random_seed,is_mirror,option=False)
    _, rend_lr=read_image(render_file, crop_size, input_height,scale,random_seed,is_mirror, option=True)

    #input_data=np.concatenate((lr,rend_lr),axis=2)

    #print(input_data.shape)
    # hr = torch.from_numpy(hr)
    # lr = torch.from_numpy(lr)
    # hr = hr.transpose(2, 0, 1)
    # lr = lr.transpose(2, 0, 1)
    return hr, lr, rend_lr

def load_file_list(path,render=False):
    ab_path=list()
    if render==True:
        data_path = os.path.join(path, 'render')
    else:
        data_path = os.path.join(path, 'data')
    dirs =os.listdir(data_path)
    dirs.sort()
    for dir_name in dirs[0:int(len(dirs))]:
        ab_path.append(os.path.join(data_path, dir_name))
    return ab_path

class MyTupleDataset(data.Dataset):
    def __init__(self,image_list,render_list,crop_size, input_height,up_scale,is_mirror):
        super(MyTupleDataset, self).__init__()
    # init your dataset here...
        self.image_list=image_list ###
        self.render_list = render_list  ###
        self.crop_size=crop_size
        self.input_height=input_height
        self.up_scale = up_scale
        self.is_mirror = is_mirror
        self.input_transform = transforms.Compose([
                                   transforms.ToTensor()
                               ])

    def __getitem__(self,index):
        ##print(index)
        hr,lr,rend_lr=load_data(self.image_list[index],self.render_list[index],self.crop_size,self.input_height,self.up_scale,self.is_mirror)

        input = self.input_transform(lr)
        target = self.input_transform(hr)
        rend_lr =self.input_transform(rend_lr)
        #print('input and render_lr and target shape',input.shape,rend_lr.shape,target.shape)
        return input, rend_lr, target

    def __len__(self):
        return len(self.image_list)

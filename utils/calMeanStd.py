import os
import numpy as np
import scipy.misc as m
from loader import get_loader
from os.path import join as pjoin

m1_mean = []
m2_mean = []
m3_mean = []
m1_std = []
m2_std = []
m3_std = []


data_loader = get_loader("NeoBrainS12")
data_path = "/home/jwliu/disk/kxie/CNN_LSTM/dataset/neobrains12/"

t_loader = data_loader(
    data_path,
    type="all",
    split="train",
)

for (images, labels, img_name) in t_loader:

    # images = images.transpose(1,2,0)
    # m.imsave(pjoin('/home/jwliu/disk/kxie/CNN_LSTM/result_image_when_training/forfun', '{}.bmp'.format(img_name)), images)
    # print(".")

    mean1 = np.mean(images[:,:,0])
    mean2 = np.mean(images[:,:,1])
    # mean3 = np.mean(images[2,:,:])

    std1 = np.std(images[:,:,0])
    std2 = np.std(images[:,:,1])
    # std3 = np.std(images[2,:,:])

    m1_mean.append(mean1)
    m2_mean.append(mean2)
    # m3_mean.append(mean3)
    m1_std.append(std1)
    m2_std.append(std2)
    # m3_std.append(std3)


print("mean:[{}, {}]".format(np.mean(m1_mean)/255.0, np.mean(m2_mean)/255.0))
print("std:[{}, {}]".format(np.mean(m1_std)/255.0, np.mean(m2_std)/255.0))

# brainweb
# mean:[0.19587023896602954, 0.17886593808488374, 0.3225062481266075]
# std:[0.25694185835052424, 0.25695371019867097, 0.4008627305422981]


# neobrains12 all


print("cal done. :)")
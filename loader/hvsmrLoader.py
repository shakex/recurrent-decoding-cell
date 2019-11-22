import collections
from os.path import join as pjoin
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import scipy.misc as m


class hvsmrLoader(data.Dataset):
    """
    docstring for hvsmrLoader
    segmentation of the blood pool and ventricular myocaedium
    """

    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        self.n_classes = 3
        self.files = collections.defaultdict(list)

        for split in ["train", "val", "trainval"]:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        # img_path = self.root + 'imgs/rgb/' + img_name + '.bmp'
        img_path = self.root + 'imgs/' + img_name + '.bmp'
        lbl_path = self.root + 'gt/' + img_name + '.bmp'

        # np:(h,w,c)
        img = m.imread(img_path)
        lbl = m.imread(lbl_path)

        img, lbl = self.transform(img, lbl)

        return img, lbl, img_name


    def transform(self, img, lbl):
        img = img.transpose((2,0,1))
        img = img.astype(float) / 255.0
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def get_brainweb_colormap(self):
        """
        0:bg
        1:blood pool
        2:myocardium

        """

        return np.asarray([[0, 0, 0], [127, 127, 127], [255, 255, 255]])


    def encode_segmap(self, mask):
        mask = mask.astype(np.uint8)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(self.get_brainweb_colormap()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(np.uint8)
        return label_mask


    def decode_segmap(self, label_mask, plot=False):

        label_colors = self.get_brainweb_colormap()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colors[ll, 0]
            g[label_mask == ll] = label_colors[ll, 1]
            b[label_mask == ll] = label_colors[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb.astype(np.uint8)


def debug_load():
    root = '{TODO: root_path}'

    t_loader = hvsmrLoader(
        root,
        split='trainval')

    trainLoader = data.DataLoader(t_loader,
                                  batch_size=1,
                                  num_workers=4,
                                  shuffle=True)

    for (images, labels, img_name) in trainLoader:
        labels = np.squeeze(labels.data.numpy())
        decoded = t_loader.decode_segmap(labels, plot=False)
        m.imsave(pjoin('{TODO: save_path}', '{}.bmp'.format(img_name[0])), decoded)


if __name__ == '__main__':
    debug_load()


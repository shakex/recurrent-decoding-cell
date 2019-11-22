import yaml
import torch
import argparse
import numpy as np
# import seaborn as sns
import scipy.misc as m
from torch.utils import data
from models import get_model
from loader import get_loader
from utils import convert_state_dict
from metrics import runningScore
from tqdm import tqdm
from os.path import join as pjoin
import matplotlib.pyplot as plt
import pandas as pd

def test(cfg):

    device = torch.device("cuda:{}".format(cfg["training"]["gpu_idx"]) if torch.cuda.is_available() else "cpu")

    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    v_loader = data_loader(
        data_path,
        split='val')

    n_classes = v_loader.n_classes
    n_val = len(v_loader.files['val'])
    valLoader = data.DataLoader(
        v_loader,
        batch_size=1,
        num_workers=cfg["training"]["n_workers"]
    )

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(cfg["testing"]["trained_model"], map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    running_metrics_val = runningScore(n_classes, n_val)
    with torch.no_grad():
        for i_val, (images_val, labels_val, img_name_val) in tqdm(enumerate(valLoader)):
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            outputs = model(images_val)

            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy())
            gt = np.squeeze(labels_val.data.cpu().numpy())

            running_metrics_val.update(gt, pred, i_val)

            decoded = v_loader.decode_segmap(pred, plot=False)
            m.imsave(pjoin(cfg["testing"]["path"], '{}.bmp'.format(img_name_val[0])),decoded)

    score = running_metrics_val.get_scores()
    acc_all, dsc_cls = running_metrics_val.get_list()
    for k, v in score[0].items():
        print(k, v)

    if cfg["testing"]["boxplot"]==True:
        sns.set_style("whitegrid")
        labels = ['CSF', 'Gray Matter', 'White Matter']
        fig1, ax1 = plt.subplots()
        ax1.set_title('Basic Plot')
        # ax1.boxplot(dsc_cls.transpose()[:,1:n_classes], showfliers=False, labels=labels)
        ax1 = sns.boxplot(data=dsc_cls.transpose()[:,1:n_classes])

        # ax1.yaxis.grid(True)
        ax1.set_xlabel('Three separate samples')
        ax1.set_ylabel('Dice Score')

        # path to save boxplot
        plt.savefig('/home/jwliu/disk/kxie/CNN_LSTM/test_results/box.pdf')




def boxplotvis(cfg):

    # device = torch.device("cuda:{}".format(cfg["other"]["gpu_idx"]) if torch.cuda.is_available() else "cpu")
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    v_loader = data_loader(
        data_path,
        split='val'
    )

    n_classes = v_loader.n_classes
    n_val = len(v_loader.files['val'])

    # test differnet models' prediction
    vgg16lstm_metric = runningScore(n_classes, n_val)
    vgg16gru_metric = runningScore(n_classes, n_val)
    segnet_metric = runningScore(n_classes, n_val)

    with torch.no_grad():
        for i_val, (images_val, labels_val, img_name_val) in tqdm(enumerate(v_loader)):
            gt = np.squeeze(labels_val.data.cpu().numpy())
            vgg16lstm_pred = m.imread(pjoin(cfg["data"]["pred_path"], 'vgg16_lstm_brainweb', img_name_val+'.bmp'))
            vgg16gru_pred = m.imread(pjoin(cfg["data"]["pred_path"], 'vgg16_gru_brainweb', img_name_val + '.bmp'))
            segnet_pred = m.imread(pjoin(cfg["data"]["pred_path"], 'segnet_brainweb', img_name_val + '.bmp'))

            vgg16lstm_encode = v_loader.encode_segmap(vgg16lstm_pred)
            vgg16gru_encode = v_loader.encode_segmap(vgg16gru_pred)
            segnet_encode = v_loader.encode_segmap(segnet_pred)

            vgg16lstm_metric.update(gt, vgg16lstm_encode, i_val)
            vgg16gru_metric.update(gt, vgg16gru_encode, i_val)
            segnet_metric.update(gt, segnet_encode, i_val)

    vgg16lstm_acc_all, vgg16lstm_dsc_cls = vgg16lstm_metric.get_list()
    vgg16gru_acc_all, vgg16gru_dsc_cls = vgg16gru_metric.get_list()
    segnet_acc_all, segnet_dsc_cls = segnet_metric.get_list()
    # dsc_list = [vgg16lstm_dsc_cls.transpose(), vgg16gru_dsc_cls.transpose(), segnet_dsc_cls.transpose()]

    data0 = array2dataframe(vgg16lstm_dsc_cls)
    data0['Method'] = 'VGG16-LSTM'
    data1 = array2dataframe(vgg16gru_dsc_cls)
    data1['Method'] = 'VGG16-GRU'
    data2 = array2dataframe(segnet_dsc_cls)
    data2['Method'] = 'SegNet'
    data = pd.concat([data0, data1, data2])

    # fig, ax = plt.subplots(figsize=(3, 5))
    #
    # sns.set(context='paper', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True)
    # ax = sns.boxplot(x="classType", y="Dice score", hue="Method", data=data,
    #                 showfliers=False,
    #                 linewidth=1,
    #                 saturation=1,
    #                 width=0.5)
    # ax.yaxis.grid(True)
    # # plt.legend(loc='lower left')
    #
    # plt.savefig(pjoin(cfg["data"]["save_path"], 'boxplot1.eps'))



    # style2
    # bg_array = np.stack((vgg16lstm_dsc_cls[0,:].transpose(),
    #                      vgg16gru_dsc_cls[0,:].transpose(),
    #                      segnet_dsc_cls[0,:].transpose()),
    #                      axis=1)
    # csf_array = np.stack((vgg16lstm_dsc_cls[1,:].transpose(),
    #                       vgg16gru_dsc_cls[1,:].transpose(),
    #                       segnet_dsc_cls[1,:].transpose()),
    #                       axis=1)
    # gm_array = np.stack((vgg16lstm_dsc_cls[2,:].transpose(),
    #                      vgg16gru_dsc_cls[2,:].transpose(),
    #                      segnet_dsc_cls[2,:].transpose()),
    #                      axis=1)
    # wm_array = np.stack((vgg16lstm_dsc_cls[3,:].transpose(),
    #                      vgg16gru_dsc_cls[3,:].transpose(),
    #                      segnet_dsc_cls[3,:].transpose()),
    #                      axis=1)
    #
    # fig, axes = plt.subplots(2,2, figsize=(8, 6))
    # labels = ['VGG16-convLSTM', 'VGG16-convGRU', 'SegNet']
    #
    #
    # plt.subplot(221)
    # plt.title('CSF')
    # plt.boxplot(csf_array,
    #             # showfliers=False,
    #             patch_artist=False,
    #             labels=labels)
    # plt.xticks(rotation=30)
    #
    # plt.subplot(222)
    # plt.title('GM')
    # plt.boxplot(gm_array,
    #             # showfliers=False,
    #             patch_artist=False,
    #             labels=labels)
    # plt.xticks(rotation=30)
    #
    # plt.subplot(223)
    # plt.title('WM')
    # plt.boxplot(wm_array,
    #             # showfliers=False,
    #             patch_artist=False,
    #             labels=labels)
    # plt.xticks(rotation=30)
    #
    # # for ax in axes:
    # #     # ax.yaxis.grid(True)
    # #     ax.set_xlabel('Segmentation Methods')
    # #     ax.set_ylabel('Dice Score')
    #
    # plt.savefig(pjoin(cfg["data"]["save_path"], 'boxplot2.eps'))



def array2dataframe(dsc_cls):
    columns = ['Dice score']
    data00 = pd.DataFrame(data=dsc_cls[0, :].transpose(), columns=columns)
    data00['classType'] = 'BG'
    data01 = pd.DataFrame(data=dsc_cls[1, :].transpose(), columns=columns)
    data01['classType'] = 'CSF'
    data02 = pd.DataFrame(data=dsc_cls[2, :].transpose(), columns=columns)
    data02['classType'] = 'GM'
    data03 = pd.DataFrame(data=dsc_cls[3, :].transpose(), columns=columns)
    data03['classType'] = 'WM'
    data0 = pd.concat([data01, data02, data03])
    return data0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/tinyrnn_hvsmr.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    test(cfg)
    # boxplotvis(cfg)


import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
# from model.unet_self_cross import UNet
# from model.unet import UNet
# from model.unet_multiscale import UNet
from model.vision_transformer import SwinUnet as ViT_seg
from model.config import get_config
import yaml
from model.unet import UNet
# from model.best.unet_corr_s import UNet
# from model.unet_self_cross import UNet
# from model.unet_yuan import UNet
from model.unet_multi_self_cross import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='D:/data/ACDC/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--config', default="D:/UniMatch-main/more-scenarios/medical/configs/acdc_swinunet.yaml")
parser.add_argument(
    '--cfg', type=str, default="D:/UniMatch-main/more-scenarios/medical/configs/swin_tiny_patch4_window7_224_lite.yaml",
    help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--patch_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--labeled-id-path', default="D:/UniMatch-main/more-scenarios/medical/splits/acdc/7/labeled.txt")
parser.add_argument('--unlabeled-id-path',
                    default="D:/UniMatch-main/more-scenarios/medical/splits/acdc/7/unlabeled.txt")

args = parser.parse_args()
config = get_config(args)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() == 0 and gt.sum() == 0:
        return dice, 0
    if pred.sum() == 0 and gt.sum() !=0:
        return dice, 50
    hd = metric.binary.hd95(pred, gt)
    return dice, hd


def test_single_volume(case, net,  test_save_path, FLAGS):
    # from torch.utils.data import DataLoader
    # from dataset.acdc import ACDCDataset
    # import torch.nn.functional as F
    #
    # trainset_u = ACDCDataset('acdc', 'D:/data/ACDC/', 'train_u',
    #                          224, args.unlabeled_id_path)
    # trainset_l = ACDCDataset('acdc', 'D:/data/ACDC/', 'train_l',
    #                          224, args.labeled_id_path, nsample=len(trainset_u.ids))
    #
    # trainloader_l = DataLoader(trainset_l, batch_size=1, shuffle=False,
    #                            num_workers=0, drop_last=False)
    # trainloader_u = DataLoader(trainset_u, batch_size=1, shuffle=True,
    #                            num_workers=0, drop_last=False)
    # net.eval()
    # dice_class = [0] * 3
    #
    # with torch.no_grad():
    #     for img, mask in trainloader_l:
    #         img, mask = img.cuda(), mask.cuda()
    #
    #         h, w = img.shape[-2:]
    #         img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
    #
    #         img = img.permute(1, 0, 2, 3)
    #
    #         pred = net(img)
    #
    #         pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
    #         pred = pred.argmax(dim=1).unsqueeze(0)
    #
    #         for cls in range(1, 4):
    #             inter = ((pred == cls) * (mask == cls)).sum().item()
    #             union = (pred == cls).sum().item() + (mask == cls).sum().item()
    #             dice_class[cls - 1] += 2.0 * inter / (union + 1e-7)
    #
    # dice_class = [dice * 100.0 / len(trainloader_l) for dice in dice_class]
    # mean_dice = sum(dice_class) / len(dice_class)
    # print(dice_class)

    h5f = h5py.File(FLAGS.root_path + "/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        # net_2.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
                # out_main_1 = net_2(input)

            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    data_path = "D:/UniMatch-main/more-scenarios/medical/splits/acdc/test.txt"
    # with open(FLAGS.root_path + '/test.txt', 'r') as f:
    with open(data_path, 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    # snapshot_path = "../model/{}_{}_labeled/{}".format(
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    # test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
    test_save_path = "../model/{}_{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    # net = net_factory(net_type=FLAGS.model, in_chns=1,
    #                   class_num=FLAGS.num_classes)
    net = UNet(1, 4).cuda()
    print(net.state_dict().keys())
    net.eval()

    # net_2 = ViT_seg(config, img_size=args.patch_size,
    #                      num_classes=cfg['nclass']).cuda()
    # save_mode_path_1 = "D:/UniMatch-main/more-scenarios/medical/save_swin_cnn_1/best_swin.pth"
    # net_2.load_state_dict(torch.load(save_mode_path_1)['model'])
    # net_2.eval()

    # save_mode_path = os.path.join(
    #     snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    # save_mode_path = "D:/UniMatch-main/more-scenarios/medical/save_shuffle_7_onlysup/best.pth"
    # save_mode_path = "D:/UniMatch-main/visualize/unimatch/best (39).pth"
    # save_mode_path = "D:/UniMatch-main/more-scenarios/medical/save_87_multi_max_3_onlysup/best (64).pth"
    # save_mode_path = "D:/UniMatch-main/more-scenarios/medical/save_7_224/best.pth"
    save_mode_path = "D:/UniMatch-main/download/best (72).pth"
    weight_dict = torch.load(save_mode_path)
    print(weight_dict.keys())
    # net.load_state_dict(torch.load(save_mode_path)['model'])
    net.load_state_dict(torch.load(save_mode_path)['model'])
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)

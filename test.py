from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
from models.main import *
import argparse
import numpy as np
import cv2
from skimage import img_as_ubyte

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # Init model
    device = torch.device("cuda")
    model_a_lot = os.listdir(args.test_model_dir_a_lot)
    for module_num in range(len(model_a_lot)):
        model = CDNet()
        model = model.to(device)
        try:
            cci_net_dict = torch.load(os.path.join(args.test_model_dir_a_lot, model_a_lot[module_num]))
            print('loaded', model_a_lot[module_num])
        except:
            cci_net_dict = torch.load(os.path.join(args.param_root, 'cdnet.pth'))
        
        model.to(device)
        model.cci_net.load_state_dict(cci_net_dict)
        model.eval()
        model.set_mode('test')

        module_name = model_a_lot[module_num].split('/')[-1]
        module_name = module_name.split('.pth')[0]

        tensor2pil = transforms.ToPILImage()
        for testset in ['CoCA','CoSOD3k','CoSal2015']:
            if testset == 'CoCA':
                test_img_path = '/home/yetcandoit/projects/CDNet/data/images/CoCA/'
                test_gt_path = '/home/yetcandoit/projects/CDNet/data/gts/CoCA/'
                saved_root = os.path.join(args.save_test_path_root_a_lot, module_name, 'CoCA')
            elif testset == 'CoSOD3k':
                test_img_path = '/home/yetcandoit/projects/CDNet/data/images/CoSOD3k/'
                test_gt_path = '/home/yetcandoit/projects/CDNet/data/gts/CoSOD3k/'
                saved_root = os.path.join(args.save_test_path_root_a_lot, module_name, 'CoSOD3k')
            elif testset == 'CoSal2015':
                test_img_path = '/home/yetcandoit/projects/CDNet/data/images/CoSal2015/'
                test_gt_path = '/home/yetcandoit/projects/CDNet/data/gts/CoSal2015/'
                saved_root = os.path.join(args.save_test_path_root_a_lot, module_name, 'CoSal2015')
            else:
                print('Unkonwn test dataset')
                print(args.dataset)
            
            test_loader = get_loader(
                test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

            for batch in tqdm(test_loader):
                inputs = batch[0].to(device).squeeze(0)
                gts = batch[1].to(device).squeeze(0)
                subpaths = batch[2]
                ori_sizes = batch[3]
                scaled_preds= model(inputs, gts)
                scaled_preds = torch.sigmoid(scaled_preds[-1])
                os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)
                num = gts.shape[0]
                for inum in range(num):
                    subpath = subpaths[inum][0]
                    ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
                    save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--test_model_dir_a_lot', default='./Checkpoint/Best_epochs', type=str, help='model folder')
    parser.add_argument('--save_test_path_root_a_lot', default='./eval/pred2', type=str, help='多测试结果')

    args = parser.parse_args()

    main(args)
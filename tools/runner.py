import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *

import cv2
import numpy as np
import traceback


def test_net(args, config):
    try:
        logger = get_logger(args.log_name)
        print_log('Tester start ... ', logger = logger)
        _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

        base_model = builder.model_builder(config.model)
        # base_model.load_model_from_ckpt(args.ckpts)
        builder.load_model(base_model, args.ckpts, logger = logger)

        if args.use_gpu:
            base_model.to(args.local_rank)

        #  DDP
        if args.distributed:
            raise NotImplementedError()

        test(base_model, test_dataloader, args, config, logger=logger)
    except Exception as e:
        # If an exception occurs, it will be caught here
        print(f"An exception occurred: {str(e)}") 
        traceback.print_exc()


# visualization
def test(base_model, test_dataloader, args, config, logger = None):
    #print('got here1')
    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156", #plane
        "04379243",  #table
        "03790512", #motorbike
        "03948459", #pistol
        "03642806", #laptop
        "03467517",     #guitar
        "03261776", #earphone
        "03001627", #chair
        "02958343", #car
        "04090263", #rifle
        "03759954", # microphone
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            #print('idx is', idx, taxonomy_ids, model_ids, data)
            # import pdb; pdb.set_trace()
            #if  taxonomy_ids[0] not in useful_cate:
                #continue
            if taxonomy_ids[0] == "02691156":
                a, b= 90, 135
            elif taxonomy_ids[0] == "04379243":
                a, b = 30, 30
            elif taxonomy_ids[0] == "03642806":
                a, b = 30, -45
            elif taxonomy_ids[0] == "03467517":
                a, b = 0, 90
            elif taxonomy_ids[0] == "03261776":
                a, b = 0, 75
            elif taxonomy_ids[0] == "03001627":
                a, b = 30, -45
            elif taxonomy_ids[0] == "04401088":
                a, b = 30, 30
            else:
                a, b = 0, 0
            #print(a,b)
            #print('got here2')
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'edepsim':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            #print('got here3')
            # dense_points, vis_points = base_model(points, vis=True)
            dense_points, vis_points, centers= base_model(points, vis=True)
            final_image = []
            data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points,a,b)
            #print('points!!!!!!!!!!!!!!!!!!!!', points)
            #print('points!!!!!!!!!!!!!!!!!!!!2', points[150:650,150:675,:])
            final_image.append(points[150:650,150:675,:])

            # centers = centers.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            # centers = misc.get_ptcloud_img(centers)
            # final_image.append(centers)

            vis_points = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_ptcloud_img(vis_points,a,b)
            #print('vis_points!!!!!!!!!!!!!!!!!!!!',vis_points)
            #print('vis_points!!!!!!!!!!!!!!!!!!!!2',vis_points[150:650,150:675,:])
            final_image.append(vis_points[150:650,150:675,:])

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points,a,b)
            #print('dense_points!!!!!!!!!!!!!!!!!!!!',dense_points)
            #print('dense_points!!!!!!!!!!!!!!!!!!!!2',dense_points[150:650,150:675,:])
            final_image.append(dense_points[150:650,150:675,:])

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)

            if idx > 1500:
                break

        return

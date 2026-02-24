#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from glob import glob
from torch.autograd import Variable
from models.vmamba_Fusion_efficross import VSSM_Fusion
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
 
from loss import Fusionloss
 
import torch
from torch.utils.data import DataLoader
import warnings

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
 
warnings.filterwarnings('ignore')
 
def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()
 
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).contiguous().view(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.view(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
 
def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.view(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
 
 
def train_fusion(num=0, logger=None):
    lr_start = 0.0002
    modelpth = 'model_last'
    Method = 'my_cross'
    modelpth = os.path.join(modelpth, Method)
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=f'./runs/{Method}_exp{num}')
    logger.info(f'TensorBoard logs will be saved to: ./runs/{Method}_exp{num}')
    
    fusionmodel = eval('VSSM_Fusion')()

    # ===== 加载预训练模型 =====
    # pretrained_model_path = 'KAIST_1ssim_10int_1grad_.pth'
    # if os.path.exists(pretrained_model_path):
    #     fusionmodel.load_state_dict(torch.load(pretrained_model_path))
    #     logger.info(f"Successfully loaded pretrained model from: {pretrained_model_path}")
    # else:
    #     logger.info(f"Pretrained model not found at {pretrained_model_path}, training from scratch.")

    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('train', length=30000)
    print("the training dataset is length:{}".format(train_dataset.length))
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss()
 
    epoch = 2
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')

    # 确保保存模型的目录存在
    os.makedirs(modelpth, exist_ok=True)
    
    for epo in range(0, epoch):
        lr_start = 0.0001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        
        for it, (image_vis, image_ir) in enumerate(train_loader):
            try:
                fusionmodel.train()
                image_vis = Variable(image_vis).cuda()
                image_ir = Variable(image_ir).cuda()
                fusion_image = fusionmodel(image_vis, image_ir)
 
            except TypeError as e:
                print(f"Caught TypeError: {e}")
                continue
 
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            optimizer.zero_grad()
 
            # fusion loss
            loss_fusion, loss_in, ssim_loss, loss_grad = criteria_fusion(
                image_vis=image_vis, image_ir=image_ir, generate_img=fusion_image, 
                i=num, labels=None
            )
 
            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'ssim_loss: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=ssim_loss.item(),
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                
                writer.add_scalar('Loss/Total', loss_total.item(), now_it)
                writer.add_scalar('Loss/Intensity', loss_in.item(), now_it)
                writer.add_scalar('Loss/Gradient', loss_grad.item(), now_it)
                writer.add_scalar('Loss/SSIM', ssim_loss.item(), now_it)
                writer.add_scalar('Hyperparameters/Learning_Rate', lr_this_epo, now_it)
                
                st = ed
            
            # ===== 新增：每100步保存一次图像可视化 =====
            if now_it % 100 == 0:
                with torch.no_grad():
                    # 取batch中的第一张图做可视化
                    vis_ir = image_ir[0:1].cpu()
                    vis_vis = image_vis[0:1].cpu()
                    vis_fusion = fusion_image[0:1].cpu()
                    
                    # 方案1：横向拼接 [IR | Visible | Fusion]
                    img_concat = torch.cat([vis_ir, vis_vis, vis_fusion], dim=3)
                    writer.add_image('Training/IR_VIS_Fusion_Concat', 
                                   vutils.make_grid(img_concat, normalize=True, scale_each=True),
                                   now_it)
                    
                    # 方案2：网格展示（更清晰）
                    img_grid = torch.cat([vis_ir, vis_vis, vis_fusion], dim=0)
                    writer.add_images('Training/Separate_Views', 
                                     img_grid, 
                                     now_it, 
                                     dataformats='NCHW')
            if now_it % 2000 == 0:
                step_model_file = os.path.join(modelpth, f'fusion_model_step_{now_it}.pth')
                torch.save(fusionmodel.state_dict(), step_model_file)
                logger.info(f"====> Checkpoint saved at step {now_it}: {step_model_file}")


        logger.info(f'Epoch {epo} finished, generating validation visualizations...')
        fusionmodel.eval()
        with torch.no_grad():
            # 从训练集中取几张图做epoch总结可视化
            val_iter = iter(train_loader)
            val_vis, val_ir = next(val_iter)
            val_vis = val_vis.cuda()
            val_ir = val_ir.cuda()
            val_fusion = fusionmodel(val_vis, val_ir)
            
            # 夹值到[0,1]
            ones = torch.ones_like(val_fusion)
            zeros = torch.zeros_like(val_fusion)
            val_fusion = torch.where(val_fusion > ones, ones, val_fusion)
            val_fusion = torch.where(val_fusion < zeros, zeros, val_fusion)
            
            # 保存多张样本的对比
            num_samples = min(4, val_vis.size(0))
            for idx in range(num_samples):
                comparison = torch.cat([
                    val_ir[idx:idx+1].cpu(),
                    val_vis[idx:idx+1].cpu(),
                    val_fusion[idx:idx+1].cpu()
                ], dim=3)
                writer.add_image(f'Epoch_{epo}/Sample_{idx}',
                               vutils.make_grid(comparison, normalize=True, scale_each=True),
                               epo)
            
            # 额外：保存一个4x3的网格图（4个样本，每个样本3张图）
            all_samples = []
            for idx in range(num_samples):
                all_samples.extend([
                    val_ir[idx:idx+1].cpu(),
                    val_vis[idx:idx+1].cpu(), 
                    val_fusion[idx:idx+1].cpu()
                ])
            grid = torch.cat(all_samples, dim=0)
            writer.add_image(f'Epoch_{epo}/Grid_Summary',
                           vutils.make_grid(grid, nrow=3, normalize=True, scale_each=True, padding=10),
                           epo)
        
        fusionmodel.train()
    
    # 保存模型
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')
    
    writer.close()
    logger.info('TensorBoard logging finished. Run: tensorboard --logdir=./runs')
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='VSSM_Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range(1):
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
    print("training Done!")
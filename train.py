import os

import argparse
import torch
from model_seg import Network3
from model_fu import FusionNet_Splitv2
from torch.utils.data import DataLoader
from Datasets import Fusion_dataset
from loss import Fusionloss
from model_distiller import DistillerNet
import logging
import time
import torch.nn as nn
import numpy as np
import random
import torch.nn.utils as nn_utils

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
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
        temp.reshape(
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
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--tchannel', type=list, default=[64, 128, 320, 512], help='teacher channel')
    parser.add_argument('--schannel', type=list, default=[16, 64, 128, 256], help='student channel')
    parser.add_argument('--batch_size', type=int, default=6, help='batch_size - reduced for memory')
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr_start', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--modelpth', type=str, default='./models', help='saving model path')
    parser.add_argument('--logpath', type=str, default='./logs', help='log path')
    parser.add_argument('--bi_hidden_size1', type=int, default=128, help='bi attention hidden size for layer 1')
    parser.add_argument('--bi_num_attention_heads1', type=int, default=2, help='number of attention heads for layer 1')
    parser.add_argument('--v_hidden_size', type=int, default=512, help='text feature dimension')
    parser.add_argument('--hidden_size1', type=int, default=128, help='visual feature dimension for layer 1')
    parser.add_argument('--v_attention_probs_dropout_prob', type=float, default=0.1,
                        help='dropout probability for text attention')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1,
                        help='dropout probability for visual attention')
    parser.add_argument('--bi_hidden_size2', type=int, default=128, help='bi attention hidden size for layer 2')
    parser.add_argument('--bi_num_attention_heads2', type=int, default=4, help='number of attention heads for layer 2')
    parser.add_argument('--hidden_size2', type=int, default=320, help='visual feature dimension for layer 2')

    opt = parser.parse_args()

    if not os.path.isdir(opt.modelpth):
        os.makedirs(opt.modelpth)
    if not os.path.isdir(opt.logpath):
        os.makedirs(opt.logpath)

    return opt


if __name__ == '__main__':
    opt = parse_option()
    Loss_list = []

    logger1 = logging.getLogger()

    best_loss = float('inf')
    best_epoch = 0

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    class BiAttentionConfig:
        def __init__(self, opt, layer):
            if layer == 1:
                self.bi_hidden_size = opt.bi_hidden_size1
                self.bi_num_attention_heads = opt.bi_num_attention_heads1
                self.v_hidden_size = opt.v_hidden_size
                self.hidden_size = opt.hidden_size1
            else:
                self.bi_hidden_size = opt.bi_hidden_size2
                self.bi_num_attention_heads = opt.bi_num_attention_heads2
                self.v_hidden_size = opt.v_hidden_size
                self.hidden_size = opt.hidden_size2
            self.v_attention_probs_dropout_prob = opt.v_attention_probs_dropout_prob
            self.attention_probs_dropout_prob = opt.attention_probs_dropout_prob

    bi_config1 = BiAttentionConfig(opt, 1)
    bi_config2 = BiAttentionConfig(opt, 2)
    fusionmodel = FusionNet_Splitv2(bi_config1, bi_config2)

    segmodel = Network3('mit_b3', num_classes=9)

    dmodel = DistillerNet(student_channels=opt.schannel, teacher_channels=opt.tchannel)

    model_list = nn.ModuleList([])
    model_list.append(fusionmodel)
    model_list.append(dmodel)
    model_list.append(segmodel)

    w = [0.5, 0.5]
    optimizer1 = torch.optim.Adam(fusionmodel.parameters(), lr=opt.lr_start, weight_decay=1e-5)  # 添加权重衰减

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1,
        T_max=opt.epochs,
        eta_min=opt.lr_start * 0.01
    )

    criterion_fusion = Fusionloss()

    if torch.cuda.is_available():
        model_list.cuda()

    train_dataset = Fusion_dataset()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    st = glob_st = time.time()

    logger1.info('Training Fusion Model start~')
    for epoch in range(opt.epochs):

        print('\n| epo #%s begin...' % epoch)

        lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Learning Rate: {lr}')

        for module in model_list:
            module.train()

        model_list[-1].eval()

        model_s = model_list[0]
        model_t = model_list[-1]

        total_loss_epoch = 0
        ototal_loss_epoch = 0

        for it, (image_vis, image_ir, vitext, irtext, name) in enumerate(train_loader):
            image_vis = image_vis.cuda()
            image_ir = image_ir.cuda()

            vitext = vitext.cuda()
            irtext = irtext.cuda()

            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_vis_y = image_vis_ycrcb[:, :1]
            feat, out = model_s(image_vis_y, image_ir, vitext, irtext)
            fusion_ycrcb = torch.cat(
                (out, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            fusion_image = YCrCb2RGB(fusion_ycrcb)

            optimizer1.zero_grad()
            fs, seg_map = model_t(fusion_image)

            dloss = dmodel(feat, fs)
            loss_fusion, loss_in, loss_grad = criterion_fusion(image_vis_ycrcb, image_ir, out)

            total_dloss = 0
            for i in range(len(dloss)):
                total_dloss += w[i] * dloss[i]

            loss_total = loss_fusion + 0.1 * total_dloss
            total_loss_epoch += loss_total.item()

            loss_total.backward()

            nn_utils.clip_grad_norm_(fusionmodel.parameters(), max_norm=1.0)

            optimizer1.step()

            scheduler.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            total_dloss = total_dloss.item()

            msg1 = ','.join(
                [
                    'epoch:{epoch}',
                    'pic_num:{it}',
                    'loss_total:{loss_total:.4f}',
                    'loss_in:{loss_in:.4f}',
                    'loss_grad:{loss_grad:.4f}',
                    'total_dloss:{total_dloss:.4f}',
                    'time: {time:.4f}',
                ]
            ).format(
                epoch=epoch,
                it=it,
                loss_total=loss_total.item(),
                loss_in=loss_in.item(),
                loss_grad=loss_grad.item(),
                total_dloss=total_dloss,
                time=t_intv,
            )

            logger1.info(msg1)

            st = ed
        avg_loss_epoch = total_loss_epoch / (it + 1)
        Loss_list.append(avg_loss_epoch)

        if (epoch + 1) % 30 == 0:
            torch.save(fusionmodel.state_dict(), f'{opt.modelpth}/fusion_model_SIM_epoch_{epoch + 1}.pth')
            logger1.info("Fusion Model Save to: {}".format(f'{opt.modelpth}/fusion_model_SIM_epoch_{epoch + 1}.pth'))

        if torch.isnan(loss_total).any():
            print(f"警告: 检测到NaN损失值在epoch {epoch}")

            if os.path.exists(f'{opt.modelpth}/fusion_model_SIM_best.pth'):
                print("从最佳模型恢复...")
                fusionmodel.load_state_dict(torch.load(f'{opt.modelpth}/fusion_model_SIM_best.pth'))

                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr * 0.1
                print(f"学习率已调整为: {lr * 0.1}")

        if avg_loss_epoch < best_loss:
            best_loss = avg_loss_epoch
            best_epoch = epoch + 1

            torch.save(fusionmodel.state_dict(), f'{opt.modelpth}/fusion_model_SIM_best.pth')
            logger1.info(
                f"Best model updated at epoch {best_epoch} with loss {best_loss:.4f}, saved to: {opt.modelpth}/fusion_model_SIM_best.pth")

    logger1.info(f"Training completed. Best model was at epoch {best_epoch} with loss {best_loss:.4f}")
    torch.save(fusionmodel.state_dict(), f'{opt.modelpth}/fusion_model_SIM_final.pth')
    logger1.info(f"Final model saved to: {opt.modelpth}/fusion_model_SIM_final.pth")

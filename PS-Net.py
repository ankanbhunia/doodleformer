import os
import datetime, wandb, cv2, shutil, time, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from evaluate import calculate_scores_given_paths
from utils.util import *
from model.resnet_generator_cond_context import *
from model.rcnn_discriminator_app import *
from utils.logger import setup_logger
from tqdm import tqdm
from torch.utils import data
from pathlib import Path
from bounding_box import bounding_box as bb



class Dataset_JSON(data.Dataset):

    def __init__(self):
        super().__init__()

        if 'bird' in  args.dataset:
            self.data = np.load(args.data_path, allow_pickle=True) 
        
        elif 'creature' in  args.dataset:
            self.data = np.load(args.data_path, allow_pickle=True) 

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):


        bbox = self.data[index]['bbox']
        intial_xy = self.data[index]['intial_xy']
        label = self.data[index]['label']
        raster = self.data[index]['raster']
        raster_initial = self.data[index]['raster_initial']
        text = self.data[index]['text']

        raster_initial = np.concatenate([raster_initial,raster_initial,raster_initial],0)
        return raster, label, bbox, intial_xy, raster_initial, text


def collate_fn(batch):

    batch = list(filter(lambda x: x is not None, batch))

    max_len = max(list((batch[i][3].shape[0] for i in range(len(batch)))))

    for idx, bt in enumerate(batch):

        batch[idx] = [batch[idx][0],batch[idx][1],batch[idx][2],
                     np.concatenate([batch[idx][3], np.zeros((max_len-len(batch[idx][3]), 2))],0),
                     batch[idx][4],
                     ]


    return torch.utils.data.dataloader.default_collate(batch)


def calculate_scores(epoch, test_dataloader, netG):

    output_path = '../output/'
   
    total_path = os.path.join(output_path,  args.exp_name + str('-') + str(epoch))

    if os.path.isdir(total_path):

        shutil.rmtree(total_path)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(total_path).mkdir(parents=True, exist_ok=True)

    print ('calculating scores : epoch '+str(epoch))
    
    for idx, batch in enumerate(tqdm(test_dataloader)):

        real_images, intial_storke,_, label, bbox = batch 

        intial_storke = intial_storke.to('cuda')
        bbox = bbox.to('cuda')
        label = label.to('cuda')

        #bbox_val = model.validate(label)

        #bbox_val, bbox_val = fix_bboxs(bbox, bbox_val, bbox_val)
        z = torch.randn(real_images.size(0), 9, 128).to('cuda:0')
        fake_images =  netG(z, intial_storke, bbox, y=label.long().to('cuda:0')).detach()

        for i in range(fake_images.shape[0]):

            im = (1 - (fake_images[i].permute(1,2,0).cpu().numpy() + 1)/2)*255

            im =  cv2.cvtColor(cv2.resize(im, (64, 64)),  cv2.COLOR_BGR2GRAY)

            cv2.imwrite(os.path.join(total_path, 'image'+str(idx*fake_images.shape[0] + i)+'.jpg'), im)

        if len(os.listdir(total_path))>10000:

            break

    fid_value, d1, d2, CS1, CS2, SDS1, SDS2 = calculate_scores_given_paths(['../data/bird_short_full_nodetail_64',total_path], 50, 1, 2048, 'birds')

    return fid_value, d1, d2, CS1, CS2, SDS1, SDS2





def visalize_bboxs(dataset, inp, out, label, bbox):

    if dataset == 'sketch-bird':

        id_to_part = {1:'initial', 2:'eye', 5:'head', 4:'body', 3:'beak', 6:'legs', 9:'wings', 7:'mouth', 8:'tail', 10: 'none'}
    
    elif dataset == 'sketch-generic':

        id_to_part = {1:'initial',  2:'eye',  3:'arms',  4:'beak',  5:'mouth',  6:'body',  7:'ears',  8:'feet',  9:'fin', 
                         10:'hair',  11:'hands',  12:'head',  13:'horns',  14:'legs',  15:'nose',  16:'paws',  17:'tail', 18:'wings', 19: 'none'}

    for i in range(len(label)):

        x, y, w, h = bbox[i]*128

        if bbox[i][0] > 0:

            bb.add(inp, x, y, x + w, y + h, id_to_part[label[i]])
            bb.add(out, x, y, x + w, y + h, id_to_part[label[i]])


    return inp, out

def main(args):
    # parameters
    img_size = 128
    z_dim = 128
    lamb_obj = 1.0
    lamb_app = 1.0
    lamb_img = 0.1

    if args.dataset == 'sketch-bird':

        num_classes = 10 #if args.dataset == 'coco' else 179
        num_obj = 9 #if args.dataset == 'coco' else 31

    elif args.dataset == 'sketch-generic':
        
        num_classes = 19 #if args.dataset == 'coco' else 179
        num_obj = 18 #if args.dataset == 'coco' else 31


    args.out_path = os.path.join(args.out_path, args.dataset + '_1gpu', str(img_size))

    # data loader


    num_gpus = torch.cuda.device_count()
    num_workers = 2
    if num_gpus > 1:
        parallel = True
        args.batch_size = args.batch_size * num_gpus
        num_workers = num_workers * num_gpus
    else:
        parallel = False

    data  = Dataset_JSON()

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, collate_fn=collate_fn, 
        drop_last=True, shuffle=True, num_workers=4)

    test_dataloader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, collate_fn=collate_fn, 
        drop_last=True, shuffle=True, num_workers=4)


    # Load model
    device = torch.device('cuda')
    netG = context_aware_generator(num_classes=num_classes, output_dim=3).to(device)
    netD = CombineDiscriminator128_app(num_classes=num_classes).to(device)


    if parallel:
        netG = DataParallelWithCallback(netG)
        netD = nn.DataParallel(netD)

    g_lr, d_lr = args.g_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]

    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))
    # writer = None
    logger = setup_logger("lostGAN", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)

    start_time = time.time()
    vgg_loss = VGGLoss()
    vgg_loss = nn.DataParallel(vgg_loss)
    l1_loss = nn.DataParallel(nn.L1Loss())
    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()
        print("Epoch {}/{}".format(epoch, args.total_epoch))
        for idx, data in enumerate(tqdm(dataloader)):

            real_images, label, bbox, intial_xy, intial_storke = data 
            real_images, intial_storke, label, bbox = real_images.to(device), intial_storke.to(device),  label.long().to(device).unsqueeze(-1), bbox.float()

            # update D network
            netD.zero_grad()
            real_images, label = real_images.to(device), label.long().to(device)
            d_out_real, d_out_robj, d_out_robj_app = netD(real_images, bbox, label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            d_loss_robj_app = torch.nn.ReLU()(1.0 - d_out_robj_app).mean()
            # print(d_loss_robj)
            # print(d_loss_robj_app)

            z = torch.randn(real_images.size(0), num_obj, z_dim).to(device)
            fake_images = netG(z, intial_storke, bbox, y=label.squeeze(dim=-1))
            
            d_out_fake, d_out_fobj, d_out_fobj_app = netD(fake_images.detach(), bbox, label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            d_loss_fobj_app = torch.nn.ReLU()(1.0 + d_out_fobj_app).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake) + lamb_app * (d_loss_robj_app + d_loss_fobj_app)
            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj, g_out_obj_app = netD(fake_images, bbox, label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                g_loss_obj_app = - g_out_obj_app.mean()

                pixel_loss = l1_loss(fake_images, real_images).mean()
                feat_loss = vgg_loss(fake_images, real_images).mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + pixel_loss + feat_loss + lamb_app * g_loss_obj_app
                g_loss.backward()
                g_optimizer.step()
    
            if (idx + 1) % 50 == 0:
                

                inpimg_list = []
                outimg_list = []

                for j in range(bbox.shape[0]):

                    inpimg = 255 - cv2.cvtColor((real_images[j].permute(1,2,0).detach().cpu().numpy() + 1)/2, cv2.COLOR_BGR2RGB)*255
                    
                    
                    outimg = 255 - cv2.cvtColor((fake_images[j].permute(1,2,0).detach().cpu().numpy() + 1)/2, cv2.COLOR_BGR2RGB)*255

                    lab = label[j].detach().cpu().numpy()[:,0]

                    bb = bbox[j].detach().cpu().numpy()

                    inpimg, outimg = visalize_bboxs(args.dataset, inpimg, outimg, lab, bb)

                    inpimg_list.append(inpimg)
                    outimg_list.append(outimg)

                inpimg_list = torch.Tensor(np.stack(inpimg_list, 0).transpose(0,3, 1, 2))
                outimg_list = torch.Tensor(np.stack(outimg_list, 0).transpose(0,3, 1, 2))


                wandb.log({'d_out_real': d_loss_real,
                    'd_out_fake': d_loss_fake,  
                    'g_loss_fake': g_loss_fake, 
                    'd_obj_real': d_loss_robj, 
                    'd_obj_fake': d_loss_fobj, 
                    'g_obj_fake': g_loss_obj, 
                    'd_loss_robj_app': d_loss_robj_app, 
                    'd_loss_fobj_app': d_loss_fobj_app, 
                    'g_loss_obj_app': g_loss_obj_app, 
                    'pixel_loss': pixel_loss, 
                    'feat_loss': feat_loss, 
                    })       
                
                caption = '(a) initial stfoke (b) Fake output (c) Real Input (d) Fake bbox (e) Real bbox' 

                wandb.log({#"Images":wandb.Image(1 - (torch.cat([fake_images, real_images], -1) + 1)/2), 

                          "bbox":[wandb.Image(image, caption = caption) for image in torch.cat([255 - ((intial_storke.cpu()+1)*255)/2, 255 - ((fake_images.cpu()+1)*255)/2,255 - ((real_images.cpu()+1)*255)/2, outimg_list, inpimg_list], -1)],
                    #"Full_Gen":wandb.Image(generated_images),
                    })

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                logger.info("Step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                              idx + 1,
                                                                                                              d_loss_real.item(),
                                                                                                              d_loss_fake.item(),
                                                                                                              g_loss_fake.item()))
                logger.info("             d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                    d_loss_robj.item(),
                    d_loss_fobj.item(),
                    g_loss_obj.item()))
                logger.info("             d_obj_real_app: {:.4f}, d_obj_fake_app: {:.4f}, g_obj_fake_app: {:.4f} ".format(
                    d_loss_robj_app.item(),
                    d_loss_fobj_app.item(),
                    g_loss_obj_app.item()))

                logger.info("             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))
                if writer is not None:
                    writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)
                    writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)

                    writer.add_scalars("D_loss_real", {"real": d_loss_real.item(),
                                                       "robj": d_loss_robj.item(),
                                                       "robj_app": d_loss_robj_app.item(),
                                                       "loss": d_loss.item()})
                    writer.add_scalars("D_loss_fake", {"fake": d_loss_fake.item(),
                                                       "fobj": d_loss_fobj.item(),
                                                       "fobj_app": d_loss_fobj_app.item()})
                    writer.add_scalars("G_loss", {"fake": g_loss_fake.item(),
                                                  "obj_app": g_loss_obj_app.item(),
                                                  "obj": g_loss_obj.item(),
                                                  "loss": g_loss.item()})

        # save model 
        if (epoch + 1) % 5 == 0:

            torch.save(netG.state_dict(), os.path.join(args.model_dir,  args.exp_name, 'G_%d.pth' % (epoch + 1)))
            torch.save(netD.state_dict(), os.path.join(args.model_dir,  args.exp_name, 'D_%d.pth' % (epoch + 1)))

        if (epoch + 1) % 10 == 0:

            try:
                fid_value, d1, d2, CS1, CS2, SDS1, SDS2 = calculate_scores(epoch, test_dataloader, netG)
                wandb.log({'fid_value': fid_value, 'GD' : d2, 'CS' : CS2, 'SDS':SDS2}) 
            except:
                print ('error when trying to obtain scores.')


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='doodleformer-psnet-training-stage-2')

    parser.add_argument('--dataset', type=str, default='sketch-bird',
                    help='training dataset')
    parser.add_argument('--exp_name', type=str, default='layout2sketch')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='mini-batch size of training data. Default: 32')
    parser.add_argument('--total_epoch', type=int, default=200,
                    help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                    help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                    help='learning rate for generator')
    parser.add_argument('--data_path', type=str, default='../../data/doodledata.npy')
    parser.add_argument('--out_path', type=str, default='./outputs/tmp/app')
    parser.add_argument('--wandb_dir', type=str, default='.')
    parser.add_argument('--model_dir', type=str, default='../../models/')

    args = parser.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.model_dir, args.exp_name)).mkdir(parents=True, exist_ok=True)

    wandb.init(settings=wandb.Settings(start_method='fork'), project="doodleformer", name =  args.exp_name)#

    main(args)

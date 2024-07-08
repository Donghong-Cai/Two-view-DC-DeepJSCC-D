# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import time
from models import create_model
from options.train_options import TrainOptions
import os
import torchvision.utils as vutils
import numpy as np
from dataload import load_data
from collections import OrderedDict

# Extract the options
opt = TrainOptions().parse()

# Prepare the dataset
extracted_folder = './data'
dataset,eval_dataset = load_data(extracted_folder,opt.batch_size)
dataset_size = len(dataset)
eval_dataset_size = len(eval_dataset)
print('#training images = %d' % dataset_size)
print('#testing images = %d' % eval_dataset_size)

# Create the checkpoint folder
opt.name = 'C32_COIL100'
path = os.path.join(opt.checkpoints_dir, opt.name)
if not os.path.exists(path):
    os.makedirs(path)    

# Initialize the model
device="cuda"
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

# download checkpoints


total_iters = 0                # the total number of training iterations
total_epoch = opt.n_epochs_joint

# Setupt the warmup stage
print('Joint learning stage begins!')
print(f'Learning rate changed to {opt.lr_joint}')
train_losses=[]
train_psnrs=[]
val_losses =[]
val_psnrs =[]
for epoch in range(opt.epoch_count, total_epoch + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    model.train()


    epoch_losss=[]
    epoch_psnrs1=[]
    epoch_psnrs2=[]
    for i, batch in enumerate(dataset):  # inner loop within one epoch
        view1,view2,label=batch
        data=torch.stack((view1,view2),dim=1)
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        model.set_input(data)    # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        loss=model.loss_G
        epoch_losss.append(loss.item())
        fake = model.fake

        # Get the int8 generated images
        img1_gen_numpy = fake[:,0,...].detach().cpu().float().numpy()
        img1_gen_numpy = np.transpose(img1_gen_numpy, (0, 2, 3, 1)) * 255.0
        img1_gen_int8 = img1_gen_numpy.astype(np.uint8)

        origin1_numpy = data[:,0,...].detach().cpu().float().numpy()
        origin1_numpy = (np.transpose(origin1_numpy, (0, 2, 3, 1))) * 255.0
        origin1_int8 = origin1_numpy.astype(np.uint8)

        diff1 = np.mean((np.float64(img1_gen_int8) - np.float64(origin1_int8))**2, (1, 2, 3))

        PSNR1 = 10 * np.log10((255**2) / diff1)
        epoch_psnrs1.append(PSNR1)
        img2_gen_numpy = fake[:,1,...].detach().cpu().float().numpy()
        img2_gen_numpy = (np.transpose(img2_gen_numpy, (0, 2, 3, 1)) )  * 255.0
        img2_gen_int8 = img2_gen_numpy.astype(np.uint8)

        origin2_numpy = data[:,1,...].detach().cpu().float().numpy()
        origin2_numpy = (np.transpose(origin2_numpy, (0, 2, 3, 1)) )  * 255.0
        origin2_int8 = origin2_numpy.astype(np.uint8)

        diff2 = np.mean((np.float64(img2_gen_int8) - np.float64(origin2_int8))**2, (1, 2, 3))

        PSNR2 = 10 * np.log10((255**2) / diff2)
        epoch_psnrs2.append(PSNR2)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
            for k, v in losses.items():
                message += '%s: %.5f ' % (k, v)
            print(message)  # print the message
            log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
            
        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)
            save_filename = '%s_net_%s.pth' % (save_suffix, 'device_images')
            save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name) , save_filename)
            torch.save(model.device_images.state_dict(),save_path)
            iter_data_time = time.time()
    epoch_losss=torch.tensor(epoch_losss).mean()
    train_losses.append(epoch_losss.item())
    epoch_psnr=torch.tensor(np.array(epoch_psnrs1)).mean()+torch.tensor(np.array(epoch_psnrs2)).mean()
    epoch_psnr=epoch_psnr/2
    train_psnrs.append(epoch_psnr.item())
    print('epoch',epoch,'train_loss:',epoch_losss.item(),'train_psnr:',epoch_psnr.item())

    #save images
    if epoch%50==0:
        data=[view1,view2]

        num_images = 8 * 8 / 2
        num_images = np.int_(num_images)

        originals1 = data[0][:num_images].cpu()
        reconstructions1 = fake[:,0,...][:num_images].cpu()

        originals2 = data[1][:num_images].cpu()
        reconstructions2 = fake[:,1,...][:num_images].cpu()

        comparison1 = torch.cat([reconstructions1, originals1])
        comparison2 = torch.cat([reconstructions2, originals2])
        comparisons=[comparison1,comparison2]

        grid1 = vutils.make_grid(comparisons[0].data, nrow=8)
        grid2 = vutils.make_grid(comparisons[1].data, nrow=8)

        vutils.save_image(grid1,'./img/{}_view1.png'.format(epoch),nrow=8,normalize=True)
        vutils.save_image(grid2, './img/{}_view2.png'.format(epoch), nrow=8, normalize=True)
        print('save image')

    '''-------'''
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)
        save_filename = '%s_net_%s.pth' % (epoch, 'device_images')
        save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name) , save_filename)
        torch.save(model.device_images.state_dict(),save_path)


    if epoch%1 == 0:
        model.eval()
        epoch_val_losses =[]
        psnrs1=[]
        psnrs2=[]
        with torch.no_grad():
            for i, testbatch in enumerate(eval_dataset):  # inner loop within one epoch
                view1,view2,label=testbatch
                data=torch.stack((view1,view2),dim=1)
                model.set_input(data)
                model.forward()
                fake = model.fake
                input = model.real_B
                model.backward_G()
                loss=model.loss_G
                epoch_val_losses.append(loss.item())
                # Get the int8 generated images
                img1_gen_numpy = fake[:,0,...].detach().cpu().float().numpy()
                img1_gen_numpy = (np.transpose(img1_gen_numpy, (0, 2, 3, 1)) )  * 255.0
                img1_gen_int8 = img1_gen_numpy.astype(np.uint8)

                origin1_numpy = input[:,0,...].detach().cpu().float().numpy()
                origin1_numpy = (np.transpose(origin1_numpy, (0, 2, 3, 1)) )  * 255.0
                origin1_int8 = origin1_numpy.astype(np.uint8)

                diff1 = np.mean((np.float64(img1_gen_int8) - np.float64(origin1_int8))**2, (1, 2, 3))

                PSNR1 = 10 * np.log10((255**2) / diff1)
                psnrs1.append(PSNR1)
                img2_gen_numpy = fake[:,1,...].detach().cpu().float().numpy()
                img2_gen_numpy = (np.transpose(img2_gen_numpy, (0, 2, 3, 1)) )  * 255.0
                img2_gen_int8 = img2_gen_numpy.astype(np.uint8)

                origin2_numpy = input[:,1,...].detach().cpu().float().numpy()
                origin2_numpy = (np.transpose(origin2_numpy, (0, 2, 3, 1)) )  * 255.0
                origin2_int8 = origin2_numpy.astype(np.uint8)

                diff2 = np.mean((np.float64(img2_gen_int8) - np.float64(origin2_int8))**2, (1, 2, 3))

                PSNR2 = 10 * np.log10((255**2) / diff2)
                psnrs2.append(PSNR2)
            val_losses.append(torch.tensor(epoch_val_losses).mean().item())
            mean_psnr=(torch.tensor(np.array(psnrs1)).mean()+torch.tensor(np.array(psnrs2)).mean())*0.5
            val_psnrs.append(mean_psnr.item())
            psnr=[torch.tensor(np.array(psnrs1)).mean(),torch.tensor(np.array(psnrs2)).mean()]
            print("val_loss:",torch.tensor(epoch_val_losses).mean().item(),"psnr1:",mean_psnr.item())

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, total_epoch, time.time() - epoch_start_time))


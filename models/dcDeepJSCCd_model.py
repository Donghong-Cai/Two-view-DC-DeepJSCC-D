# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

from .base_model import BaseModel
from . import networks
import math
from utils import *
import torch.nn.functional as F

class dcDeepJSCCdModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L','PSNR1','PSNR2','G_Com','G_Dis']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['E', 'D','Dis']

        # define networks
        self.netE = networks.define_E( norm=opt.norm, init_type=opt.init_type,
                                       init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.netD = networks.define_D( norm=opt.norm, init_type=opt.init_type,
                                       init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.device_images = torch.nn.Embedding(2, embedding_dim=32 * 32).to(self.device)

        self.netDis=networks.define_Dis(init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids)



        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            # self.criterionL2 = torch.nn.BCELoss()
            params = list(self.netE.parameters()) + list(self.netD.parameters()) +list(self.device_images.parameters())+list(self.netDis.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr_joint, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)

        self.opt = opt
        # self.temp = opt.temp_init if opt.isTrain else 5

    def name(self):
        return 'DC_DeepJSCC_D_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def forward(self):

        # Generate SNR
        if self.opt.isTrain:
            self.snr = torch.rand(self.real_A.shape[0], 1).to(self.device) * (self.opt.SNR_MAX-self.opt.SNR_MIN) + self.opt.SNR_MIN
        else:
            self.snr = torch.ones(self.real_A.shape[0], 1).to(self.device) * self.opt.SNR

        emb = torch.stack(
            [
                self.device_images(
                    torch.ones((self.real_A.size(0)), dtype=torch.long, device=self.device) * i
                ).view(self.real_A.size(0), 1, 32, 32)
                for i in range(2)
            ],
            dim=1,
        )
        x = torch.cat([self.real_A, emb], dim=2)
        # Generate latent vector
        transmissions=[]
        for i in range(2):
            latent = self.netE(x[:,i,...], self.snr)
            transmissions.append(latent)
        transmission_stacked = torch.stack(transmissions, dim=1)
        self.disfeature = self.netDis(transmission_stacked)
        # disfeature_reshape= torch.stack((self.disfeature [:,0,...].view(latent.shape),self.disfeature [:,1,...].view(latent.shape)),dim=1)
        self.Feature=transmission_stacked


        # Normalize each channel
        hids_shape = self.disfeature .size() #hids.size() batch,_,C,H,W)
        hids = self.disfeature .view(hids_shape[0] * hids_shape[1], 2, -1) #(batch*_,2,C*H*W/2)
        hids = torch.complex(hids[:, 0, :], hids[:, 1, :]) #(batch*_,C*H*W/2)
        norm_factor = 1.0*torch.sqrt(1.0 / torch.tensor(hids_shape[1], dtype=torch.float32))  * torch.sqrt(torch.tensor(hids.real.size(1), dtype=torch.float32)).cuda()
        hids = hids * torch.complex(norm_factor/torch.sqrt(torch.sum((hids * torch.conj(hids)).real, keepdims=True, dim=1)), torch.tensor(0.0, device=hids.device))
        hids = torch.cat([hids.real, hids.imag], dim=1)
        hids = hids.view(hids_shape)

        # Pass through the AWGN channel

        latent_res = torch.sum(hids, dim=1)
        with torch.no_grad():
            sigma = 10**(-self.snr / 20)
            noise = sigma.view(self.real_A.shape[0], 1, 1,1) * torch.randn_like(latent_res)
            noise = noise * torch.sqrt(torch.tensor(0.5, device=x.device))

        latent_res = latent_res + noise

        self.noise_feature=latent_res.to(self.device)
        recon = self.netD(latent_res.view(latent.shape), self.snr)
        xi = recon.view(recon.size(0), 2, 3, recon.size(2), recon.size(3))
        self.fake=xi


    def backward_G(self):

        self.loss_G_L =torch.stack(
            [
                self.criterionL2(self.fake[:, i, ...], self.real_B[:, i, ...])
                for i in range(2)
            ]
        ).sum()
        N,_,dim,_,_=self.disfeature.size()
        hs1=self.disfeature[:,0,...][:,:dim//2,...].view(N,-1)
        hc1=self.disfeature[:,0,...][:,dim//2:,...].view(N,-1)
        hs2=self.disfeature[:,1,...][:,:dim//2,...].view(N,-1)
        hc2=self.disfeature[:,1,...][:,dim//2:,...].view(N,-1)

        self.loss_G_Dis=(loss_dependence(hs1,hc1,N)+loss_dependence(hs2,hc2,N))/0.001
        self.loss_G_Com=CMD(hc1,hc2,2)

        self.loss_PSNR1= 10 * np.log10((1**2) / torch.nn.MSELoss()(self.real_B[:,0,...],self.fake[:,0,...]).detach().cpu().float().numpy())
        self.loss_PSNR2= 10 * np.log10((1**2) / torch.nn.MSELoss()(self.real_B[:,1,...],self.fake[:,1,...]).detach().cpu().float().numpy())

        self.loss_G = self.opt.lambda_L2 * self.loss_G_L+ self.opt.lambda_D * self.loss_G_Dis +  self.opt.lambda_C * self.loss_G_Com


    def optimize_parameters(self):

        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.loss_G.backward()
        self.optimizer_G.step()             # udpate G's weights





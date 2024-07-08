# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.test_options import TestOptions
from dataload import *
from collections import OrderedDict
from utils import *


def load_and_update_model(model, name, device):
    load_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name) , name)
    state_dict = torch.load(load_path, map_location=str(device))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = "module." + k
        new_state_dict[name] = v
    return model.load_state_dict(new_state_dict)

# Extract the options
opt = TestOptions().parse()
device="cuda"


extracted_folder = './data/test'
dataset=TwoCamerasDataset(extracted_folder,transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))
test_dataloader = DataLoader(dataset, batch_size=16, shuffle=True,drop_last=True)



dataset_size = len(test_dataloader)
print('#test images = %d' % dataset_size)

opt.name = 'C32_COIL100'

model = create_model(opt)
# download checkpoints
load_and_update_model(model.netE, 'net_E.pth', device)
load_and_update_model(model.netD, 'net_D.pth', device)
load_and_update_model(model.netDis, 'net_Dis.pth', device)
load_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name) , 'net_device_images.pth')
state_dict = torch.load(load_path, map_location=str(device))
model.device_images.load_state_dict(state_dict)
model.eval()

PSNR_list = []
PSNR_view_list = [[] for _ in range(2)]
Features=[]
Labels=[]
inputdata=[]
noise_Features=[]
for i, data in enumerate(test_dataloader):
    view1,view2,label=data
    print(i)
    if i >= 24:#opt.num_test:  # only apply our model to opt.num_test images.
        break
    start_time = time.time()
    input = torch.stack((view1,view2),dim=1)
    model.set_input(input)
    model.forward()
    fake = model.fake

    #解纠缠显示
    inputdata.append(input)
    Features.append(model.Feature)
    noise_Features.append(model.noise_feature)

    PSNR= 10 * np.log10((1**2) / torch.nn.MSELoss()(fake[:,0,...],model.real_B[:,0,...]).detach().cpu().float().numpy())
    # Get the int8 generated images
    for i in range (2):
        img_gen_numpy = fake[:,i,...].detach().cpu().float().numpy()
        img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)))  * 255.0
        img_gen_int8 = img_gen_numpy.astype(np.uint8)

        origin_numpy = input[:,i,...].detach().cpu().float().numpy()
        origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) ) * 255.0
        origin_int8 = origin_numpy.astype(np.uint8)

        diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))

        PSNR = 10 * np.log10((255**2) / diff)
        PSNR_view_list[i].append(np.mean(PSNR))
#
#
print("测试SNR：",opt.SNR)
print(f'Mean PSNR1: {np.mean(PSNR_view_list[0]):.3f}')
print(f'Mean PSNR2: {np.mean(PSNR_view_list[1]):.3f}')
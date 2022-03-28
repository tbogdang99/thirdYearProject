from unittest import result, skip
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance


import os
import glob
import time
import numpy as np
import concurrent.futures
from pathlib import Path
from tqdm.notebook import tqdm
from skimage.color import rgb2lab, lab2rgb
import base64

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import threading


from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet

# Select the device to be cuda in order to use the GPU
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G



def build_res34_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

SIZE = 256

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm


from IPython import embed


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))



class ColorizationDataset(Dataset):
    def __init__(self, paths):
        self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        self.size = SIZE
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Open the image
        img = Image.open(self.paths[idx]).convert("RGB")
        # Transform the image to the corresponding sizes
        img = self.transforms(img)
        # Convert it to an np array
        img = np.array(img)
        # Convert from RGB representation to LAB and convert it to tensor
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        #Normalize the values   
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}


SIZE = 256
class ColorizationDataset2(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] # Between -1 and 1
        ab = img_lab[[1, 2], ...] # Between -1 and 1
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

#Code from the internet, an implementation of the paper
# Unet with skip connections
class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)


#PatchGan discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        # Multiply by 0.5 to slow down the rate at which D learns relative to G
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        # Alternate between one gradient descent step on D
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        # And then one sten on G
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


uploaded_file = None
st.set_page_config(layout="wide")
sidebar_title = '<p style="text-align: center; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 40px;">ColorAIze</p>'
st.sidebar.markdown(sidebar_title, unsafe_allow_html=True)
add_selectbox = st.sidebar.selectbox(
    "Choose the page you would like to acess",
    ("Home Page 🏠","Upload 📎", "Take picture 📷")
)

img = None
option = "Pix2PixModel"



results = []
def thread_function(name):
    #print(name)
    file_ = open(name + ".gif", "rb")
    #print(name + ".gif")
    contents = file_.read()
    name= base64.b64encode(contents).decode("utf-8")
    #print(name)
    #st.markdown(
    #f'<img src="data:image/gif;base64,{name}" width="300" height="300" alt="cat gif">',
    #unsafe_allow_html=True,
    #)
    file_.close()
    results.append(name)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(thread_function, ["gif_livingroom","gif_abraham","gif_livingroom2","gif_tesla"])




if add_selectbox == "Home Page 🏠":
    application_title1 = '<p style="text-align: center; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 50px;">Welcome to ColorAIze</p>'
    st.markdown(application_title1, unsafe_allow_html=True)
    welcome_text = '<p style="text-align: center; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 25px;">Explore the power of deep learning & AI</p>'
    st.markdown(welcome_text, unsafe_allow_html=True)

    col1, col2, col3,col4 = st.columns([1,1,1,1])

    with col1:
        st.markdown(
        f'<img src="data:image/gif;base64,{results[0]}" width="300" height="300" alt="cat gif">',
        unsafe_allow_html=True,
        )
        
    with col2:
        st.markdown(
        f'<img src="data:image/gif;base64,{results[1]}" width="300" height="300" alt="cat gif">',
        unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
        f'<img src="data:image/gif;base64,{results[2]}" width="300" height="300" alt="cat gif">',
        unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
        f'<img src="data:image/gif;base64,{results[3]}" width="300" height="300" alt="cat gif">',
        unsafe_allow_html=True,
        )

    next_text = '<p style="text-align: center; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 25px;">Select from the sidebar either you want to upload a picture or take one</p>'
    st.markdown(next_text, unsafe_allow_html=True)

if add_selectbox == "Upload 📎":
    application_title = '<p style="text-align: center; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 40px;">ColorAIze</p>'
    st.markdown(application_title, unsafe_allow_html=True)
    col1, col2, col3,col4 = st.columns([1,1,1,1])
    with col1:
        option = st.selectbox(
        'Select the model',
        ('Pix2PixModel','ECCVGenerator'))
    if option == 'Pix2PixModel':        
        with col2:      
            option2 = st.selectbox(
            'Select weigths:',
            ('Random Weights', 'Unet COCO dataset', 'Unet COCO dataset Resnet18', 'Unet COCO dataset Resnet34',
            'Unet CelebA dataset','Unet CelebA dataset Resnet18', 'Unet CelebA dataset Resnet34' ))
            if option2 =='Random Weights':
                model = MainModel()
            elif option2 == 'Unet COCO dataset':
                with col3:
                    option20 = st.selectbox(
                    'Select training size:',
                    ('8000 images', '16000 images', '20000 images'))
                    if option20 == '8000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("UNET_trained_by_me/UNET-trainedbyme8000pictures16batchsize.pth", map_location=device))
                    if option20 == '16000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("UNET_trained_by_me/UNET-trainedbyme16000pictures16batchsize.pth", map_location=device))
                    if option20 == '20000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("UNET_trained_by_me/UNET-trainedbyme20000pictures16batchsizeV2.pth", map_location=device))

            elif option2 == 'Unet COCO dataset Resnet18':
                with col3:
                    option21 = st.selectbox(
                    'Select version of Resnet18:',
                    ('Not pretrained', 'Pretrained'))
                    if option21 == 'Not pretrained':
                        with col4:
                            option22 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option22 == '8000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET18/UNET-trainedbyme8000pictures16batchsizeResnet18.pth", map_location=device))   
                            if option22 == '16000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET18/UNET-trainedbyme16000pictures16batchsizeResnet18.pth", map_location=device))       

                    if option21 == 'Pretrained':
                        with col4:
                            option23 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option23 == '8000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbyme/Resnet-18-pretrainedbyme8000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained/UnetTrainedByMe8000pictures16batchsizeResnet18-pretrainedbyme_with_data_augmentationV2.pth", map_location=device))   
                            if option23 == '16000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbyme/Resnet-18-pretrainedbyme16000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained/UnetTrainedByMe16000pictures16batchsizeResnet18-pretrainedbyme_with_data_augmentationV2.pth", map_location=device))

            elif option2 == 'Unet COCO dataset Resnet34':
                with col3:
                    option24 = st.selectbox(
                    'Select version of Resnet18:',
                    ('Not pretrained', 'Pretrained'))
                    if option24 == 'Not pretrained':
                        with col4:
                            option25 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option25 == '8000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET34/UNET-trainedbyme8000pictures16batchsizeResnet34.pth", map_location=device))   
                            if option25 == '16000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET34/UNET-trainedbyme16000pictures16batchsizeResnet34.pth", map_location=device))       

                    if option24 == 'Pretrained':
                        with col4:
                            option26 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option26 == '8000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_34_pretrainedbyme/Resnet-34-pretrainedbyme8000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet34_pretrained/UnetTrainedByMe8000pictures16batchsizeResnet34-pretrainedbyme_with_data_augmentationV2.pth", map_location=device))   
                            if option26 == '16000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_34_pretrainedbyme/Resnet-34-pretrainedbyme16000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet34_pretrained/UnetTrainedByMe16000pictures16batchsizeResnet34-pretrainedbyme_with_data_augmentationV3.pth", map_location=device))

            elif option2 == 'Unet CelebA dataset':
                with col3:
                    option27 = st.selectbox(
                    'Select training size:',
                    ('8000 images', '18000 images'))
                    if option27 == '8000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("Unet_trained_by_me_CELEBAdataset/UnetTrainedByMe8000picturescelebADataset.pth", map_location=device))
                    if option27 == '18000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("Unet_trained_by_me_CELEBAdataset/UnetTrainedByMe18000picturescelebADataset.pth", map_location=device))

            elif option2 == 'Unet CelebA dataset Resnet18':
                with col3:
                    option28 = st.selectbox(
                    'Select training size:',
                    ('8000 images', '18000 images'))

                    if option28 == '8000 images':
                        net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                        net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbymeCELEBA/Resnet-18-pretrainedbyme8000picturescelebAdataset.pth", map_location=device))
                        model = MainModel(net_G=net_G_loaded)
                        model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained_CELEBA/UnetTrainedByMe8000picturescelebADataset-pretrained_Resnet18.pth", map_location=device))                             
                    if option28 == '18000 images':
                        net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                        net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbymeCELEBA/Resnet-18-pretrainedbyme18000picturescelebAdataset.pth", map_location=device))
                        model = MainModel(net_G=net_G_loaded)
                        model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained_CELEBA/UnetTrainedByMe18000picturescelebADataset_using_pretrained_Resnet18_18000pictures.pth", map_location=device))  
 
            elif option2 == 'Unet CelebA dataset Resnet34':
                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                net_G_loaded.load_state_dict(torch.load("RESNET_34_pretrainedbymeCELEBA/Resnet-34-pretrainedbyme18000picturescelebADataset.pth", map_location=device))
                model = MainModel(net_G=net_G_loaded)
                model.load_state_dict(torch.load("UNET_with_Resnet34_pretrained_CELEBA/UnetTrainedByMe18000picturescelebADataset_using_pretrained_Resnet34_18000pictures.pth", map_location=device)) 
                
            

    if option == "ECCVGenerator":
        with col2:
            option2 = st.selectbox(
            'Select weigths',
            ('Random Weights', 'Unet Coco Dataset'))
            if option2 =='Random Weights':
                model = ECCVGenerator()           
            elif option2 == 'Unet Coco Dataset':
                model = ECCVGenerator()
                model.load_state_dict(torch.load("ECCVWEIGHTS.pth", map_location=device))

    
    upload_photo_here = '<p style="text-align: left; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 40px;">Upload your photo here</p>'
    st.markdown(upload_photo_here, unsafe_allow_html=True)
    #Add file uploader to allow users to upload photos
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])



if add_selectbox == "Take picture 📷":
    take_picture_title = '<p style="text-align: center; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 60px;">Take a picture to test it yourself!</p>'
    st.markdown(take_picture_title, unsafe_allow_html=True)
    picture = st.camera_input("")
    col1, col2, col3,col4 = st.columns([1,1,1,1])
    with col1:
        option = st.selectbox(
        'Select the model',
        ('Pix2PixModel','ECCVGenerator'))
    if option == 'Pix2PixModel':        
        with col2:      
            option2 = st.selectbox(
            'Select weigths:',
            ('Random Weights', 'Unet COCO dataset', 'Unet COCO dataset Resnet18', 'Unet COCO dataset Resnet34',
            'Unet CelebA dataset','Unet CelebA dataset Resnet18', 'Unet CelebA dataset Resnet34' ))
            if option2 =='Random Weights':
                model = MainModel()
            elif option2 == 'Unet COCO dataset':
                with col3:
                    option20 = st.selectbox(
                    'Select training size:',
                    ('8000 images', '16000 images', '20000 images'))
                    if option20 == '8000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("UNET_trained_by_me/UNET-trainedbyme8000pictures16batchsize.pth", map_location=device))
                    if option20 == '16000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("UNET_trained_by_me/UNET-trainedbyme16000pictures16batchsize.pth", map_location=device))
                    if option20 == '20000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("UNET_trained_by_me/UNET-trainedbyme20000pictures16batchsizeV2.pth", map_location=device))

            elif option2 == 'Unet COCO dataset Resnet18':
                with col3:
                    option21 = st.selectbox(
                    'Select version of Resnet18:',
                    ('Not pretrained', 'Pretrained'))
                    if option21 == 'Not pretrained':
                        with col4:
                            option22 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option22 == '8000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET18/UNET-trainedbyme8000pictures16batchsizeResnet18.pth", map_location=device))   
                            if option22 == '16000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET18/UNET-trainedbyme16000pictures16batchsizeResnet18.pth", map_location=device))       

                    if option21 == 'Pretrained':
                        with col4:
                            option23 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option23 == '8000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbyme/Resnet-18-pretrainedbyme8000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained/UnetTrainedByMe8000pictures16batchsizeResnet18-pretrainedbyme_with_data_augmentationV2.pth", map_location=device))   
                            if option23 == '16000 pictures':
                                net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbyme/Resnet-18-pretrainedbyme16000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained/UnetTrainedByMe16000pictures16batchsizeResnet18-pretrainedbyme_with_data_augmentationV2.pth", map_location=device))

            elif option2 == 'Unet COCO dataset Resnet34':
                with col3:
                    option24 = st.selectbox(
                    'Select version of Resnet18:',
                    ('Not pretrained', 'Pretrained'))
                    if option24 == 'Not pretrained':
                        with col4:
                            option25 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option25 == '8000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET34/UNET-trainedbyme8000pictures16batchsizeResnet34.pth", map_location=device))   
                            if option25 == '16000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_trained_by_me_RESNET34/UNET-trainedbyme16000pictures16batchsizeResnet34.pth", map_location=device))       

                    if option24 == 'Pretrained':
                        with col4:
                            option26 = st.selectbox(
                            'Select training size:',
                            ('8000 pictures', '16000 pictures'))
                            if option26 == '8000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_34_pretrainedbyme/Resnet-34-pretrainedbyme8000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet34_pretrained/UnetTrainedByMe8000pictures16batchsizeResnet34-pretrainedbyme_with_data_augmentationV2.pth", map_location=device))   
                            if option26 == '16000 pictures':
                                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                                net_G_loaded.load_state_dict(torch.load("RESNET_34_pretrainedbyme/Resnet-34-pretrainedbyme16000pictures_with_data-Augmentation.pth", map_location=device))
                                model = MainModel(net_G=net_G_loaded)
                                model.load_state_dict(torch.load("UNET_with_Resnet34_pretrained/UnetTrainedByMe16000pictures16batchsizeResnet34-pretrainedbyme_with_data_augmentationV3.pth", map_location=device))

            elif option2 == 'Unet CelebA dataset':
                with col3:
                    option27 = st.selectbox(
                    'Select training size:',
                    ('8000 images', '18000 images'))
                    if option27 == '8000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("Unet_trained_by_me_CELEBAdataset/UnetTrainedByMe8000picturescelebADataset.pth", map_location=device))
                    if option27 == '18000 images':
                        model = MainModel()
                        model.load_state_dict(torch.load("Unet_trained_by_me_CELEBAdataset/UnetTrainedByMe18000picturescelebADataset.pth", map_location=device))

            elif option2 == 'Unet CelebA dataset Resnet18':
                with col3:
                    option28 = st.selectbox(
                    'Select training size:',
                    ('8000 images', '18000 images'))

                    if option28 == '8000 images':
                        net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                        net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbymeCELEBA/Resnet-18-pretrainedbyme8000picturescelebAdataset.pth", map_location=device))
                        model = MainModel(net_G=net_G_loaded)
                        model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained_CELEBA/UnetTrainedByMe8000picturescelebADataset-pretrained_Resnet18.pth", map_location=device))                             
                    if option28 == '18000 images':
                        net_G_loaded = build_res_unet(n_input=1, n_output=2, size=256)
                        net_G_loaded.load_state_dict(torch.load("RESNET_18_pretrainedbymeCELEBA/Resnet-18-pretrainedbyme18000picturescelebAdataset.pth", map_location=device))
                        model = MainModel(net_G=net_G_loaded)
                        model.load_state_dict(torch.load("UNET_with_Resnet18_pretrained_CELEBA/UnetTrainedByMe18000picturescelebADataset_using_pretrained_Resnet18_18000pictures.pth", map_location=device))  
 
            elif option2 == 'Unet CelebA dataset Resnet34':
                net_G_loaded = build_res34_unet(n_input=1, n_output=2, size=256)
                net_G_loaded.load_state_dict(torch.load("RESNET_34_pretrainedbymeCELEBA/Resnet-34-pretrainedbyme18000picturescelebADataset.pth", map_location=device))
                model = MainModel(net_G=net_G_loaded)
                model.load_state_dict(torch.load("UNET_with_Resnet34_pretrained_CELEBA/UnetTrainedByMe18000picturescelebADataset_using_pretrained_Resnet34_18000pictures.pth", map_location=device)) 

    if picture:
        uploaded_file = picture
        


    
#Add 'before' and 'after' columns
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_width, original_height = image.size
    col1, col2, col3 = st.columns( [1, 1, 1])
    with col1:
        st.markdown('<p style="text-align: left; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 25px;"">Uploaded picture</p>',unsafe_allow_html=True)
        image = np.array(image)
        width = 256
        height = 256
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        st.image(image, use_column_width = 'always')  

    with col2:
        st.markdown('<p style="text-align: left; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 25px;""> Black and white input</p>',unsafe_allow_html=True)
        # Open the image
        img = Image.open(uploaded_file).convert("RGB")
        # Convert it to an np array
        img = np.array(img)
        # Convert from RGB representation to LAB and convert it to tensor
        image_lab = rgb2lab(img/255)
        image_lab_resize = (image_lab + [0, 128, 128]) / [100, 255, 255]
        (H_orig,W_orig) = img.shape[:2]

        img_l = image_lab_resize[:,:,0]
        print(img_l.shape)
        width = 256
        height = 256
        dim = (width, height)
        resized = cv2.resize(img_l, (original_width, original_height), interpolation = cv2.INTER_AREA)
        st.image(resized,clamp = True, use_column_width = 'always')
        #img = Image.fromarray((resized).astype(np.uint8))
#        cv2.imwrite("black_and_white.png", resized)
        img = Image.fromarray((img_l * 255).astype(np.uint8))
        img = img.resize((original_width, original_height), Image.ANTIALIAS)



    if option == "Pix2PixModel" or  add_selectbox == "Take picture":
        with col3:

            st.markdown('<p style="text-align: left; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 25px;"">Colorized automatically</p>',unsafe_allow_html=True)
            with st.spinner('Wait for it...'):
                list = []
                list.append(uploaded_file)
                dataset = ColorizationDataset(list)
                training_dataloader = DataLoader(dataset, batch_size=1, num_workers = 2)
                iterator = iter(training_dataloader)
                data = next(iterator)
                model.net_G.eval()
                with torch.no_grad():
                    model.setup_input(data)
                    model.forward()
                model.net_G.train()
                fake_color = model.fake_color.detach()
                real_color = model.ab
                L = model.L
                fake_imgs = lab_to_rgb(L, fake_color)

                fake_imgs = np.squeeze(fake_imgs)
                img = Image.fromarray((fake_imgs * 255).astype(np.uint8))
                img = img.resize((original_width, original_height), Image.ANTIALIAS)
                img.save('colorized.png')
                st.image(img,use_column_width = 'always')

        if img is not None:
            with open("colorized.png", "rb") as file:
                btn = st.download_button(
                label="Download the colorized image",
                data=file,
                file_name="colorized.png",
                mime="image/png"
                )      
    elif option == "ECCVGenerator":
        with col3:
            st.markdown('<p style="text-align: left; font-family:Courier; color:#DDC5A2; font-family: Cooper Black; font-size: 25px;"">Colorized automatically</p>',unsafe_allow_html=True)
            list = []
            list.append(uploaded_file)
            dataset = ColorizationDataset2(list)
            training_dataloader = DataLoader(dataset, batch_size=1, num_workers = 2)
            iterator = iter(training_dataloader)
            data = next(iterator)
            model.eval()
            fake_color = model.forward(data['L'])
            fake_color = fake_color.detach()
            L = data['L']
            out_lab_orig = torch.cat((L, fake_color), dim=1)
            img = lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))
            #fake_imgs = labtorgb(L, fake_color)
            st.image(img, width = 256)




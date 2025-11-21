from PIL import Image
import torch
from torch import nn
import os
import numpy as np
import einops

from models.model import LFAN
from raw_process import RawProcess
from configs import config
import torchvision.transforms as transforms
    

class Pipeline(nn.Module):
    def __init__(self, device, data_path, data_name):
        super(Pipeline, self).__init__()
        self.device = device
        # self.video_path = video_path
        self.data_path = data_path
        self.data_name = data_name
        self.config = config
        self.raw_processor = RawProcess(device=self.device, data_path=data_path, data_name=data_name)
        self.model = LFAN(
            backbone_settings=self.config['backbone_settings'],
            modality=['video', "vggish", 'logmel'],
            example_length=300,
            kernel_size=5,
            tcn_channel=self.config['tcn']['channels'], 
            modal_dim=32, 
            num_heads=2,
            root_dir='pretrained_model', 
            device=self.device
        )
        self.model.init()
        self.model.to(self.device)
        self.model.eval()

    def forward(self, video_path):
        # video_path: path to the input video file
        # data_path: path to the directory where processed data will be saved
        # data_name: base name for the processed files

        processed_data = self.raw_processor.preprocess(video_path)
        
        vggish = processed_data
        video = self.load_video_data()
        
        length = min(video.shape[2], vggish.shape[1])
        
        vggish = einops.rearrange(torch.tensor(vggish), 't d -> 1 1 t d')
        # pad with last frame to make length 300
        if vggish.shape[2] < 300:
            pad_size = 300 - vggish.shape[2]
            pad_tensor = vggish[:, :, -1:, :].repeat(1, 1, pad_size, 1)
            vggish = torch.cat([vggish, pad_tensor], dim=2)
        elif vggish.shape[2] > 300:
            vggish = vggish[:, :, :300]
        
        video = einops.rearrange(video, 't c h w -> 1 t c h w')
        if video.shape[1] < 300:
            pad_size = 300 - video.shape[1]
            pad_tensor = video[:, -1:, :, :, :].repeat(1, pad_size, 1, 1, 1)
            video = torch.cat([video, pad_tensor], dim=1)
        elif video.shape[1] > 300:
            video = video[:, :300, :, :, :]
        
        X = {
            'video': video.to(self.device),
            'vggish': vggish.to(self.device),
        }
        
        output = self.model(X)
        output = output[0, :length].mean(dim=0)
        
        return output
    
    def load_video_data(self):
        path = os.path.join(self.data_path, self.data_name, 'frames')
        num_images = len(os.listdir(path))
        
        imgs=[]
        
        for i in range(num_images):
            jpg_filename = os.path.join(path, f'{i:05d}.jpg')
            img = Image.open(jpg_filename).convert("RGB")

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((40, 40)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            imgs.append(img_tensor)
        data = torch.cat(imgs, dim=0)
        return data

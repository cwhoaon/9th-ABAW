from __future__ import absolute_import
from __future__ import division

from torch.nn import init
import torch
import math
from torch import nn
from torch.nn import functional as F
import sys
from .av_crossatten import DCNLayer
from .layer import LSTM

class TAGF(nn.Module):
    def __init__(self):
        super(TAGF, self).__init__()

        self.coattn = DCNLayer(256, 128, 2, 0.6)
        
        self.video_temporal_encoder = LSTM(embed_size=256, dim=512, num_layers=1, dropout=0.1)
        self.audio_temporal_encoder = LSTM(embed_size=128, dim=256, num_layers=1, dropout=0.1)

        self.vregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

        self.Joint = LSTM(1024, 512, 2, dropout=0, residual_embeddings=True)

        self.aregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))
                                           
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)#추가
        self.init_weights()

    def init_weights(net, init_type='xavier', init_gain=1):

        if torch.cuda.is_available():
            net.cuda()

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.uniform_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>
    
    def forward(self, f1_norm, f2_norm):
        video = F.normalize(f1_norm, dim=-1)
        audio = F.normalize(f2_norm, dim=-1)

        atten_video, atten_audio = self.coattn(video, audio)

        # Step 1: 반복 단계 feature stack
        video_stack = torch.stack([video, atten_video[-2], atten_video[-1]], dim=2)  # [B, L, M, Dv]
        audio_stack = torch.stack([audio, atten_audio[-2], atten_audio[-1]], dim=2)  # [B, L, M, Da]

        # Step 2: reshape for temporal encoder
        B, L, M, Dv = video_stack.shape
        video_input = video_stack.view(B * L, M, Dv)  # [B*L, M, Dv]
        video_encoded = self.video_temporal_encoder(video_input)  # [B*L, M, Dv]

        # 🔁 NEW: Add scoring layer to get weights over M steps
        video_score = torch.mean(video_encoded, dim=-1)  # [B*L, M]
        video_weights = F.softmax(video_score / 0.1, dim=1).unsqueeze(-1)  # [B*L, M, 1]
        fused_video = torch.sum(video_input * video_weights, dim=1).view(B, L, Dv)

        # 동일하게 오디오도
        B, L, M, Da = audio_stack.shape
        audio_input = audio_stack.view(B * L, M, Da)
        audio_encoded = self.audio_temporal_encoder(audio_input)  # [B*L, M, Da]
        audio_score = torch.mean(audio_encoded, dim=-1)  # [B*L, M]
        audio_weights = F.softmax(audio_score / 0.1, dim=1).unsqueeze(-1)  # [B*L, M, 1]
        fused_audio = torch.sum(audio_input * audio_weights, dim=1).view(B, L, Da)

        audiovisualfeatures = torch.cat((fused_video, fused_audio), -1)
        return audiovisualfeatures

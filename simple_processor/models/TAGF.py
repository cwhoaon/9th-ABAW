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
    def __init__(self, reduce_frame_level_features=False, simple_gate=False):
        super(TAGF, self).__init__()
        self.reduce_frame_level_features = reduce_frame_level_features
        self.simple_gate = simple_gate

        self.coattn = DCNLayer(256, 128, 2, 0.6)
        
        if self.simple_gate:    
            self.video_gate_mlp = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )
            self.audio_gate_mlp = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
        else:
            self.video_temporal_encoder = LSTM(embed_size=256, dim=512, num_layers=1, dropout=0.1)
            self.audio_temporal_encoder = LSTM(embed_size=128, dim=256, num_layers=1, dropout=0.1)
                                           
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        if reduce_frame_level_features: 
            self.frame_attn = nn.Linear(384, 1)
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
    
    def forward(self, f1_norm, f2_norm, mask):
        video = F.normalize(f1_norm, dim=-1)
        audio = F.normalize(f2_norm, dim=-1)

        atten_video, atten_audio = self.coattn(video, audio)

        # Step 1: 반복 단계 feature stack
        video_stack = torch.stack([video, atten_video[-2], atten_video[-1]], dim=2)  # [B, L, M, Dv]
        audio_stack = torch.stack([audio, atten_audio[-2], atten_audio[-1]], dim=2)  # [B, L, M, Da]

        # Step 2: reshape for temporal encoder
        B, L, M, Dv = video_stack.shape
        B, L, M, Da = audio_stack.shape
        
        if self.simple_gate is False:
            video_input = video_stack.view(B * L, M, Dv)  # [B*L, M, Dv]
            video_encoded = self.video_temporal_encoder(video_input)  # [B*L, M, Dv]

            # NEW: Add scoring layer to get weights over M steps
            video_score = torch.mean(video_encoded, dim=-1)  # [B*L, M]
            video_weights = F.softmax(video_score / 0.1, dim=1).unsqueeze(-1)  # [B*L, M, 1]
            fused_video = torch.sum(video_input * video_weights, dim=1).view(B, L, Dv)

            audio_input = audio_stack.view(B * L, M, Da)
            audio_encoded = self.audio_temporal_encoder(audio_input)  # [B*L, M, Da]
            audio_score = torch.mean(audio_encoded, dim=-1)  # [B*L, M]
            audio_weights = F.softmax(audio_score / 0.1, dim=1).unsqueeze(-1)  # [B*L, M, 1]
            fused_audio = torch.sum(audio_input * audio_weights, dim=1).view(B, L, Da)
        else:
            v_flat = video_stack.reshape(B * L * M, Dv)
            v_scores = self.video_gate_mlp(v_flat)  # [B*L*M, 1]
            v_scores = v_scores.view(B * L, M)      # [B*L, M]
            v_alpha = F.softmax(v_scores, dim=1)    # [B*L, M]
            v_alpha = v_alpha.view(B, L, M, 1)      # [B, L, M, 1]
            fused_video = torch.sum(video_stack * v_alpha, dim=2)  # [B, L, Dv]
            
            a_flat = audio_stack.reshape(B * L * M, Da)
            a_scores = self.audio_gate_mlp(a_flat)  # [B*L*M, 1]
            a_scores = a_scores.view(B * L, M)      # [B*L  , M]
            a_alpha = F.softmax(a_scores, dim=1)    # [B*L, M]
            a_alpha = a_alpha.view(B, L, M, 1)      # [B, L, M, 1]
            fused_audio = torch.sum(audio_stack * a_alpha, dim=2)  # [B, L, Da]
        
        audiovisualfeatures = torch.cat((fused_video, fused_audio), -1)
        
        # reduce frame features to a single feature vector
        if self.reduce_frame_level_features:
            scores = self.frame_attn(audiovisualfeatures).squeeze(-1) # [B, L]
            scores = scores.masked_fill(mask == 0, float('-inf'))  # [B, L]
            weight = F.softmax(scores, dim=1).unsqueeze(-1)   # [B, L, 1]
            audiovisualfeatures = torch.sum(audiovisualfeatures * weight, dim=1)        # [B, D]
        
        return audiovisualfeatures

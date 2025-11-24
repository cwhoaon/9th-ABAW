from models.arcface_model import Backbone


import torch
from torch import nn

from torchvision import models

from torch.nn.init import xavier_uniform_, constant_
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            # nn.ReLU(True)
        )

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


class VGGish(VGG):
    def __init__(self):
        super().__init__(make_layers())

    def forward(self, x, fs=None):
        x = torch.tensor(x)[:, None, :, :].float()
        
        x = VGG.forward(self, x)
        return x


class VisualBackboneSmall(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 8,
        use_pretrained: bool = True,
        backbone_type: str = "resnet18",   # "resnet18", "mobilenet_v3_small", "efficientnet_b0"
        embedding_dim: int = 512,
        state_dict_path: str = "",
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.embedding_dim = embedding_dim

        if backbone_type == "resnet18":
            base = models.resnet18(pretrained=use_pretrained)
            in_feats = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        elif backbone_type == "mobilenet":
            base = models.mobilenet_v3_small(pretrained=use_pretrained)
            in_feats = base.classifier[-1].in_features
            base.classifier[-1] = nn.Identity()
            self.backbone = base

        elif backbone_type == "efficientnet":
            base = models.efficientnet_b0(pretrained=use_pretrained)
            in_feats = base.classifier[-1].in_features
            base.classifier[-1] = nn.Identity()
            self.backbone = base

        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        self.embedding = nn.Linear(in_feats, embedding_dim)
        self.logits = nn.Linear(embedding_dim, num_classes)

        if state_dict_path:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)


        xavier_uniform_(self.embedding.weight)
        constant_(self.embedding.bias, 0)

        xavier_uniform_(self.logits.weight)
        constant_(self.logits.bias, 0)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, embedding_dim]
        """
        feat = self.backbone(x)          # [B, in_feats]
        emb = self.embedding(feat)       # [B, embedding_dim]
        return emb

    def extract(self, x):
        return self.forward(x)


class VisualBackbone(nn.Module):
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_path="", mode="ir",
                 embedding_dim=512):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            state_dict = torch.load(state_dict_path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                        Dropout(0.4),
                                                        Flatten(),
                                                        Linear(embedding_dim * 5 * 5, embedding_dim),
                                                        BatchNorm1d(embedding_dim))

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

        from torch.nn.init import xavier_uniform_, constant_

        for m in self.backbone.output_layer.modules():
            if isinstance(m, nn.Linear):
                m.weight = xavier_uniform_(m.weight)
                m.bias = constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        self.logits.weight = xavier_uniform_(self.logits.weight)
        self.logits.bias = constant_(self.logits.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def extract(self, x):
        x = self.backbone(x)
        return x


class AudioBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = VGGish()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x, extract_vggish=False):
        x = self.backbone(x)

        return x
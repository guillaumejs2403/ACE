import torchvision.models as models
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class DenseNet121(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.feat_extract = models.densenet121(pretrained=pretrained)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024

    def forward(self, x):
        return self.feat_extract(x)


class DecisionDensenetModel(nn.Module):

    def __init__(self, num_classes=40, pretrained=False, query_label=-1):
        super().__init__()
        self.feat_extract = DenseNet121(pretrained=pretrained)
        self.classifier = nn.Linear(self.feat_extract.output_size, num_classes)
        self.query_label = query_label

    def forward(self, x, before_sigmoid=True):

        x = self.feat_extract(x)
        x = self.classifier(x)
        if not before_sigmoid:
            x = torch.sigmoid(x)
        return x[:, self.query_label]

import torch
import torchvision


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class DenseNet121(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.densenet121(pretrained=False)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024
    
    def forward(self, x):
        return self.feat_extract(x)


class DiVEDenseNet121(torch.nn.Module):
    def __init__(self, path_to_weights, query_label):
        '''
        :alpha: when alpha is equal to 0, we have no entanglement,
                when set to 1, it is equal to have query_label=entangled_label
        '''
        super().__init__()
        self.feat_extract = DenseNet121()
        self.classifier = torch.nn.Linear(self.feat_extract.output_size, 40)
        self.query_label = query_label

        # load the model from the checkpoint
        state_dict = torch.load(path_to_weights, map_location='cpu')
        self.feat_extract.load_state_dict(state_dict['feat_extract'])
        self.classifier.load_state_dict(state_dict['classifier'])

    def forward(self, x, get_other_attrs=False):
        x = self.feat_extract(x)
        x = self.classifier(x)

        if get_other_attrs:
            return x[:, self.query_label], x
        else:
            return x[:, self.query_label]

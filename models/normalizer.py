import torch

class Normalizer(torch.nn.Module):
    '''
    normalizing module. Useful for computing the gradient
    to a x image (x in [0, 1]) when using a classifier with
    different normalization inputs (i.e. f((x - mu) / sigma))
    '''
    def __init__(self, classifier,
                 mu=[0.485, 0.456, 0.406],
                 sigma=[0.229, 0.224, 0.225]):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor(mu).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma).view(1, -1, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.classifier(x)

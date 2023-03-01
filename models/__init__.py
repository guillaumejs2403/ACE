import torch

from torchvision import models

from models.normalizer import Normalizer
from models.dive.densenet import DiVEDenseNet121
from models.steex.DecisionDensenetModel import DecisionDensenetModel


def get_classifier(args):
    if args.dataset in ['CelebA', 'CelebAMV']:
        classifier = Normalizer(
            DiVEDenseNet121(args.classifier_path, args.label_query),
            [0.5] * 3, [0.5] * 3
        )

    elif args.dataset == 'CelebAHQ':
        assert args.label_query in [20, 31, 39], 'Query label MUST be 20 (Gender), 31 (Smile), or 39 (Gender) for CelebAHQ'
        ql = 0
        if args.label_query in [31, 39]:
            ql = 1 if args.label_query == 31 else 2
        classifier = DecisionDensenetModel(3, pretrained=False,
                                           query_label=ql)
        classifier.load_state_dict(torch.load(args.classifier_path, map_location='cpu')['model_state_dict'])
        classifier = Normalizer(
            classifier,
            [0.5] * 3, [0.5] * 3
        )

    elif args.dataset in ['BDDOIA', 'BDD100k']:
        classifier = DecisionDensenetModel(4, pretrained=False,
                                           query_label=args.label_query)
        classifier.load_state_dict(torch.load(args.classifier_path, map_location='cpu')['model_state_dict'])
        classifier = Normalizer(
            classifier,
            [0.5] * 3, [0.5] * 3
        )

    else:
        classifier = Normalizer(
            models.resnet50(pretrained=True)
        )

    return classifier

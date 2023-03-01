import torch


def accuracy(logits, label, topk=(1, 5), binary=False):
    '''
    Computes the topx accuracy between the logits and label.
    If set the binary flag to true, it will compute the top1 and the rest will return 1
    '''
    if binary:
        res = [((logits > 0).float() == label)]
        res += [torch.ones_like(res[0])] * (len(topk) - 1)
    else:
        maxk = max(topk)
        _, pred_k = torch.topk(logits, maxk, dim=1)
        correct_k = (pred_k == label.view(-1, 1))

        res = []
        for k in topk:
            res.append(correct_k[:, :k].sum(dim=1))

    return res


@torch.no_grad()
def get_prediction(classifier, img, binary):
    log = classifier(img)
    if binary:
        pred = (log > 0).float()
    else:
        pred = log.argmax(dim=1)

    return log, pred

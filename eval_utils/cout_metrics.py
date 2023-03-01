'''
COUT metric extratec from: https://github.com/khorrams/c3lt

We implemented the RGB version for the evaluation and
the binary version
'''


from tqdm import tqdm

from torch.autograd import Variable

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from pdb import set_trace

def gen_masks(inputs, targets, mode='abs'):
    """
        generates a difference masks give two images (inputs and targets).
    :param inputs:
    :param targets:
    :param mode:
    :return:
    """
    masks = targets - inputs
    masks = masks.view(inputs.size(0), 3, -1)

    if mode == 'abs':
        masks = masks.abs()
        masks = masks.sum(dim=1)
        # normalize 0 to 1
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    elif mode == "mse":
        masks = masks ** 2
        masks = masks.sum(dim=1)
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    else:
        raise ValueError("mode value is not valid!")

    return masks.view(inputs.size(0), 1, inputs.size(2), inputs.size(3))


@torch.no_grad()
def evaluate(label, target, classifier, loader, device, binary):
    """
        evaluates loss values and metrics.
    :param encoder:
    :param maps:
    :param generator:
    :param discriminator:
    :param classifier:
    :param dataloader:
    :param writer:
    :param epoch:
    :return:
    """
    # eval params
    cout_num_steps = 50

    # init scores
    cout = 0
    total_samples = 0
    plot_data = {'c_curve': [],
                 'c_prime_curve': []}

    with torch.no_grad():
        for i, (img, cf) in enumerate(tqdm(loader)):
            img = img.to(device, dtype=torch.float)
            cf = cf.to(device, dtype=torch.float)

            # calculate metrics
            cout_score, plot_info = calculate_cout(
                img,
                cf,
                gen_masks(img, cf, mode='abs'),
                classifier,
                label,
                target,
                max(1, (img.size(2) * img.size(3)) // cout_num_steps),
                binary
            )
            plot_data['c_curve'].append([d.cpu() for d in plot_info[0]])
            plot_data['c_prime_curve'].append([d.cpu() for d in plot_info[1]])
            cout += cout_score

            # update total sample number
            total_samples += img.shape[0]

    # process plot info
    curves = torch.zeros(2, len(plot_info[2]))
    for idx, curve_name in enumerate(['c_curve', 'c_prime_curve']):
        for data_points in plot_data[curve_name]:
            data_points = torch.cat([d.unsqueeze(dim=1) for d in data_points], dim=1)
            curves[idx, :] += data_points.sum(dim=0)
    curves /= total_samples
    cout /= total_samples
    print(f"\nEVAL [COUT: {cout:.4f}]")
    return cout, {'indexes': plot_info[2], 'c_curve': curves[0, :].numpy(), 'c_prime_curve': curves[1, :].numpy()}


@torch.no_grad()
def get_probs(label, target, img, model, binary):
    '''
        Computes the probabilities of the target/label classes
    '''
    if binary:
        # for the binary classification, the target is irrelevant since it is 1 - label
        output = model(img)
        pos = (label == 1).float()
        c_curve = torch.sigmoid(pos * output - (1 - pos) * output)
        c_prime_curve = 1 - c_curve
    else:
        output =  F.softmax(model(img))
        c_curve = output[:, label]
        c_prime_curve = output[:, target]

    return c_curve, c_prime_curve


@torch.no_grad()
def calculate_cout(imgs, cfs, masks, model, label, target, step, binary):
    """
        calculates the counterfactual transition (cout) score.
        Produce the results solely for correctly classified images
    :param imgs:
    :param cfs:
    :param masks:
    :param model:
    :param cls_1:
    :param cls_2:
    :param step:
    :return:
    """

    # The dimensions for the image
    img_size = imgs.shape[-2:]
    mask_size = masks.shape[-2:]

    # Compute the total number of pixels in a mask
    num_pixels = torch.prod(torch.tensor(masks.shape[1:])).item()
    l = torch.arange(imgs.shape[0])

    if binary:
        label = (model(imgs) > 0.0)

    # Initial values for the curves
    c_curve, c_prime_curve = get_probs(label, target, imgs, model, binary)
    c_curve = [c_curve]
    c_prime_curve = [c_prime_curve]
    index = [0.]

    # init upsampler
    up_sample = torch.nn.UpsamplingBilinear2d(size=img_size).to(imgs.device)

    # updating mask and the ordering
    cur_mask = torch.zeros((masks.shape[0], num_pixels)).to(imgs.device)
    elements = torch.argsort(masks.view(masks.shape[0], -1), dim=1, descending=True)

    for pixels in range(0, num_pixels, step):
        # Get the indices used in this iteration
        indices = elements[l, pixels:pixels + step].squeeze().view(imgs.shape[0], -1)

        # Set those indices to 1
        cur_mask[l, indices.permute(1, 0)] = 1
        up_masks = up_sample(cur_mask.view(-1, 1, *mask_size))

        # perturb the image using cur mask and calculate scores
        perturbed = phi(cfs, imgs, up_masks)
        score_c, score_c_prime = get_probs(label, target, perturbed, model, binary)

        # obtain the scores
        c_curve.append(score_c)
        c_prime_curve.append(score_c_prime)
        index.append((pixels + step) / num_pixels)

    auc_c, auc_c_prime = auc(c_curve), auc(c_prime_curve)
    auc_c *= step / (mask_size[0] * mask_size[1])
    auc_c_prime *= step / (mask_size[0] * mask_size[1])
    cout = auc_c_prime.sum().item() - auc_c.sum().item()

    return cout, (c_curve, c_prime_curve, index)


def phi(img, baseline, mask):
    """
        composes an image from img and baseline according to the mask values.
    :param img:
    :param baseline:
    :param mask:
    :return:
    """
    return img.mul(mask) + baseline.mul(1-mask)


def auc(curve):
    """
        calculates the area under the curve
    :param curve:
    :return:
    """
    return curve[0]/2 + sum(curve[1:-1]) + curve[-1]/2
# =======================================================
# =======================================================
# Naive and direct ablation
# =======================================================
# =======================================================


@torch.enable_grad()
def get_naive_class_grad(x, y, classifier,
                         s, use_logits):

    kwargs = {}
    
    x_in = x.detach().requires_grad_(True)
    logits = classifier(x_in, **kwargs)

    y = y.to(logits.device).float()
    # Select the target logits,
    # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
    # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
    selected = y * logits - (1 - y) * logits
    if use_logits:
        selected = -selected
    else:
        selected = -F.logsigmoid(selected)

    selected = selected * s
    grads = torch.autograd.grad(selected.sum(), x_in)[0]

    return grads


def get_naive_clean_fn(use_sampling=False,
                       sampling_scale=1.):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.no_grad()
    def p_sample_loop(
        diffusion,
        model,
        shape,
        num_timesteps,
        img,
        t,
        z_t=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        class_grad=None,
        x_t_sampling=True,
        is_x_t_sampling=False,
        guided_iterations=9999999,
    ):

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            acp = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0
            
            if (class_grad is not None) and (guided_iterations > jdx):
                grads = grads + class_grad / acp

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                scale = sampling_scale if is_x_t_sampling else 1
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img) * scale
                )

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


def get_direct_clean(use_sampling=False,
                     sampling_scale=1.):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.no_grad()
    def p_sample_loop(
        diffusion,
        model,
        shape,
        num_timesteps,
        img,
        t,
        z_t=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        class_grad_fn=None,
        class_grad_kwargs=None,
        dist_grad_fn=None,
        dist_grad_kargs=None,
        x_t_sampling=True,
        is_x_t_sampling=False,
        guided_iterations=9999999,
    ):

        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)
            z_t_steps.append(z_t.detach())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            acp = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0
            
            if (class_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + class_grad_fn(x=z_t,
                                              t=diffusion._scale_timesteps(t),
                                              **class_grad_kwargs)

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x=img,
                                             x_t=None,
                                             acp=1,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                scale = sampling_scale if is_x_t_sampling else 1
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img) * scale
                )

        return z_t, None, z_t_steps

    return p_sample_loop
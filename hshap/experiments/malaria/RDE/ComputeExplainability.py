import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def generate_explainability_map(
    x, forward_model, num_iter, step_size, batch_size, l1_lambda, device
):
    # I just pack everthing into this function so that I can just import it and call it from somewhere else

    # x is expected to be given as single image with color channels in the first component
    # x can be a tensor or a numpy array, works both way i think
    num_channel, img_width, img_height = x.shape

    # Initializing s as optimizable variable with adam optimizer
    s = torch.zeros(1, img_width, img_height, dtype=torch.float32, device=device)
    s = torch.autograd.Variable(s, requires_grad=True)
    optimizer = torch.optim.Adam([s], lr=step_size)

    # generating output from forward pass and extracting highest prediction component
    # in the paper in section 5 it is described how only the distortion in this component is used
    x_input = torch.Tensor(x[None, :, :, :]).to(device)
    x_out = forward_model(x_input.clone()).detach()
    highest_dim = int(np.argmax(x_out.cpu().numpy(), axis=1))

    # using list to log the loss values
    loss_log = []

    for i in range(num_iter // batch_size):
        # generating noise
        n = torch.randn(
            (batch_size, *x_input.shape[1:]), dtype=torch.float32, device=device
        )

        # input for the changed model, broadcasting handels the batch dimension
        data_input = (x_input - n) * s + n

        # for the l1-regularization, I use the mean, adjust l1_lambda if you want to use sum
        out = forward_model(data_input)
        loss = 0.5 * torch.mean(
            (out[:, highest_dim] - x_out[:, highest_dim]) ** 2
        ) + l1_lambda * torch.mean(torch.abs(s))

        loss_log.append(loss.data.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ensuring the [0,1]-constraint
        # this += thing was a weird hack to not lose Variable status of s for pytorch. s.data().clamp_ would probably
        # also work... but ya know... never touch a running system
        with torch.no_grad():
            s += s.clamp_(0, 1) - s

    return s.detach().cpu().numpy(), loss_log
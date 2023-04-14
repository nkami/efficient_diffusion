import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from torchvision.datasets import LFWPeople, ImageFolder, DatasetFolder
import lpips
import sys


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def plot_p2_loss_params():
    steps = 1000
    k = 1
    linear_betas = np.linspace(1e-4, 0.02, num=steps)
    cosine_betas = cosine_beta_schedule(steps).numpy()
    linear_alphas = [np.prod(1 - linear_betas[:t + 1]) for t in range(steps)]
    cosine_alphas = [np.prod(1 - cosine_betas[:t + 1]) for t in range(steps)]
    linear_lambdas = [((1 - linear_betas[t]) * (1 - linear_alphas[t])) / linear_betas[t] for t in range(steps)]
    cosine_lambdas = [((1 - cosine_betas[t]) * (1 - cosine_alphas[t])) / cosine_betas[t] for t in range(steps)]

    plt.plot([t for t in range(steps)], [1 / linear_lambdas[t] for t in range(steps)])
    plt.plot([t for t in range(steps)], [1 / cosine_lambdas[t] for t in range(steps)])
    plt.legend(['Linear Schedule', 'Cosine Schedule'])
    plt.title('Original Coefficient Value')
    plt.xlabel('t')
    plt.ylabel(r'$\frac{1}{\lambda_{t}}$')
    plt.yscale('log')
    plt.xticks([100 * i for i in range(int(steps / 100) + 1)])
    plt.savefig('./loss_analysis_results/orig_coef_val.png')
    plt.close()

    linear_snrs = [linear_alphas[t] / (1 - linear_alphas[t]) for t in range(steps)]
    cosine_snrs = [cosine_alphas[t] / (1 - cosine_alphas[t]) for t in range(steps)]
    # plt.plot([t for t in range(steps)], [linear_snrs[t] for t in range(steps)])
    # plt.plot([t for t in range(steps)], [cosine_snrs[t] for t in range(steps)])
    # plt.legend(['Linear Schedule', 'Cosine Schedule'])
    # plt.xlabel('t')
    # plt.ylabel(r'$SNR$')
    # plt.yscale('log')
    # plt.xticks([100 * i for i in range(int(steps / 100) + 1)])
    # plt.show()

    lambda_y2 = [linear_lambdas[t] / (k + linear_snrs[t]) ** 2 for t in range(steps)]
    lambda_y1 = [linear_lambdas[t] / (k + linear_snrs[t]) ** 1. for t in range(steps)]
    lambda_y0_5 = [linear_lambdas[t] / (k + linear_snrs[t]) ** 0.5 for t in range(steps)]
    plt.plot([t for t in range(steps)], [lambda_y2[t] / linear_lambdas[t] for t in range(steps)])
    plt.plot([t for t in range(steps)], [lambda_y1[t] / linear_lambdas[t] for t in range(steps)])
    plt.plot([t for t in range(steps)], [lambda_y0_5[t] / linear_lambdas[t] for t in range(steps)])
    plt.legend([r'$\gamma=2$', r'$\gamma=1$', r'$\gamma=0.5$'])
    plt.title('Total Weights Of Diffusion Steps')
    plt.xlabel('t')
    plt.ylabel(r"$\frac{\lambda'_{t}}{\lambda_{t}}$")
    plt.xticks([100 * i for i in range(int(steps / 100) + 1)])
    plt.savefig('./loss_analysis_results/total_weight_steps.png')
    plt.close()


def plot_snr_issues():
    steps = 1000
    betas = np.linspace(1e-4, 0.02, num=steps)
    linear_alphas = np.cumprod(1 - betas)

    cat2_256_256 = (plt.imread('./data/cat_imgs/cat2_256_256.jpg').astype(np.float32) / 255) * 2 - 1
    cat3_256_256 = plt.imread('./data/cat_imgs/cat3_256_256.png').astype(np.float32) * 2 - 1
    cat_64_64 = plt.imread('./data/cat_imgs/cat_64_64.png').astype(np.float32) * 2 - 1
    cat_256_256 = plt.imread('./data/cat_imgs/cat_256_256.png').astype(np.float32) * 2 - 1

    eval_steps = [0, 49, 99, 149, 199, 249, 299, 349, 399]
    imgs = [cat_64_64, cat_256_256, cat2_256_256, cat3_256_256]
    fig, axs = plt.subplots(len(imgs), len(eval_steps))
    for img_idx, img in enumerate(imgs):
        for t_idx, t in enumerate(eval_steps):
            noise = np.random.normal(0, 1, size=img.shape).astype(np.float32)
            cur_img = (linear_alphas[t] ** 0.5) * img + ((1 - linear_alphas[t]) ** 0.5) * noise
            tmp_img = cur_img
            tmp_img = tmp_img - np.min(tmp_img)
            tmp_img = tmp_img / np.amax(tmp_img)
            if img_idx == 0:
                axs[img_idx, t_idx].set_title(f't={t + 1}')
            axs[img_idx, t_idx].axis('off')
            axs[img_idx, t_idx].imshow(tmp_img)
    plt.savefig('./loss_analysis_results/snr_issues.png')
    plt.close()


def plot_image_dependent_lpips(starting_img=0, num_steps=500, eval_every_steps=10):
    loss_fn_alex = lpips.LPIPS(net='alex')
    ds = ImageFolder('./data/128x128_flowers/class_folder')
    steps = 1000
    betas = np.linspace(1e-4, 0.02, num=steps)
    linear_alphas = np.cumprod(1 - betas)
    eval_steps = [i * eval_every_steps for i in range(int(num_steps / eval_every_steps) - 1)]
    eval_steps[-1] -= 1
    points = [[] for _ in range(len(eval_steps))]
    samples, i = 200, 0
    noise = np.random.normal(0, 1, size=np.array(ds[0][0]).shape).astype(np.float32)
    for i in range(samples):
        img, _ = ds[starting_img + i]
        img = np.array(img)
        normalized_img = (np.array(img) / 255) - 1
        for t_idx, t in enumerate(eval_steps):
            normalized_noisy_img = (linear_alphas[t] ** 0.5) * normalized_img + ((1 - linear_alphas[t]) ** 0.5) * noise
            a = torch.transpose(torch.unsqueeze(torch.from_numpy(normalized_img).float(), dim=0), dim0=1, dim1=3)
            b = torch.transpose(torch.unsqueeze(torch.from_numpy(normalized_noisy_img).float(), dim=0), dim0=1, dim1=3)
            cur_lpips = loss_fn_alex(a, b).detach().item()
            points[t_idx].append(cur_lpips)
    plt.figure(figsize=(16, 6))
    plt.boxplot(points, showfliers=False)
    plt.title('Image Dependent LPIPS Distances')
    plt.xlabel(r'$\frac{t}{10}$')
    plt.ylabel('LPIPS($x_{0}, x_{t}$)')
    plt.savefig('./loss_analysis_results/im_dependent_lpips.png', dpi=200)
    plt.close()


def plot_noise_dependent_lpips(starting_img=0, num_steps=500, eval_every_steps=10):
    loss_fn_alex = lpips.LPIPS(net='alex')
    ds = ImageFolder('./data/128x128_flowers/class_folder')
    steps = 1000
    betas = np.linspace(1e-4, 0.02, num=steps)
    linear_alphas = np.cumprod(1 - betas)
    eval_steps = [i * eval_every_steps for i in range(int(num_steps / eval_every_steps) - 1)]
    eval_steps[-1] -= 1
    points = [[] for _ in range(len(eval_steps))]
    samples, i = 200, 0
    img, _ = ds[starting_img]
    normalized_img = (np.array(img) / 255) - 1
    for i in range(samples):
        for t_idx, t in enumerate(eval_steps):
            noise = np.random.normal(0, 1, size=normalized_img.shape).astype(np.float32)
            normalized_noisy_img = (linear_alphas[t] ** 0.5) * normalized_img + ((1 - linear_alphas[t]) ** 0.5) * noise

            a = torch.transpose(torch.unsqueeze(torch.from_numpy(normalized_img).float(), dim=0), dim0=1, dim1=3)
            b = torch.transpose(torch.unsqueeze(torch.from_numpy(normalized_noisy_img).float(), dim=0), dim0=1, dim1=3)
            cur_lpips = loss_fn_alex(a, b).detach().item()
            points[t_idx].append(cur_lpips)
    plt.figure(figsize=(16, 6))
    plt.boxplot(points, showfliers=False)
    plt.title('Noise Dependent LPIPS Distances')
    plt.xlabel(r'$\frac{t}{10}$')
    plt.ylabel('LPIPS($x_{0}, x_{t}$)')
    plt.savefig('./loss_analysis_results/noise_dependent_lpips.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    plot_p2_loss_params()
    print('done p2_loss_params')
    sys.stdout.flush()
    plot_snr_issues()
    print('done snr_issues')
    sys.stdout.flush()
    plot_image_dependent_lpips()
    print('done image_dependent_lpips')
    sys.stdout.flush()
    plot_noise_dependent_lpips()
    print('done noise_dependent_lpips')
    sys.stdout.flush()

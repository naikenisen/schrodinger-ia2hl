import os
import sys
import copy
import time
import random
import datetime
from itertools import repeat

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

from pytorch_lightning.loggers import CSVLogger as _CSVLogger

import config as cfg
from models.unet import UNetModel
from dataloader import dataloader

cmp = lambda x: transforms.Compose([*x])


# ============================================================================
#  REPEATER (boucle infinie sur un DataLoader)
# ============================================================================

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


# ============================================================================
#  LOGGER
# ============================================================================

class Logger:
    def log_metrics(self, metric_dict, step=None, save=False):
        pass

    def log_hparams(self, hparams_dict):
        pass


class CSVLogger(Logger):
    def __init__(self, directory='./', name='logs', save_stride=1):
        self.logger = _CSVLogger(directory, name=name)
        self.count = 0
        self.stride = save_stride

    def log_metrics(self, metrics, step=None, save=False):
        self.count += 1
        self.logger.log_metrics(metrics, step=step)
        if self.count % self.stride == 0:
            self.logger.save()
            self.logger.metrics = []
        if self.count > self.stride * 10:
            self.count = 0
        if save:
            self.logger.save()

    def log_hparams(self, hparams_dict):
        self.logger.log_hyperparams(hparams_dict)
        self.logger.save()


# ============================================================================
#  PLOTTER
# ============================================================================

def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]
    frames[0].save(
        os.path.join(output_directory, f'{gif_name}.gif'),
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )


class ImPlotter:
    def __init__(self, im_dir='./im', gif_dir='./gif', plot_level=3):
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)
        self.im_dir = im_dir
        self.gif_dir = gif_dir
        self.num_plots = 100
        self.num_digits = 20
        self.plot_level = plot_level

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        if self.plot_level > 0:
            x_tot_plot = x_tot_plot[:, :self.num_plots]
            name = '{0}_{1}_{2}'.format(forward_or_backward, n, i)
            im_dir = os.path.join(self.im_dir, name)

            if not os.path.isdir(im_dir):
                os.mkdir(im_dir)

            if self.plot_level > 0:
                plt.clf()
                filename_grid_png = os.path.join(im_dir, 'im_grid_first.png')
                vutils.save_image(initial_sample, filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir, 'im_grid_final.png')
                vutils.save_image(x_tot_plot[-1], filename_grid_png, nrow=10)

            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []
                num_steps, num_particles, channels, H, W = x_tot_plot.shape
                plot_steps = np.linspace(0, num_steps - 1, self.num_plots, dtype=int)

                for k in plot_steps:
                    filename_grid_png = os.path.join(im_dir, 'im_grid_{0}.png'.format(k))
                    plot_paths.append(filename_grid_png)
                    vutils.save_image(x_tot_plot[k], filename_grid_png, nrow=10)

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)


# ============================================================================
#  EMA HELPER
# ============================================================================

class EMAHelper:
    def __init__(self, mu=0.999, device="cpu"):
        self.mu = mu
        self.shadow = {}
        self.device = device

    def register(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            inner_module = module.module
            locs = inner_module.locals
            module_copy = type(inner_module)(*locs).to(self.device)
            module_copy.load_state_dict(inner_module.state_dict())
            if isinstance(module, nn.DataParallel):
                module_copy = nn.DataParallel(module_copy)
        else:
            locs = module.locals
            module_copy = type(module)(*locs).to(self.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ============================================================================
#  LANGEVIN DYNAMICS
# ============================================================================

def grad_gauss(x, m, var):
    return -(x - m) / var


def ornstein_ulhenbeck(x, gradx, gamma):
    return x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)


class Langevin(torch.nn.Module):

    def __init__(self, num_steps, shape, gammas, time_sampler, device=None,
                 mean_final=torch.tensor([0., 0.]), var_final=torch.tensor([.5, .5]),
                 mean_match=True):
        super().__init__()
        self.mean_match = mean_match
        self.mean_final = mean_final
        self.var_final = var_final
        self.num_steps = num_steps
        self.d = shape
        self.gammas = gammas.float()

        gammas_vec = torch.ones(self.num_steps, *self.d, device=device)
        for k in range(num_steps):
            gammas_vec[k] = gammas[k].float()
        self.gammas_vec = gammas_vec

        self.device = device if device is not None else gammas.device
        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()
        self.time_sampler = time_sampler

    def record_init_langevin(self, init_samples):
        mean_final = self.mean_final
        var_final = self.var_final
        x = init_samples
        N = x.shape[0]
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps_expanded = time

        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        num_iter = self.num_steps

        for k in range(num_iter):
            gamma = self.gammas[k]
            gradx = grad_gauss(x, mean_final, var_final)
            t_old = x + gamma * gradx
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma) * z
            gradx = grad_gauss(x, mean_final, var_final)
            t_new = x + gamma * gradx
            x_tot[:, k, :] = x
            out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples, t_batch=None, ipf_it=0, sample=False):
        mean_final = self.mean_final
        var_final = self.var_final
        x = init_samples
        N = x.shape[0]
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps = time
        steps_expanded = steps

        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        num_iter = self.num_steps

        if self.mean_match:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = net(x, steps[:, k, :])
                if sample and (k == num_iter - 1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                t_new = net(x, steps[:, k, :])
                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)
        else:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = x + net(x, steps[:, k, :])
                if sample and (k == num_iter - 1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                t_new = x + net(x, steps[:, k, :])
                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)

        return x_tot, out, steps_expanded

    def forward(self, net, init_samples, t_batch, ipf_it):
        return self.record_langevin_seq(net, init_samples, t_batch, ipf_it)


# ============================================================================
#  CACHE LOADER
# ============================================================================

class CacheLoader(Dataset):
    def __init__(self, fb, sample_net, dataloader_b, num_batches, langevin, n,
                 mean, std, batch_size, device='cpu',
                 dataloader_f=None, transfer=False):
        super().__init__()
        start = time.time()
        shape = langevin.d
        num_steps = langevin.num_steps

        # Stockage du cache sur CPU pour économiser la VRAM
        self.data = torch.zeros(
            (num_batches, batch_size * num_steps, 2, *shape))  # CPU
        self.steps_data = torch.zeros(
            (num_batches, batch_size * num_steps, 1))           # CPU

        with torch.no_grad():
            for b in range(num_batches):
                if fb == 'b':
                    batch = next(dataloader_b)[0]
                    batch = batch.to(device)
                elif fb == 'f' and transfer:
                    batch = next(dataloader_f)[0]
                    batch = batch.to(device)
                else:
                    batch = mean + std * torch.randn((batch_size, *shape), device=device)

                if (n == 1) and (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(batch)
                else:
                    x, out, steps_expanded = langevin.record_langevin_seq(sample_net, batch, ipf_it=n)


                # On transfère sur CPU immédiatement pour libérer la mémoire GPU
                x = x.cpu().unsqueeze(2)
                out = out.cpu().unsqueeze(2)
                batch_data = torch.cat((x, out), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data

                flat_steps = steps_expanded.cpu().flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

                # Libération explicite des intermédiaires GPU
                del x, out, batch_data, flat_data, flat_steps, batch
                if device != 'cpu':
                    torch.cuda.empty_cache()

        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

        stop = time.time()
        print('Cache size: {0}'.format(self.data.shape))
        print("Load time: {0}".format(stop - start))

    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]
        return x, out, steps

    def __len__(self):
        return self.data.shape[0]


# ============================================================================
#  CONFIG GETTERS (modèle, optimiseur, données, plotter, logger)
# ============================================================================

def get_models():
    image_size = cfg.IMAGE_SIZE
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in cfg.ATTENTION_RESOLUTIONS.split(","):
        attention_ds.append(image_size // int(res))

    kwargs = {
        "in_channels": cfg.CHANNELS,
        "model_channels": cfg.NUM_CHANNELS,
        "out_channels": cfg.CHANNELS,
        "num_res_blocks": cfg.NUM_RES_BLOCKS,
        "attention_resolutions": tuple(attention_ds),
        "dropout": cfg.DROPOUT,
        "channel_mult": channel_mult,
        "num_classes": None,
        "use_checkpoint": cfg.USE_CHECKPOINT,
        "num_heads": cfg.NUM_HEADS,
        "num_heads_upsample": cfg.NUM_HEADS_UPSAMPLE,
        "use_scale_shift_norm": cfg.USE_SCALE_SHIFT_NORM,
    }
    net_f, net_b = UNetModel(**kwargs), UNetModel(**kwargs)
    return net_f, net_b


def get_optimizers(net_f, net_b, lr):
    return (torch.optim.Adam(net_f.parameters(), lr=lr),
            torch.optim.Adam(net_b.parameters(), lr=lr))


def get_datasets():
    train_transform = [
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
    if cfg.RANDOM_FLIP:
        train_transform.insert(2, transforms.RandomHorizontalFlip())

    root = os.path.join(cfg.DATA_DIR, 'dataset_v2')

    init_ds = dataloader(root, image_size=cfg.IMAGE_SIZE,
                       domain='HES', transform=cmp(train_transform))
    final_ds = dataloader(root, image_size=cfg.IMAGE_SIZE,
                        domain='CD30', transform=cmp(train_transform))
    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1. * 10 ** 3)

    return init_ds, final_ds, mean_final, var_final


def get_plotter():
    return ImPlotter(plot_level=cfg.PLOT_LEVEL)


def get_logger(name='logs'):
    if cfg.LOGGER == 'CSV':
        return CSVLogger(directory=cfg.CSV_LOG_DIR, name=name)
    return Logger()


# ============================================================================
#  IPF BASE
# ============================================================================

class IPFBase(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.accelerator = Accelerator(mixed_precision="no", cpu=(cfg.DEVICE == 'cpu'))
        self.device = self.accelerator.device

        # training params
        self.n_ipf = cfg.N_IPF
        self.num_steps = cfg.NUM_STEPS
        self.batch_size = cfg.BATCH_SIZE
        self.num_iter = cfg.NUM_ITER
        self.grad_clipping = cfg.GRAD_CLIPPING
        self.fast_sampling = cfg.FAST_SAMPLING
        self.lr = cfg.LR

        n = self.num_steps // 2
        if cfg.GAMMA_SPACE == 'linspace':
            gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        elif cfg.GAMMA_SPACE == 'geomspace':
            gamma_half = np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        gammas = torch.tensor(gammas).to(self.device)
        self.T = torch.sum(gammas)

        # models
        self.build_models()
        self.build_ema()

        # optimizers
        self.build_optimizers()

        # loggers
        self.logger = get_logger()
        self.save_logger = get_logger('plot_logs')

        # data
        self.build_dataloaders()

        # langevin
        if cfg.WEIGHT_DISTRIB:
            alpha = cfg.WEIGHT_DISTRIB_ALPHA
            prob_vec = (1 + alpha) * torch.sum(gammas) - torch.cumsum(gammas, 0)
        else:
            prob_vec = gammas * 0 + 1
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)

        batch = next(self.save_init_dl)[0]
        shape = batch[0].shape
        self.shape = shape
        self.langevin = Langevin(self.num_steps, shape, gammas,
                                 time_sampler, device=self.device,
                                 mean_final=self.mean_final, var_final=self.var_final,
                                 mean_match=cfg.MEAN_MATCH)

        # checkpoint
        date = str(datetime.datetime.now())[0:10]
        self.name_all = date

        self.checkpoint_run = cfg.CHECKPOINT_RUN
        if cfg.CHECKPOINT_RUN:
            self.checkpoint_it = cfg.CHECKPOINT_IT
            self.checkpoint_pass = cfg.CHECKPOINT_PASS
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = 'b'

        self.plotter = get_plotter()

        if self.accelerator.process_index == 0:
            os.makedirs('./im', exist_ok=True)
            os.makedirs('./gif', exist_ok=True)
            os.makedirs('./checkpoints', exist_ok=True)

        self.stride = cfg.GIF_STRIDE
        self.stride_log = cfg.LOG_STRIDE

    def build_models(self, forward_or_backward=None):
        net_f, net_b = get_models()

        if cfg.CHECKPOINT_RUN:
            if cfg.SAMPLE_CHECKPOINT_F:
                net_f.load_state_dict(torch.load(cfg.SAMPLE_CHECKPOINT_F))
            if cfg.SAMPLE_CHECKPOINT_B:
                net_b.load_state_dict(torch.load(cfg.SAMPLE_CHECKPOINT_B))

        if cfg.DATAPARALLEL:
            net_f = torch.nn.DataParallel(net_f)
            net_b = torch.nn.DataParallel(net_b)

        if forward_or_backward is None:
            net_f = net_f.to(self.device)
            net_b = net_b.to(self.device)
            self.net = torch.nn.ModuleDict({'f': net_f, 'b': net_b})
        if forward_or_backward == 'f':
            net_f = net_f.to(self.device)
            self.net.update({'f': net_f})
        if forward_or_backward == 'b':
            net_b = net_b.to(self.device)
            self.net.update({'b': net_b})

    def accelerate(self, forward_or_backward):
        (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(
            self.net[forward_or_backward], self.optimizer[forward_or_backward])

    def update_ema(self, forward_or_backward):
        if cfg.EMA:
            self.ema_helpers[forward_or_backward] = EMAHelper(
                mu=cfg.EMA_RATE, device=self.device)
            self.ema_helpers[forward_or_backward].register(
                self.net[forward_or_backward])

    def build_ema(self):
        if cfg.EMA:
            self.ema_helpers = {}
            self.update_ema('f')
            self.update_ema('b')

            if cfg.CHECKPOINT_RUN:
                sample_net_f, sample_net_b = get_models()

                if cfg.SAMPLE_CHECKPOINT_F:
                    sample_net_f.load_state_dict(
                        torch.load(cfg.SAMPLE_CHECKPOINT_F))
                    if cfg.DATAPARALLEL:
                        sample_net_f = torch.nn.DataParallel(sample_net_f)
                    sample_net_f = sample_net_f.to(self.device)
                    self.ema_helpers['f'].register(sample_net_f)
                if cfg.SAMPLE_CHECKPOINT_B:
                    sample_net_b.load_state_dict(
                        torch.load(cfg.SAMPLE_CHECKPOINT_B))
                    if cfg.DATAPARALLEL:
                        sample_net_b = torch.nn.DataParallel(sample_net_b)
                    sample_net_b = sample_net_b.to(self.device)
                    self.ema_helpers['b'].register(sample_net_b)

    def build_optimizers(self):
        optimizer_f, optimizer_b = get_optimizers(
            self.net['f'], self.net['b'], self.lr)
        self.optimizer = {'f': optimizer_f, 'b': optimizer_b}

    def build_dataloaders(self):
        init_ds, final_ds, mean_final, var_final = get_datasets()

        self.mean_final = mean_final.to(self.device)
        self.var_final = var_final.to(self.device)
        self.std_final = torch.sqrt(var_final).to(self.device)

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index)

        self.kwargs = {"num_workers": cfg.NUM_WORKERS,
                       "pin_memory": cfg.PIN_MEMORY,
                       "worker_init_fn": worker_init_fn,
                       "drop_last": True}

        self.save_init_dl = DataLoader(
            init_ds, batch_size=cfg.PLOT_NPAR, shuffle=True, **self.kwargs)
        self.cache_init_dl = DataLoader(
            init_ds, batch_size=cfg.CACHE_NPAR, shuffle=True, **self.kwargs)
        (self.cache_init_dl, self.save_init_dl) = self.accelerator.prepare(
            self.cache_init_dl, self.save_init_dl)
        self.cache_init_dl = repeater(self.cache_init_dl)
        self.save_init_dl = repeater(self.save_init_dl)

        if cfg.TRANSFER:
            self.save_final_dl = DataLoader(
                final_ds, batch_size=cfg.PLOT_NPAR, shuffle=True, **self.kwargs)
            self.cache_final_dl = DataLoader(
                final_ds, batch_size=cfg.CACHE_NPAR, shuffle=True, **self.kwargs)
            (self.cache_final_dl, self.save_final_dl) = self.accelerator.prepare(
                self.cache_final_dl, self.save_final_dl)
            self.cache_final_dl = repeater(self.cache_final_dl)
            self.save_final_dl = repeater(self.save_final_dl)
        else:
            self.cache_final_dl = None
            self.save_final_dl = None

    def new_cacheloader(self, forward_or_backward, n, use_ema=True):
        sample_direction = 'f' if forward_or_backward == 'b' else 'b'
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(
                self.net[sample_direction])
        else:
            sample_net = self.net[sample_direction]

        if forward_or_backward == 'b':
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader('b', sample_net, self.cache_init_dl,
                                 cfg.NUM_CACHE_BATCHES, self.langevin, n,
                                 mean=None, std=None,
                                 batch_size=cfg.CACHE_NPAR,
                                 device=self.device,
                                 dataloader_f=self.cache_final_dl,
                                 transfer=cfg.TRANSFER)
        else:
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader('f', sample_net, None,
                                 cfg.NUM_CACHE_BATCHES, self.langevin, n,
                                 mean=self.mean_final, std=self.std_final,
                                 batch_size=cfg.CACHE_NPAR,
                                 device=self.device,
                                 dataloader_f=self.cache_final_dl,
                                 transfer=cfg.TRANSFER)

        new_dl = DataLoader(new_dl, batch_size=self.batch_size)
        new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)
        return new_dl

    def train(self):
        pass

    def save_step(self, i, n, fb):
        if self.accelerator.is_local_main_process:
            if ((i % self.stride == 0) or (i % self.stride == 1)) and (i > 0):

                if cfg.EMA:
                    sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
                else:
                    sample_net = self.net[fb]

                name_net = 'net_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                name_net_ckpt = './checkpoints/' + name_net

                if cfg.DATAPARALLEL:
                    torch.save(self.net[fb].module.state_dict(), name_net_ckpt)
                else:
                    torch.save(self.net[fb].state_dict(), name_net_ckpt)

                if cfg.EMA:
                    name_net = 'sample_net_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                    name_net_ckpt = './checkpoints/' + name_net
                    if cfg.DATAPARALLEL:
                        torch.save(sample_net.module.state_dict(), name_net_ckpt)
                    else:
                        torch.save(sample_net.state_dict(), name_net_ckpt)

                with torch.no_grad():
                    self.set_seed(seed=0 + self.accelerator.process_index)
                    if fb == 'f':
                        batch = next(self.save_init_dl)[0]
                        batch = batch.to(self.device)
                    elif cfg.TRANSFER:
                        batch = next(self.save_final_dl)[0]
                        batch = batch.to(self.device)
                    else:
                        batch = self.mean_final + self.std_final * \
                            torch.randn((cfg.PLOT_NPAR, *self.shape), device=self.device)

                    x_tot, out, steps_expanded = self.langevin.record_langevin_seq(
                        sample_net, batch, ipf_it=n, sample=True)
                    shape_len = len(x_tot.shape)
                    x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
                    x_tot_plot = x_tot.detach()

                init_x = batch.detach().cpu().numpy()
                final_x = x_tot_plot[-1].detach().cpu().numpy()
                std_final = np.std(final_x)
                std_init = np.std(init_x)
                mean_final = np.mean(final_x)
                mean_init = np.mean(init_x)

                print('Initial variance: ' + str(std_init ** 2))
                print('Final variance: ' + str(std_final ** 2))

                self.save_logger.log_metrics({
                    'FB': fb,
                    'init_var': std_init ** 2, 'final_var': std_final ** 2,
                    'mean_init': mean_init, 'mean_final': mean_final,
                    'T': self.T,
                })

                self.plotter(batch, x_tot_plot, i, n, fb)

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        torch.cuda.empty_cache()


# ============================================================================
#  IPF SEQUENTIAL (entraînement)
# ============================================================================

class IPFSequential(IPFBase):

    def __init__(self):
        super().__init__()

    def ipf_step(self, forward_or_backward, n):
        new_dl = self.new_cacheloader(forward_or_backward, n, cfg.EMA)

        if not cfg.USE_PREV_NET:
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)

        self.build_optimizers()
        self.accelerate(forward_or_backward)

        for i in tqdm(range(self.num_iter + 1)):
            self.set_seed(seed=n * self.num_iter + i)

            x, out, steps_expanded = next(new_dl)
            x = x.to(self.device)
            out = out.to(self.device)
            steps_expanded = steps_expanded.to(self.device)
            eval_steps = self.T - steps_expanded

            if cfg.MEAN_MATCH:
                pred = self.net[forward_or_backward](x, eval_steps) - x
            else:
                pred = self.net[forward_or_backward](x, eval_steps)

            loss = F.mse_loss(pred, out)

            self.accelerator.backward(loss)

            if self.grad_clipping:
                clipping_param = cfg.GRAD_CLIP
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net[forward_or_backward].parameters(), clipping_param)
            else:
                total_norm = 0.

            if (i % self.stride_log == 0) and (i > 0):
                self.logger.log_metrics({
                    'forward_or_backward': forward_or_backward,
                    'loss': loss,
                    'grad_norm': total_norm,
                }, step=i + self.num_iter * n)

            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad()
            if cfg.EMA:
                self.ema_helpers[forward_or_backward].update(
                    self.net[forward_or_backward])

            self.save_step(i, n, forward_or_backward)

            if (i % cfg.CACHE_REFRESH_STRIDE == 0) and (i > 0):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(
                    forward_or_backward, n, cfg.EMA)

        new_dl = None
        self.clear()

    def train(self):
        # INITIAL FORWARD PASS
        if self.accelerator.is_local_main_process:
            init_sample = next(self.save_init_dl)[0]
            init_sample = init_sample.to(self.device)
            x_tot, _, _ = self.langevin.record_init_langevin(init_sample)
            shape_len = len(x_tot.shape)
            x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
            x_tot_plot = x_tot.detach()

            self.plotter(init_sample, x_tot_plot, 0, 0, 'f')
            x_tot_plot = None
            x_tot = None
            torch.cuda.empty_cache()

        for n in range(self.checkpoint_it, self.n_ipf + 1):
            print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f', n)
            else:
                self.ipf_step('b', n)
                self.ipf_step('f', n)


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':
    print('=== DSB Training: HES -> CD30 ===')
    print(f'Image size : {cfg.IMAGE_SIZE}')
    print(f'Batch size : {cfg.BATCH_SIZE}')
    print(f'Num iter   : {cfg.NUM_ITER}')
    print(f'Num IPF    : {cfg.N_IPF}')
    print(f'Num steps  : {cfg.NUM_STEPS}')
    print(f'Device     : {cfg.DEVICE}')
    print(f'Transfer   : {cfg.TRANSFER}')
    print(f'Data dir   : {cfg.DATA_DIR}')
    print('Directory  : ' + os.getcwd())

    ipf = IPFSequential()
    ipf.train()

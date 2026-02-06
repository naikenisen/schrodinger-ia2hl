"""
Configuration pour l'entraînement DSB : HES -> CD30 virtual staining.
Tous les hyperparamètres sont centralisés ici.
"""


class Config:
    """Objet de configuration compatible avec l'accès par attributs (args.xxx)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        items = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'Config({items})'


def get_config():
    """Retourne la configuration complète pour HES -> CD30."""

    # --- Sous-config data ---
    data = Config(
        dataset="HES_CD30",
        image_size=256,
        channels=3,
        random_flip=True,
    )

    # --- Sous-config model (UNET) ---
    model = Config(
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )

    # --- Config principale ---
    args = Config(
        # ----- Dataset -----
        Dataset="hes_cd30",
        Dataset_transfer="hes_cd30",
        data=data,
        data_dir="./",                          # dataset_v2/ doit être dans ce dossier
        transfer=True,                          # mode transfer : HES -> CD30
        load=False,

        # ----- Modèle -----
        Model="UNET",
        model=model,

        # ----- Device -----
        device="cuda",                          # "cpu" si pas de GPU
        dataparallel=True,
        num_workers=4,
        pin_memory=True,

        # ----- Entraînement -----
        batch_size=8,
        lr=1e-4,
        num_iter=50000,
        n_ipf=20,
        n_ipf_init=1,
        cache_npar=32,
        num_cache_batches=1,
        cache_refresh_stride=100,
        use_prev_net=True,
        mean_match=True,

        # ----- EMA -----
        ema=True,
        ema_rate=0.999,

        # ----- Gradient clipping -----
        grad_clipping=True,
        grad_clip=1.0,

        # ----- Schedule de diffusion -----
        num_steps=50,
        gamma_max=0.1,
        gamma_min=1e-5,
        gamma_space="linspace",                 # "linspace" ou "geomspace"
        weight_distrib=True,
        weight_distrib_alpha=100,
        fast_sampling=True,

        # ----- Gaussian final (non utilisé en mode transfer, mais requis) -----
        final_adaptive=False,
        adaptive_mean=False,
        mean_final="torch.zeros([3, 256, 256])",
        var_final="torch.ones([3, 256, 256])",

        # ----- Logging -----
        LOGGER="CSV",
        CSV_log_dir="./",
        log_stride=10,
        gif_stride=5000,
        plot_npar=16,
        plot_level=1,

        # ----- Checkpoint -----
        checkpoint_run=False,
        checkpoint_it=1,
        checkpoint_pass="b",
        sample_checkpoint_f="",
        sample_checkpoint_b="",
    )

    return args

# --- Architecture & Data ---
IMAGE_SIZE = 256
NUM_CHANNELS = 128
NUM_RES_BLOCKS = 2
ATTENTION_RESOLUTIONS = "16"
DROPOUT = 0.0

# --- Training ---
BATCH_SIZE = 4
LR = 1e-4
NUM_ITER = 1500
N_IPF = 10
GRAD_CLIP = 1.0

# --- Cache & Langevin (Schrödinger Bridge) ---
CACHE_NPAR = 4
NUM_CACHE_BATCHES = 10
CACHE_REFRESH_STRIDE = 25
NUM_WORKERS = 2

# --- Diffusion Physics ---
NUM_STEPS = 10
GAMMA_MAX = 0.1
GAMMA_MIN = 1e-5
GAMMA_SPACE = "linspace"
EMA_RATE = 0.999

device = "cuda"
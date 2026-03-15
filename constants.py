import os
from stablepy import (
    scheduler_names,
    SD15_TASKS,
    SDXL_TASKS,
    ALL_BUILTIN_UPSCALERS,
)

IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))

# ── Model to auto-download at startup ─────────────────────────
DOWNLOAD_MODEL = os.getenv("DOWNLOAD_MODEL", "")

# ── VAE to auto-download ──────────────────────────────────────
DOWNLOAD_VAE = os.getenv("DOWNLOAD_VAE", "https://huggingface.co/fp16-guy/anything_kl-f8-anime2_vae-ft-mse-840000-ema-pruned_blessed_clearvae_fp16_cleaned/resolve/main/vae-ft-mse-840000-ema-pruned_fp16.safetensors?download=true")

# ── LoRAs to auto-download ────────────────────────────────────
DOWNLOAD_LORA = os.getenv("DOWNLOAD_LORA", "")

# ── Default model list ────────────────────────────────────────
LOAD_DIFFUSERS_FORMAT_MODEL = [
    "votepurchase/pornmasterPro_noobV3VAE",
]

DIFFUSERS_FORMAT_LORAS = []

DOWNLOAD_EMBEDS = []

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")
HF_TOKEN        = os.environ.get("HF_READ_TOKEN", "")

DIRECTORY_MODELS    = "models"
DIRECTORY_LORAS     = "loras"
DIRECTORY_VAES      = "vaes"
DIRECTORY_EMBEDS    = "embedings"
DIRECTORY_UPSCALERS = "upscalers"

CACHE_HF_ROOT = os.path.expanduser("~/.cache/huggingface")
CACHE_HF      = os.path.join(CACHE_HF_ROOT, "hub")

# ── Tasks ──────────────────────────────────────────────────────
TASK_STABLEPY = {
    'txt2img': 'txt2img',
    'img2img': 'img2img',
    'inpaint': 'inpaint',
}

TASK_MODEL_LIST = list(TASK_STABLEPY.keys())

# ── Upscalers ──────────────────────────────────────────────────
UPSCALER_DICT_GUI = {
    None: None,
    **{bu: bu for bu in ALL_BUILTIN_UPSCALERS if bu not in ["HAT x4", "DAT x4", "DAT x3", "DAT x2", "SwinIR 4x"]},
    "4x-UltraSharp":          "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    "Real-ESRGAN-Anime":      "https://huggingface.co/danhtran2mind/Real-ESRGAN-Anime-finetuning/resolve/main/Real-ESRGAN-Anime-finetuning.pth",
    "AnimeSharp 4x":          "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    "4x_foolhardy_Remacri":   "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
}

UPSCALER_KEYS = list(UPSCALER_DICT_GUI.keys())

WARNING_MSG_VAE = (
    "Use the right VAE for your model. The wrong VAE can lead to poor results."
)

SDXL_TASK  = [k for k, v in TASK_STABLEPY.items() if v in SDXL_TASKS]
SD_TASK    = [k for k, v in TASK_STABLEPY.items() if v in SD15_TASKS]
FLUX_TASK  = list(TASK_STABLEPY.keys())[:2]

MODEL_TYPE_TASK = {
    "SD 1.5": SD_TASK,
    "SDXL":   SDXL_TASK,
    "FLUX":   FLUX_TASK,
}

POST_PROCESSING_SAMPLER = ["Use same sampler"] + [
    name_s for name_s in scheduler_names if "Auto-Loader" not in name_s
]

SUBTITLE_GUI = (
    "### Lightweight anime image generation powered by stablepy & diffusers."
)

HELP_GUI = ""
EXAMPLES_GUI_HELP = ""
EXAMPLES_GUI = []
RESOURCES = ""

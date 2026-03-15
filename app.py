import os
import gc
import random
import warnings
import urllib.request
import torch
import gradio as gr
import numpy as np
from PIL import Image
from datetime import datetime
from stablepy import Model_Diffusers, scheduler_names
from stablepy import load_upscaler_model

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────
IS_COLAB    = bool(os.getenv("IS_COLAB", "0") == "1")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "./outputs")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
CIVITAI_KEY = os.getenv("CIVITAI_API_KEY", "")

for d in [OUTPUT_DIR, "./models", "./loras", "./vaes", "./upscalers"]:
    os.makedirs(d, exist_ok=True)

MAX_SEED         = np.iinfo(np.int32).max
DEFAULT_MODEL    = "votepurchase/pornmasterPro_noobV3VAE"
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, very displeasing, "
    "signature, watermark, username, blurry, bad feet"
)

QUALITY_TAGS = {
    "NoobAI — High":           ("masterpiece, best quality, very aesthetic, absurdres", ""),
    "NoobAI — Standard":       ("masterpiece, best quality", ""),
    "Animagine v4 — High":     ("masterpiece, high score, great score, absurdres, highres", "low score, bad score, average score"),
    "Animagine v4 — Standard": ("masterpiece, high score, great score", "low score, bad score, average score"),
    "Pony — High":             ("score_9, score_8_up, score_7_up, masterpiece, best quality", "score_6, score_5, score_4"),
    "None":                    ("", ""),
}

STYLES = {
    "(None)":       ("", ""),
    "Anime":        ("{prompt}, anime style", "photorealistic, 3d render"),
    "Illustration": ("{prompt}, illustration, digital art", "photo, realistic"),
    "Watercolor":   ("{prompt}, watercolor painting", ""),
    "Sketch":       ("{prompt}, pencil sketch, line art", "color, painted"),
    "Cinematic":    ("{prompt}, cinematic lighting, dramatic", ""),
}

UPSCALER_URL  = "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth"
UPSCALER_PATH = "./upscalers/AnimeSharp_4x.pth"

# ── Helpers ────────────────────────────────────────────────────
def lora_chk(lora):
    if isinstance(lora, str) and lora.strip() not in ["", "None"]:
        return lora.strip()
    return None

def download_url(directory, url, civitai_key=""):
    if not url or not url.strip():
        return None
    url = url.strip()
    if "civitai.com" in url and civitai_key:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={civitai_key}"
    os.makedirs(directory, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0] or "downloaded.safetensors"
    path = os.path.join(directory, filename)
    print(f"[DiffuseLite] Downloading {filename}...")
    urllib.request.urlretrieve(url, path)
    print(f"[DiffuseLite] Saved to {path}")
    return path

def apply_quality(prompt, neg, quality_key, add_tags):
    if not add_tags or quality_key == "None":
        return prompt, neg
    pos, extra_neg = QUALITY_TAGS[quality_key]
    p = f"{pos}, {prompt}".strip(", ") if pos else prompt
    n = f"{extra_neg}, {neg}".strip(", ") if extra_neg else neg
    return p, n

def apply_style(prompt, neg, style_key):
    if style_key == "(None)":
        return prompt, neg
    pos, sneg = STYLES[style_key]
    p = pos.replace("{prompt}", prompt) if "{prompt}" in pos else f"{prompt}, {pos}"
    n = f"{sneg}, {neg}".strip(", ") if sneg else neg
    return p, n

def run_upscale(image, upscale_by):
    if not os.path.exists(UPSCALER_PATH):
        print("[DiffuseLite] Downloading AnimeSharp 4x upscaler...")
        urllib.request.urlretrieve(UPSCALER_URL, UPSCALER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = load_upscaler_model(
        model=UPSCALER_PATH, tile=192, tile_overlap=8,
        device=device, half=(device == "cuda"),
    )
    return scaler.upscale(image, upscale_by, True)

# ── Model ──────────────────────────────────────────────────────
sd_model         = None
current_model_id = None

def load_model(model_id, progress=gr.Progress(track_tqdm=True)):
    global sd_model, current_model_id
    model_id = model_id.strip()
    if current_model_id == model_id and sd_model is not None:
        return f"<p style='color:green'>✅ Already loaded: <b>{model_id}</b></p>"
    try:
        if model_id.startswith("http"):
            model_id = download_url("./models", model_id, CIVITAI_KEY)
        print(f"[DiffuseLite] Loading {model_id}...")
        if sd_model is None:
            sd_model = Model_Diffusers(
                base_model_id=model_id,
                task_name="txt2img",
                type_model_precision=torch.float16,
                retain_task_model_in_cache=False,
            )
        else:
            sd_model.load_pipe(
                model_id,
                task_name="txt2img",
                type_model_precision=torch.float16,
                retain_task_model_in_cache=False,
            )
        current_model_id = model_id
        print("[DiffuseLite] Model loaded!")
        return f"<p style='color:green'>✅ Loaded: <b>{model_id}</b></p>"
    except Exception as e:
        return f"<p style='color:red'>❌ Error: {e}</p>"

# ── Generation ─────────────────────────────────────────────────
def txt2img(
    model_id, prompt, neg_prompt,
    quality_key, add_quality, style_key,
    img_width, img_height,
    guidance_scale, num_steps, sampler,
    seed, randomize_seed, num_images,
    lora1, lora_scale1,
    use_upscaler, upscale_by,
    progress=gr.Progress(track_tqdm=True),
):
    global sd_model, current_model_id
    model_id = model_id.strip()

    if sd_model is None or current_model_id != model_id:
        msg = load_model(model_id)
        if sd_model is None:
            raise gr.Error(f"Model failed to load: {model_id}")

    if current_model_id == model_id and hasattr(sd_model, 'task_name') and sd_model.task_name != "txt2img":
        sd_model.load_pipe(model_id, task_name="txt2img",
                           type_model_precision=torch.float16,
                           retain_task_model_in_cache=False)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    prompt, neg_prompt = apply_quality(prompt, neg_prompt, quality_key, add_quality)
    prompt, neg_prompt = apply_style(prompt, neg_prompt, style_key)

    # ── Critical: stream_config like DiffuseCraft ──────────────
    concurrency = 1
    sd_model.stream_config(concurrency=concurrency, latent_resize_by=1, vae_decoding=False)

    images_out = []
    final_seed = seed

    try:
        for result in sd_model(
            prompt=prompt,
            negative_prompt=neg_prompt,
            img_height=int(img_height),
            img_width=int(img_width),
            num_images=int(num_images),
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            clip_skip=True,
            seed=int(seed),
            sampler=sampler,
            lora_A=lora_chk(lora1),
            lora_scale_A=float(lora_scale1),
            syntax_weights="Classic",
            gui_active=True,
            loop_generation=1,
            save_generated_images=True,
            image_storage_location=OUTPUT_DIR,
            display_images=False,
            image_previews=False,
            disable_progress_bar=False,
            leave_progress_bar=False,
        ):
            if isinstance(result, (list, tuple)) and len(result) == 2:
                img, info = result
                if isinstance(info, (list, tuple)) and len(info) >= 2:
                    s, image_paths = info[0], info[1]
                    if image_paths:
                        images_out = image_paths
                        final_seed = s

        if use_upscaler and images_out:
            upscaled = []
            for p in images_out:
                up = run_upscale(Image.open(p).convert("RGB"), upscale_by)
                up_path = p.replace(".png", "_up.png")
                up.save(up_path, format="PNG")
                upscaled.append(up_path)
            images_out = upscaled

        dl = images_out[0] if images_out else None
        return images_out, final_seed, gr.update(value=dl, visible=dl is not None)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def img2img_fn(
    model_id, input_image,
    prompt, neg_prompt,
    quality_key, add_quality, style_key,
    strength, guidance_scale, num_steps, sampler,
    seed, randomize_seed,
    lora1, lora_scale1,
    use_upscaler, upscale_by,
    progress=gr.Progress(track_tqdm=True),
):
    global sd_model, current_model_id
    if input_image is None:
        raise gr.Error("Please upload an image.")
    model_id = model_id.strip()

    if sd_model is None or current_model_id != model_id:
        load_model(model_id)
        if sd_model is None:
            raise gr.Error(f"Model failed to load: {model_id}")

    sd_model.load_pipe(model_id, task_name="img2img",
                       type_model_precision=torch.float16,
                       retain_task_model_in_cache=False)
    current_model_id = model_id

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    prompt, neg_prompt = apply_quality(prompt, neg_prompt, quality_key, add_quality)
    prompt, neg_prompt = apply_style(prompt, neg_prompt, style_key)

    tmp = "/tmp/i2i_input.png"
    Image.fromarray(input_image).convert("RGB").save(tmp)

    # ── Critical: stream_config like DiffuseCraft ──────────────
    sd_model.stream_config(concurrency=1, latent_resize_by=1, vae_decoding=False)

    images_out = []
    final_seed = seed

    try:
        for result in sd_model(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_images=1,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            clip_skip=True,
            seed=int(seed),
            sampler=sampler,
            image=tmp,
            strength=float(strength),
            lora_A=lora_chk(lora1),
            lora_scale_A=float(lora_scale1),
            syntax_weights="Classic",
            gui_active=True,
            loop_generation=1,
            save_generated_images=True,
            image_storage_location=OUTPUT_DIR,
            display_images=False,
            image_previews=False,
            disable_progress_bar=False,
            leave_progress_bar=False,
        ):
            if isinstance(result, (list, tuple)) and len(result) == 2:
                img, info = result
                if isinstance(info, (list, tuple)) and len(info) >= 2:
                    s, image_paths = info[0], info[1]
                    if image_paths:
                        images_out = image_paths
                        final_seed = s

        if use_upscaler and images_out:
            upscaled = []
            for p in images_out:
                up = run_upscale(Image.open(p).convert("RGB"), upscale_by)
                up_path = p.replace(".png", "_up.png")
                up.save(up_path, format="PNG")
                upscaled.append(up_path)
            images_out = upscaled

        dl = images_out[0] if images_out else None
        return images_out, final_seed, gr.update(value=dl, visible=dl is not None)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def download_lora_ui(url, civitai_key):
    if not url.strip():
        return "❌ Enter a URL.", ""
    try:
        path = download_url("./loras", url.strip(), civitai_key)
        return f"✅ Downloaded! Use this path: {path}", path or ""
    except Exception as e:
        return f"❌ Error: {e}", ""


# ── UI ─────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; margin-bottom: 4px; }
#title h1 { font-size: 2.2em; }
#gen-btn { height: 50px; font-size: 1.1em; }
"""

with gr.Blocks(theme="NoCrypt/miku@1.2.1", css=CSS) as demo:
    gr.HTML("<h1 id='title'>🎨 DiffuseLite</h1>")
    gr.Markdown("Lightweight anime image generation · txt2img · img2img · upscaler · LoRA")

    with gr.Row():
        model_input    = gr.Textbox(label="🤖 Model (HuggingFace repo ID or URL)", value=DEFAULT_MODEL, scale=4)
        load_model_btn = gr.Button("📥 Load Model", variant="primary", scale=1)
    model_status = gr.HTML()
    load_model_btn.click(fn=load_model, inputs=model_input, outputs=model_status)

    with gr.Row():
        with gr.Column(scale=2):

            with gr.Tab("🖼️ Txt2Img"):
                prompt_t2i = gr.Textbox(label="Prompt", lines=4, placeholder="1girl, solo, smile, looking at viewer...")
                neg_t2i    = gr.Textbox(label="Negative Prompt", lines=2, value=DEFAULT_NEGATIVE)
                with gr.Accordion("⭐ Quality & Style", open=True):
                    with gr.Row():
                        add_q_t2i   = gr.Checkbox(label="Add Quality Tags", value=True)
                        quality_t2i = gr.Dropdown(choices=list(QUALITY_TAGS.keys()), value="NoobAI — High", label="Quality Preset")
                    style_t2i = gr.Radio(choices=list(STYLES.keys()), value="(None)", label="Style")
                with gr.Accordion("📐 Image Size", open=True):
                    w_t2i = gr.Slider(512, 2048, step=8, value=896,  label="Width")
                    h_t2i = gr.Slider(512, 2048, step=8, value=1152, label="Height")
                with gr.Accordion("⚙️ Settings", open=False):
                    sampler_t2i  = gr.Dropdown(choices=scheduler_names, value="Euler a", label="Sampler")
                    steps_t2i    = gr.Slider(1, 50, step=1, value=28, label="Steps")
                    cfg_t2i      = gr.Slider(1, 12, step=0.5, value=7.0, label="CFG Scale")
                    num_imgs_t2i = gr.Slider(1, 4, step=1, value=1, label="Number of Images")
                    with gr.Row():
                        seed_t2i = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
                        rand_t2i = gr.Checkbox(label="Randomize", value=True)
                with gr.Accordion("🔍 Upscaler — AnimeSharp 4x", open=False):
                    use_up_t2i = gr.Checkbox(label="Enable Upscaler", value=False)
                    upby_t2i   = gr.Slider(1.1, 4.0, step=0.1, value=2.0, label="Upscale by")
                with gr.Accordion("🎛️ LoRA", open=False):
                    gr.Markdown("Download a LoRA first, then its path auto-fills below.")
                    lora_dl_url     = gr.Textbox(label="Civitai / HF Download URL", placeholder="https://civitai.com/api/download/models/...")
                    dl_lora_btn     = gr.Button("⬇️ Download LoRA", variant="secondary")
                    lora_status_t2i = gr.Textbox(label="Download Status", interactive=False)
                    lora1_t2i       = gr.Textbox(label="LoRA path (auto-filled after download)", placeholder="./loras/my_lora.safetensors")
                    lora_scale1_t2i = gr.Slider(0.1, 1.5, step=0.05, value=0.8, label="LoRA Scale")
                gen_t2i = gr.Button("🎨 Generate", variant="primary", elem_id="gen-btn")

            with gr.Tab("🖼️➡️🖼️ Img2Img"):
                input_img  = gr.Image(label="Input Image", type="numpy", height=260)
                prompt_i2i = gr.Textbox(label="Prompt", lines=4, placeholder="1girl, solo, smile...")
                neg_i2i    = gr.Textbox(label="Negative Prompt", lines=2, value=DEFAULT_NEGATIVE)
                with gr.Accordion("⭐ Quality & Style", open=True):
                    with gr.Row():
                        add_q_i2i   = gr.Checkbox(label="Add Quality Tags", value=True)
                        quality_i2i = gr.Dropdown(choices=list(QUALITY_TAGS.keys()), value="NoobAI — High", label="Quality Preset")
                    style_i2i = gr.Radio(choices=list(STYLES.keys()), value="(None)", label="Style")
                with gr.Accordion("⚙️ Settings", open=False):
                    strength_i2i = gr.Slider(0.1, 1.0, step=0.05, value=0.75, label="Denoising Strength")
                    sampler_i2i  = gr.Dropdown(choices=scheduler_names, value="Euler a", label="Sampler")
                    steps_i2i    = gr.Slider(1, 50, step=1, value=28, label="Steps")
                    cfg_i2i      = gr.Slider(1, 12, step=0.5, value=7.0, label="CFG Scale")
                    with gr.Row():
                        seed_i2i = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
                        rand_i2i = gr.Checkbox(label="Randomize", value=True)
                with gr.Accordion("🔍 Upscaler — AnimeSharp 4x", open=False):
                    use_up_i2i = gr.Checkbox(label="Enable Upscaler", value=False)
                    upby_i2i   = gr.Slider(1.1, 4.0, step=0.1, value=2.0, label="Upscale by")
                with gr.Accordion("🎛️ LoRA", open=False):
                    lora1_i2i       = gr.Textbox(label="LoRA path", placeholder="./loras/my_lora.safetensors")
                    lora_scale1_i2i = gr.Slider(0.1, 1.5, step=0.05, value=0.8, label="LoRA Scale")
                gen_i2i = gr.Button("🎨 Generate", variant="primary", elem_id="gen-btn")

        with gr.Column(scale=3):
            output_gallery = gr.Gallery(
                label="Output", columns=2, height=580,
                preview=True, show_label=False, object_fit="contain",
            )
            used_seed    = gr.Number(label="🌱 Seed Used", interactive=False)
            download_btn = gr.File(label="⬇️ Download PNG (full resolution)", interactive=False, visible=False)

    with gr.Row():
        civitai_key_input = gr.Textbox(
            label="🔑 Civitai API Key (for LoRA downloads)",
            placeholder="Paste your Civitai API key here",
            type="password",
        )

    dl_lora_btn.click(
        fn=download_lora_ui,
        inputs=[lora_dl_url, civitai_key_input],
        outputs=[lora_status_t2i, lora1_t2i],
    )

    gen_t2i.click(
        fn=txt2img,
        inputs=[model_input, prompt_t2i, neg_t2i, quality_t2i, add_q_t2i, style_t2i,
                w_t2i, h_t2i, cfg_t2i, steps_t2i, sampler_t2i, seed_t2i, rand_t2i,
                num_imgs_t2i, lora1_t2i, lora_scale1_t2i, use_up_t2i, upby_t2i],
        outputs=[output_gallery, used_seed, download_btn],
    )

    gen_i2i.click(
        fn=img2img_fn,
        inputs=[model_input, input_img, prompt_i2i, neg_i2i, quality_i2i, add_q_i2i,
                style_i2i, strength_i2i, cfg_i2i, steps_i2i, sampler_i2i, seed_i2i,
                rand_i2i, lora1_i2i, lora_scale1_i2i, use_up_i2i, upby_i2i],
        outputs=[output_gallery, used_seed, download_btn],
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        share=IS_COLAB, debug=IS_COLAB,
        allowed_paths=[OUTPUT_DIR, "./loras"],
    )

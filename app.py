import os
import gc
import random
import warnings
import torch
import gradio as gr
import numpy as np
from PIL import Image
from datetime import datetime
from stablepy import (
    Model_Diffusers,
    scheduler_names,
    ALL_BUILTIN_UPSCALERS,
)
from stablepy import load_upscaler_model
from utils import download_things

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────
IS_COLAB    = bool(os.getenv("IS_COLAB", "0") == "1")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "./outputs")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
CIVITAI_KEY = os.getenv("CIVITAI_API_KEY", "")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./loras",  exist_ok=True)
os.makedirs("./vaes",   exist_ok=True)
os.makedirs("./upscalers", exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max

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

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model holder ──────────────────────────────────────────────
model = None
current_model_id = None


def load_model(model_id, progress=gr.Progress(track_tqdm=True)):
    global model, current_model_id
    model_id = model_id.strip()
    if current_model_id == model_id and model is not None:
        return f"<p style='color:green'>✅ Already loaded: <b>{model_id}</b></p>"

    try:
        # Download if it's a URL
        if model_id.startswith("http"):
            model_id = download_things("./models", model_id, HF_TOKEN, CIVITAI_KEY)

        print(f"[DiffuseLite] Loading {model_id}...")
        model = Model_Diffusers(
            base_model_id=model_id,
            task_name="txt2img",
            type_model_precision=torch.float16,
            retain_task_model_in_cache=False,
        )
        current_model_id = model_id
        print("[DiffuseLite] Model loaded!")
        return f"<p style='color:green'>✅ Loaded: <b>{model_id}</b></p>"
    except Exception as e:
        return f"<p style='color:red'>❌ Error: {e}</p>"


# ── Helpers ───────────────────────────────────────────────────
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


def download_upscaler():
    if not os.path.exists(UPSCALER_PATH):
        print("[DiffuseLite] Downloading AnimeSharp 4x upscaler...")
        import urllib.request
        urllib.request.urlretrieve(UPSCALER_URL, UPSCALER_PATH)
        print("[DiffuseLite] Upscaler downloaded.")


def run_upscale(image, upscale_by):
    download_upscaler()
    scaler = load_upscaler_model(
        model=UPSCALER_PATH, tile=192, tile_overlap=8,
        device=device, half=(device == "cuda"),
    )
    return scaler.upscale(image, upscale_by, True)


def download_lora(url, civitai_key):
    import urllib.request
    if "civitai.com" in url and civitai_key:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={civitai_key}"
    filename = f"lora_{datetime.now().strftime('%H%M%S')}.safetensors"
    path = os.path.join("./loras", filename)
    print(f"[DiffuseLite] Downloading LoRA...")
    urllib.request.urlretrieve(url, path)
    return path


# ── Generation ────────────────────────────────────────────────
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
    global model, current_model_id
    if model is None or current_model_id != model_id.strip():
        load_model(model_id)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    prompt, neg_prompt = apply_quality(prompt, neg_prompt, quality_key, add_quality)
    prompt, neg_prompt = apply_style(prompt, neg_prompt, style_key)

    lora_a = lora1.strip() if lora1 and lora1.strip() not in ["", "None"] else None

    try:
        model.load_pipe(
            model_id.strip(),
            task_name="txt2img",
            type_model_precision=torch.float16,
            retain_task_model_in_cache=False,
        )
    except Exception:
        pass

    images_out = []
    try:
        for img, [s, paths, meta] in model(
            prompt=prompt,
            negative_prompt=neg_prompt,
            img_height=int(img_height),
            img_width=int(img_width),
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            sampler=sampler,
            lora_A=lora_a,
            lora_scale_A=lora_scale1,
            save_generated_images=True,
            image_storage_location=OUTPUT_DIR,
            gui_active=True,
            display_images=False,
            image_previews=False,
            disable_progress_bar=False,
        ):
            if paths:
                images_out = paths
                seed = s

        if use_upscaler and images_out:
            upscaled = []
            for p in images_out:
                img_pil = Image.open(p).convert("RGB")
                up = run_upscale(img_pil, upscale_by)
                up_path = p.replace(".png", "_upscaled.png")
                up.save(up_path, format="PNG")
                upscaled.append(up_path)
            images_out = upscaled

        return images_out, seed, images_out[0] if images_out else None

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
    global model, current_model_id
    if input_image is None:
        raise gr.Error("Please upload an image.")

    if model is None or current_model_id != model_id.strip():
        load_model(model_id)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    prompt, neg_prompt = apply_quality(prompt, neg_prompt, quality_key, add_quality)
    prompt, neg_prompt = apply_style(prompt, neg_prompt, style_key)

    lora_a = lora1.strip() if lora1 and lora1.strip() not in ["", "None"] else None

    # Save input image temporarily
    tmp_input = "/tmp/i2i_input.png"
    Image.fromarray(input_image).convert("RGB").save(tmp_input)

    try:
        model.load_pipe(
            model_id.strip(),
            task_name="img2img",
            type_model_precision=torch.float16,
            retain_task_model_in_cache=False,
        )
    except Exception:
        pass

    images_out = []
    try:
        for img, [s, paths, meta] in model(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_images=1,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            sampler=sampler,
            image=tmp_input,
            strength=strength,
            lora_A=lora_a,
            lora_scale_A=lora_scale1,
            save_generated_images=True,
            image_storage_location=OUTPUT_DIR,
            gui_active=True,
            display_images=False,
            image_previews=False,
            disable_progress_bar=False,
        ):
            if paths:
                images_out = paths
                seed = s

        if use_upscaler and images_out:
            upscaled = []
            for p in images_out:
                img_pil = Image.open(p).convert("RGB")
                up = run_upscale(img_pil, upscale_by)
                up_path = p.replace(".png", "_upscaled.png")
                up.save(up_path, format="PNG")
                upscaled.append(up_path)
            images_out = upscaled

        return images_out, seed, images_out[0] if images_out else None

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_lora_ui(lora_url, civitai_key):
    if not lora_url.strip():
        return "❌ Enter a LoRA URL or HF repo ID."
    try:
        if lora_url.startswith("http"):
            path = download_lora(lora_url.strip(), civitai_key)
            return f"✅ Downloaded: {os.path.basename(path)}\nPath: {path}"
        return f"✅ Using HF repo: {lora_url.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


# ── UI ────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; margin-bottom: 4px; }
#title h1 { font-size: 2.2em; }
#gen-btn { height: 50px; font-size: 1.1em; }
"""

with gr.Blocks(theme="NoCrypt/miku@1.2.1", css=CSS) as demo:
    gr.HTML("<h1 id='title'>🎨 DiffuseLite</h1>")
    gr.Markdown("Lightweight anime image generation · txt2img · img2img · upscaler · LoRA")

    with gr.Row():
        model_input    = gr.Textbox(label="🤖 Model (HuggingFace repo ID or download URL)", value=DEFAULT_MODEL, scale=4)
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
                    gr.Markdown("Paste a **Civitai URL** or **HuggingFace repo ID**. Use the Download button first for Civitai links.")
                    lora1_t2i       = gr.Textbox(label="LoRA path / HF repo ID", placeholder="./loras/my_lora.safetensors")
                    lora_scale1_t2i = gr.Slider(0.1, 1.5, step=0.05, value=0.8, label="LoRA Scale")
                    lora_dl_url     = gr.Textbox(label="Civitai / HF Download URL", placeholder="https://civitai.com/api/download/models/...")
                    dl_lora_btn     = gr.Button("⬇️ Download LoRA", variant="secondary")
                    lora_status_t2i = gr.Textbox(label="LoRA Status", interactive=False)
                    dl_lora_btn.click(fn=lambda url, key: load_lora_ui(url, key),
                                      inputs=[lora_dl_url, gr.State("")],
                                      outputs=lora_status_t2i)
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
                    lora1_i2i       = gr.Textbox(label="LoRA path / HF repo ID", placeholder="./loras/my_lora.safetensors")
                    lora_scale1_i2i = gr.Slider(0.1, 1.5, step=0.05, value=0.8, label="LoRA Scale")
                gen_i2i = gr.Button("🎨 Generate", variant="primary", elem_id="gen-btn")

        with gr.Column(scale=3):
            output_gallery = gr.Gallery(label="Output", columns=2, height=580, preview=True, show_label=False, object_fit="contain")
            used_seed    = gr.Number(label="🌱 Seed Used", interactive=False)
            download_btn = gr.File(label="⬇️ Download PNG (full resolution)", interactive=False, visible=False)

    with gr.Row():
        civitai_key_input = gr.Textbox(
            label="🔑 Civitai API Key (for LoRA downloads)",
            placeholder="Paste your Civitai API key here",
            type="password",
        )

    # wire download lora with civitai key
    dl_lora_btn.click(
        fn=lambda url, key: load_lora_ui(url, key),
        inputs=[lora_dl_url, civitai_key_input],
        outputs=lora_status_t2i,
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
        share=IS_COLAB,
        debug=IS_COLAB,
        allowed_paths=[OUTPUT_DIR, "./loras"],
    )

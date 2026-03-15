import os
import gc
import random
import warnings
import torch
import gradio as gr
import numpy as np
from PIL import Image, PngImagePlugin
from datetime import datetime
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.models import AutoencoderKL
from compel import Compel, ReturnedEmbeddingsType

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

IS_COLAB   = bool(os.getenv("IS_COLAB", "0") == "1")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./upscalers", exist_ok=True)
os.makedirs("./loras", exist_ok=True)

MAX_SEED        = np.iinfo(np.int32).max
DEFAULT_MODEL   = "votepurchase/pornmasterPro_noobV3VAE"
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "")
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, very displeasing, "
    "signature, watermark, username, blurry, bad feet"
)

SAMPLERS = {
    "Euler a":       lambda c: EulerAncestralDiscreteScheduler.from_config(c),
    "Euler":         lambda c: EulerDiscreteScheduler.from_config(c),
    "DPM++ 2M":      lambda c: DPMSolverMultistepScheduler.from_config(c),
    "DPM++ 2M SDE":  lambda c: DPMSolverMultistepScheduler.from_config(c, algorithm_type="sde-dpmsolver++"),
    "DPM++ SDE":     lambda c: DPMSolverSinglestepScheduler.from_config(c),
    "DPM2 a Karras": lambda c: KDPM2AncestralDiscreteScheduler.from_config(c),
    "Heun":          lambda c: HeunDiscreteScheduler.from_config(c),
    "DDIM":          lambda c: DDIMScheduler.from_config(c),
    "UniPC":         lambda c: UniPCMultistepScheduler.from_config(c),
}

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipelineManager:
    def __init__(self):
        self.pipe          = None
        self.current_model = None

    def load(self, model_id):
        model_id = model_id.strip()
        if self.current_model == model_id and self.pipe is not None:
            return
        print(f"[DiffuseLite] Loading {model_id}...")
        self.unload()
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        load_fn = StableDiffusionXLPipeline.from_single_file if model_id.endswith(".safetensors") else StableDiffusionXLPipeline.from_pretrained
        self.pipe = load_fn(model_id, vae=vae, torch_dtype=torch.float16, use_safetensors=True, add_watermarker=False).to(device)
        self.current_model = model_id
        print("[DiffuseLite] Model loaded!")

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.current_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_lora(self, lora_path, lora_scale):
        if not self.pipe:
            return "❌ Load a model first."
        try:
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora(lora_scale=lora_scale)
            return f"✅ LoRA loaded: {os.path.basename(lora_path)}"
        except Exception as e:
            return f"❌ Failed: {e}"

    def clear_loras(self):
        if not self.pipe:
            return "❌ No model loaded."
        try:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            return "✅ LoRAs cleared."
        except Exception as e:
            return f"❌ {e}"


manager = PipelineManager()


def encode_prompt_compel(pipe, prompt, neg_prompt):
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )
    pos_embeds, pos_pooled = compel([prompt, prompt])
    neg_embeds, neg_pooled = compel([neg_prompt, neg_prompt])
    [pos_embeds, neg_embeds] = compel.pad_last_dim([pos_embeds, neg_embeds])
    return pos_embeds, neg_embeds, pos_pooled, neg_pooled


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


def set_scheduler(pipe, sampler):
    pipe.scheduler = SAMPLERS[sampler](pipe.scheduler.config)


def save_png(image, prompt, seed, model_id):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{ts}_seed{seed}.png")
    meta = PngImagePlugin.PngInfo()
    meta.add_text("prompt", prompt)
    meta.add_text("seed", str(seed))
    meta.add_text("model", model_id)
    image.save(path, format="PNG", pnginfo=meta)
    return path


def download_upscaler():
    if not os.path.exists(UPSCALER_PATH):
        print("[DiffuseLite] Downloading AnimeSharp 4x upscaler...")
        import urllib.request
        urllib.request.urlretrieve(UPSCALER_URL, UPSCALER_PATH)
        print("[DiffuseLite] Upscaler downloaded.")


def run_upscale(image, upscale_by):
    download_upscaler()
    from stablepy import load_upscaler_model
    scaler = load_upscaler_model(model=UPSCALER_PATH, tile=192, tile_overlap=8,
                                  device=("cuda" if torch.cuda.is_available() else "cpu"),
                                  half=torch.cuda.is_available())
    return scaler.upscale(image, upscale_by, True)


def download_civitai_lora(url, civitai_key):
    import urllib.request
    if "civitai.com" in url and civitai_key:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={civitai_key}"
    filename = f"lora_{datetime.now().strftime('%H%M%S')}.safetensors"
    path = os.path.join("./loras", filename)
    print(f"[DiffuseLite] Downloading LoRA...")
    urllib.request.urlretrieve(url, path)
    return path


def txt2img(model_id, prompt, neg_prompt, quality_key, add_quality, style_key,
            img_width, img_height, guidance_scale, num_steps, sampler,
            seed, randomize_seed, num_images, use_upscaler, upscale_by,
            progress=gr.Progress(track_tqdm=True)):
    manager.load(model_id)
    pipe = manager.pipe
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    prompt, neg_prompt = apply_quality(prompt, neg_prompt, quality_key, add_quality)
    prompt, neg_prompt = apply_style(prompt, neg_prompt, style_key)
    set_scheduler(pipe, sampler)
    pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_compel(pipe, prompt, neg_prompt)
    try:
        images = pipe(
            prompt_embeds=pos_embeds, negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled, negative_pooled_prompt_embeds=neg_pooled,
            width=int(img_width), height=int(img_height),
            guidance_scale=guidance_scale, num_inference_steps=num_steps,
            num_images_per_prompt=num_images, generator=generator,
        ).images
        if use_upscaler:
            images = [run_upscale(img, upscale_by) for img in images]
        paths = [save_png(img, prompt, seed, model_id) for img in images]
        return paths, seed, paths[0] if paths else None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def img2img_fn(model_id, input_image, prompt, neg_prompt, quality_key, add_quality, style_key,
               strength, guidance_scale, num_steps, sampler, seed, randomize_seed,
               use_upscaler, upscale_by, progress=gr.Progress(track_tqdm=True)):
    manager.load(model_id)
    pipe = manager.pipe
    if input_image is None:
        raise gr.Error("Please upload an image.")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    prompt, neg_prompt = apply_quality(prompt, neg_prompt, quality_key, add_quality)
    prompt, neg_prompt = apply_style(prompt, neg_prompt, style_key)
    set_scheduler(pipe, sampler)
    pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_compel(pipe, prompt, neg_prompt)
    i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    pil = Image.fromarray(input_image).convert("RGB")
    try:
        images = i2i(
            prompt_embeds=pos_embeds, negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled, negative_pooled_prompt_embeds=neg_pooled,
            image=pil, strength=strength, guidance_scale=guidance_scale,
            num_inference_steps=num_steps, generator=generator,
        ).images
        if use_upscaler:
            images = [run_upscale(img, upscale_by) for img in images]
        paths = [save_png(img, prompt, seed, model_id) for img in images]
        return paths, seed, paths[0] if paths else None
    finally:
        del i2i
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_lora_fn(lora_url, lora_scale, civitai_key):
    if not lora_url.strip():
        return "❌ Please enter a LoRA URL or HF repo ID."
    try:
        if lora_url.startswith("http"):
            path = download_civitai_lora(lora_url, civitai_key)
        else:
            path = lora_url.strip()
        return manager.load_lora(path, lora_scale)
    except Exception as e:
        return f"❌ Error: {e}"


CSS = """
#title { text-align: center; margin-bottom: 4px; }
#title h1 { font-size: 2.2em; }
#gen-btn { height: 50px; font-size: 1.1em; }
"""

with gr.Blocks(theme="NoCrypt/miku@1.2.1", css=CSS) as demo:
    gr.HTML("<h1 id='title'>🎨 DiffuseLite</h1>")
    gr.Markdown("Lightweight anime image generation · txt2img · img2img · upscaler · LoRA")

    with gr.Row():
        model_input = gr.Textbox(label="🤖 Model (HuggingFace repo ID or .safetensors path)", value=DEFAULT_MODEL, scale=4)
        load_model_btn = gr.Button("📥 Load Model", variant="primary", scale=1)
    model_status = gr.HTML()

    def load_model_ui(model_id):
        try:
            manager.load(model_id.strip())
            return f"<p style='color:green'>✅ Loaded: <b>{model_id.strip()}</b></p>"
        except Exception as e:
            return f"<p style='color:red'>❌ {e}</p>"

    load_model_btn.click(fn=load_model_ui, inputs=model_input, outputs=model_status)

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
                    sampler_t2i  = gr.Dropdown(choices=list(SAMPLERS.keys()), value="Euler a", label="Sampler")
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
                    gr.Markdown("Paste a **Civitai download URL** or a **HuggingFace repo ID**.")
                    lora_url_t2i   = gr.Textbox(label="LoRA URL or HF Repo ID", placeholder="https://civitai.com/api/download/models/...")
                    lora_scale_t2i = gr.Slider(0.1, 1.5, step=0.05, value=0.8, label="LoRA Scale")
                    with gr.Row():
                        load_lora_t2i  = gr.Button("📥 Load LoRA", variant="primary")
                        clear_lora_t2i = gr.Button("🗑️ Clear LoRAs", variant="secondary")
                    lora_status_t2i = gr.Textbox(label="Status", interactive=False)
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
                    sampler_i2i  = gr.Dropdown(choices=list(SAMPLERS.keys()), value="Euler a", label="Sampler")
                    steps_i2i    = gr.Slider(1, 50, step=1, value=28, label="Steps")
                    cfg_i2i      = gr.Slider(1, 12, step=0.5, value=7.0, label="CFG Scale")
                    with gr.Row():
                        seed_i2i = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
                        rand_i2i = gr.Checkbox(label="Randomize", value=True)
                with gr.Accordion("🔍 Upscaler — AnimeSharp 4x", open=False):
                    use_up_i2i = gr.Checkbox(label="Enable Upscaler", value=False)
                    upby_i2i   = gr.Slider(1.1, 4.0, step=0.1, value=2.0, label="Upscale by")
                with gr.Accordion("🎛️ LoRA", open=False):
                    gr.Markdown("Paste a **Civitai download URL** or a **HuggingFace repo ID**.")
                    lora_url_i2i   = gr.Textbox(label="LoRA URL or HF Repo ID", placeholder="https://civitai.com/api/download/models/...")
                    lora_scale_i2i = gr.Slider(0.1, 1.5, step=0.05, value=0.8, label="LoRA Scale")
                    with gr.Row():
                        load_lora_i2i  = gr.Button("📥 Load LoRA", variant="primary")
                        clear_lora_i2i = gr.Button("🗑️ Clear LoRAs", variant="secondary")
                    lora_status_i2i = gr.Textbox(label="Status", interactive=False)
                gen_i2i = gr.Button("🎨 Generate", variant="primary", elem_id="gen-btn")

        with gr.Column(scale=3):
            output_gallery = gr.Gallery(label="Output", columns=2, height=580, preview=True, show_label=False)
            used_seed    = gr.Number(label="🌱 Seed Used", interactive=False)
            download_btn = gr.File(label="⬇️ Download PNG (full resolution)", interactive=False, visible=False)

    with gr.Row():
        civitai_key_input = gr.Textbox(
            label="🔑 Civitai API Key (needed for Civitai LoRA downloads)",
            placeholder="Paste your Civitai API key here",
            type="password",
        )

    gen_t2i.click(
        fn=txt2img,
        inputs=[model_input, prompt_t2i, neg_t2i, quality_t2i, add_q_t2i, style_t2i,
                w_t2i, h_t2i, cfg_t2i, steps_t2i, sampler_t2i, seed_t2i, rand_t2i,
                num_imgs_t2i, use_up_t2i, upby_t2i],
        outputs=[output_gallery, used_seed, download_btn],
    )
    gen_i2i.click(
        fn=img2img_fn,
        inputs=[model_input, input_img, prompt_i2i, neg_i2i, quality_i2i, add_q_i2i,
                style_i2i, strength_i2i, cfg_i2i, steps_i2i, sampler_i2i, seed_i2i,
                rand_i2i, use_up_i2i, upby_i2i],
        outputs=[output_gallery, used_seed, download_btn],
    )
    load_lora_t2i.click(fn=lambda u, s, k: load_lora_fn(u, s, k), inputs=[lora_url_t2i, lora_scale_t2i, civitai_key_input], outputs=lora_status_t2i)
    clear_lora_t2i.click(fn=lambda: manager.clear_loras(), outputs=lora_status_t2i)
    load_lora_i2i.click(fn=lambda u, s, k: load_lora_fn(u, s, k), inputs=[lora_url_i2i, lora_scale_i2i, civitai_key_input], outputs=lora_status_i2i)
    clear_lora_i2i.click(fn=lambda: manager.clear_loras(), outputs=lora_status_i2i)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(share=IS_COLAB, debug=IS_COLAB, allowed_paths=[OUTPUT_DIR, "./loras"])

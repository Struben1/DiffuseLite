import os
import gc
import random
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
    LCMScheduler,
)
from diffusers.models import AutoencoderKL
from huggingface_hub import hf_hub_download
import subprocess

# ── Constants ────────────────────────────────────────────────
IS_COLAB = bool(os.getenv("IS_COLAB", "0") == "1")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max

ASPECT_RATIOS = [
    "1024 x 1024",
    "1152 x 896",
    "896 x 1152",
    "1216 x 832",
    "832 x 1216",
    "1344 x 768",
    "768 x 1344",
    "1536 x 640",
    "640 x 1536",
    "Custom",
]

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
    "LCM":           lambda c: LCMScheduler.from_config(c),
}

MODELS = {
    "✨ Animagine XL 4.0":         "cagliostrolab/animagine-xl-4.0",
    "🎨 NoobAI XL 1.0":            "Laxhar/noobai-XL-1.0",
    "🎨 NoobAI XL Vpred 1.0":      "Laxhar/noobai-XL-Vpred-1.0",
    "🌸 Wai NSFW Illustrious v14": "John6666/wai-nsfw-illustrious-sdxl-v140-sdxl",
    "🖌️ Illustrious XL":           "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "🎯 Pony Diffusion v6 XL":     "kitty7779/ponyDiffusionV6XL",
    "🌟 SDXL Base 1.0":            "stabilityai/stable-diffusion-xl-base-1.0",
}

QUALITY_TAGS = {
    "Animagine v4 — High":    ("masterpiece, high score, great score, absurdres, highres", "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"),
    "Animagine v4 — Standard":("masterpiece, high score, great score", "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score"),
    "NoobAI — High":          ("masterpiece, best quality, very aesthetic, absurdres", "lowres, bad anatomy, bad hands, worst quality, low quality, very displeasing"),
    "NoobAI — Standard":      ("masterpiece, best quality", "lowres, worst quality, low quality"),
    "Pony — High":            ("score_9, score_8_up, score_7_up, masterpiece, best quality", "score_6, score_5, score_4, worst quality, low quality"),
    "None":                   ("", ""),
}

STYLES = {
    "(None)":           ("", ""),
    "Anime":            ("{prompt}, anime style", "photorealistic, 3d render"),
    "Illustration":     ("{prompt}, illustration, digital art", "photo, realistic"),
    "Watercolor":       ("{prompt}, watercolor painting", "digital art, 3d"),
    "Sketch":           ("{prompt}, pencil sketch, line art", "color, painted"),
    "Cinematic":        ("{prompt}, cinematic lighting, dramatic", "flat lighting"),
    "Neon":             ("{prompt}, neon lights, cyberpunk aesthetic", ""),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pipeline holder ──────────────────────────────────────────
class PipelineManager:
    def __init__(self):
        self.pipe = None
        self.current_model = None
        self.loras_loaded = []

    def load(self, model_key):
        model_id = MODELS.get(model_key, model_key)
        if self.current_model == model_id and self.pipe is not None:
            return
        print(f"[DiffuseLite] Loading {model_id}...")
        self.unload()

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

        load_fn = (
            StableDiffusionXLPipeline.from_single_file
            if model_id.endswith(".safetensors")
            else StableDiffusionXLPipeline.from_pretrained
        )

        self.pipe = load_fn(
            model_id,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        ).to(device)

        self.current_model = model_id
        self.loras_loaded = []
        print(f"[DiffuseLite] Loaded!")

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.current_model = None
            self.loras_loaded = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_lora(self, lora_id, lora_scale=0.8):
        if not self.pipe or not lora_id.strip():
            return "No model loaded or empty LoRA ID."
        try:
            # Support HF repo or local path
            if "/" in lora_id and not os.path.exists(lora_id):
                self.pipe.load_lora_weights(lora_id)
            else:
                self.pipe.load_lora_weights(lora_id)
            self.pipe.fuse_lora(lora_scale=lora_scale)
            self.loras_loaded.append(lora_id)
            return f"✅ LoRA loaded: {lora_id}"
        except Exception as e:
            return f"❌ Failed to load LoRA: {e}"

    def clear_loras(self):
        if not self.pipe:
            return "No model loaded."
        try:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self.loras_loaded = []
            return "✅ LoRAs cleared."
        except Exception as e:
            return f"❌ Error: {e}"


manager = PipelineManager()


# ── Helpers ──────────────────────────────────────────────────
def get_dimensions(aspect_ratio, custom_w, custom_h):
    if aspect_ratio == "Custom":
        return int(custom_w), int(custom_h)
    w, h = aspect_ratio.split(" x ")
    return int(w), int(h)


def apply_quality_tags(prompt, neg_prompt, quality_key, add_tags):
    if not add_tags or quality_key == "None":
        return prompt, neg_prompt
    pos_tags, neg_tags = QUALITY_TAGS[quality_key]
    new_prompt = f"{pos_tags}, {prompt}".strip(", ") if pos_tags else prompt
    new_neg = f"{neg_tags}, {neg_prompt}".strip(", ") if neg_tags else neg_prompt
    return new_prompt, new_neg


def apply_style(prompt, neg_prompt, style_key):
    if style_key == "(None)":
        return prompt, neg_prompt
    style_pos, style_neg = STYLES[style_key]
    new_prompt = style_pos.replace("{prompt}", prompt) if "{prompt}" in style_pos else f"{prompt}, {style_pos}"
    new_neg = f"{style_neg}, {neg_prompt}".strip(", ") if style_neg else neg_prompt
    return new_prompt, new_neg


def set_scheduler(pipe, sampler_name):
    pipe.scheduler = SAMPLERS[sampler_name](pipe.scheduler.config)


def save_image(image, prompt, seed, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_seed{seed}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("prompt", prompt)
    meta.add_text("seed", str(seed))
    meta.add_text("model", model_name)
    image.save(path, pnginfo=meta)
    return path


# ── Generation functions ─────────────────────────────────────
def txt2img(
    model_key,
    prompt,
    negative_prompt,
    quality_key,
    add_quality_tags,
    style_key,
    aspect_ratio,
    custom_width,
    custom_height,
    guidance_scale,
    num_steps,
    sampler,
    seed,
    randomize_seed,
    num_images,
    use_upscaler,
    upscale_by,
    upscaler_strength,
    progress=gr.Progress(track_tqdm=True),
):
    manager.load(model_key)
    pipe = manager.pipe

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    width, height = get_dimensions(aspect_ratio, custom_width, custom_height)

    prompt, negative_prompt = apply_quality_tags(prompt, negative_prompt, quality_key, add_quality_tags)
    prompt, negative_prompt = apply_style(prompt, negative_prompt, style_key)
    set_scheduler(pipe, sampler)

    try:
        if use_upscaler:
            upscaler = StableDiffusionXLImg2ImgPipeline(**pipe.components)
            latents = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                num_images_per_prompt=num_images,
                generator=generator,
                output_type="latent",
            ).images

            upscaled = torch.nn.functional.interpolate(
                latents, scale_factor=upscale_by, mode="nearest"
            )

            images = upscaler(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_steps * upscaler_strength),
                strength=upscaler_strength,
                generator=generator,
            ).images
            del upscaler
        else:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                num_images_per_prompt=num_images,
                generator=generator,
            ).images

        paths = [save_image(img, prompt, seed, model_key) for img in images]
        return paths, seed

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def img2img(
    model_key,
    input_image,
    prompt,
    negative_prompt,
    quality_key,
    add_quality_tags,
    style_key,
    strength,
    guidance_scale,
    num_steps,
    sampler,
    seed,
    randomize_seed,
    progress=gr.Progress(track_tqdm=True),
):
    manager.load(model_key)
    pipe = manager.pipe

    if input_image is None:
        raise gr.Error("Please upload an image.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(int(seed))

    prompt, negative_prompt = apply_quality_tags(prompt, negative_prompt, quality_key, add_quality_tags)
    prompt, negative_prompt = apply_style(prompt, negative_prompt, style_key)
    set_scheduler(pipe, sampler)

    img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)

    pil_image = Image.fromarray(input_image).convert("RGB")

    try:
        images = img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images

        paths = [save_image(img, prompt, seed, model_key) for img in images]
        return paths, seed

    finally:
        del img2img_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_lora_fn(lora_id, lora_scale):
    return manager.load_lora(lora_id, lora_scale)


def clear_loras_fn():
    return manager.clear_loras()


# ── Gradio UI ────────────────────────────────────────────────
CSS = """
#title { text-align: center; margin-bottom: 10px; }
#title h1 { font-size: 2.2em; }
#generate-btn { height: 50px; font-size: 1.1em; }
.output-gallery { min-height: 400px; }
"""

with gr.Blocks(theme="NoCrypt/miku@1.2.1", css=CSS) as demo:

    gr.HTML("<h1 id='title'>🎨 DiffuseLite</h1>")
    gr.Markdown("Lightweight anime image generation — txt2img, img2img, upscaler & LoRA support.")

    with gr.Row():
        model_selector = gr.Dropdown(
            label="🤖 Model",
            choices=list(MODELS.keys()),
            value=list(MODELS.keys())[0],
            interactive=True,
        )

    with gr.Row():
        # ── Left column ──────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tab("🖼️ Txt2Img"):
                prompt_t2i = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="1girl, solo, smile, looking at viewer, outdoors...",
                )
                neg_prompt_t2i = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    placeholder="lowres, bad anatomy, worst quality...",
                )

                with gr.Accordion("⭐ Quality & Style", open=True):
                    with gr.Row():
                        add_quality = gr.Checkbox(label="Add Quality Tags", value=True)
                        quality_selector = gr.Dropdown(
                            label="Quality Preset",
                            choices=list(QUALITY_TAGS.keys()),
                            value="Animagine v4 — High",
                        )
                    style_selector = gr.Radio(
                        label="Style Preset",
                        choices=list(STYLES.keys()),
                        value="(None)",
                    )

                with gr.Accordion("📐 Resolution", open=False):
                    aspect_ratio = gr.Radio(
                        label="Aspect Ratio",
                        choices=ASPECT_RATIOS,
                        value="896 x 1152",
                    )
                    with gr.Row(visible=False) as custom_res_row:
                        custom_w = gr.Slider(512, 2048, step=8, value=1024, label="Width")
                        custom_h = gr.Slider(512, 2048, step=8, value=1024, label="Height")

                with gr.Accordion("⚙️ Generation Settings", open=False):
                    with gr.Row():
                        sampler_t2i = gr.Dropdown(
                            label="Sampler",
                            choices=list(SAMPLERS.keys()),
                            value="Euler a",
                        )
                        num_steps_t2i = gr.Slider(1, 50, step=1, value=28, label="Steps")
                    with gr.Row():
                        cfg_t2i = gr.Slider(1, 12, step=0.5, value=7.0, label="CFG Scale")
                        num_images_t2i = gr.Slider(1, 4, step=1, value=1, label="Images")
                    with gr.Row():
                        seed_t2i = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
                        rand_seed_t2i = gr.Checkbox(label="Randomize", value=True)

                with gr.Accordion("🔍 Upscaler", open=False):
                    use_upscaler = gr.Checkbox(label="Enable Latent Upscaler", value=False)
                    with gr.Row():
                        upscale_by = gr.Slider(1.1, 2.0, step=0.1, value=1.5, label="Upscale by")
                        upscaler_strength = gr.Slider(0.1, 0.9, step=0.05, value=0.55, label="Strength")

                gen_btn_t2i = gr.Button("🎨 Generate", variant="primary", elem_id="generate-btn")

            with gr.Tab("🖼️➡️🖼️ Img2Img"):
                input_image = gr.Image(label="Input Image", type="numpy", height=300)
                prompt_i2i = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="1girl, solo, smile...",
                )
                neg_prompt_i2i = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    placeholder="lowres, worst quality...",
                )

                with gr.Accordion("⭐ Quality & Style", open=True):
                    with gr.Row():
                        add_quality_i2i = gr.Checkbox(label="Add Quality Tags", value=True)
                        quality_selector_i2i = gr.Dropdown(
                            label="Quality Preset",
                            choices=list(QUALITY_TAGS.keys()),
                            value="Animagine v4 — High",
                        )
                    style_selector_i2i = gr.Radio(
                        label="Style Preset",
                        choices=list(STYLES.keys()),
                        value="(None)",
                    )

                with gr.Accordion("⚙️ Settings", open=False):
                    strength_i2i = gr.Slider(0.1, 1.0, step=0.05, value=0.75, label="Denoising Strength", info="Higher = more changes")
                    with gr.Row():
                        sampler_i2i = gr.Dropdown(
                            label="Sampler",
                            choices=list(SAMPLERS.keys()),
                            value="Euler a",
                        )
                        num_steps_i2i = gr.Slider(1, 50, step=1, value=28, label="Steps")
                    with gr.Row():
                        cfg_i2i = gr.Slider(1, 12, step=0.5, value=7.0, label="CFG Scale")
                    with gr.Row():
                        seed_i2i = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
                        rand_seed_i2i = gr.Checkbox(label="Randomize", value=True)

                gen_btn_i2i = gr.Button("🎨 Generate", variant="primary", elem_id="generate-btn")

            with gr.Tab("🎛️ LoRA"):
                gr.Markdown("Load LoRAs by HuggingFace repo ID (e.g. `nerijs/animation2k-flux`) or local path.")
                lora_id = gr.Textbox(label="LoRA ID / Path", placeholder="nerijs/animation2k-flux")
                lora_scale = gr.Slider(0.1, 1.5, step=0.05, value=0.8, label="LoRA Scale")
                with gr.Row():
                    load_lora_btn = gr.Button("📥 Load LoRA", variant="primary")
                    clear_lora_btn = gr.Button("🗑️ Clear LoRAs", variant="secondary")
                lora_status = gr.Textbox(label="Status", interactive=False)

        # ── Right column ─────────────────────────────────────
        with gr.Column(scale=3):
            output_gallery = gr.Gallery(
                label="Output",
                columns=2,
                height=600,
                preview=True,
                show_label=False,
                elem_classes="output-gallery",
            )
            used_seed = gr.Number(label="Seed Used", interactive=False)

    # ── Event handlers ────────────────────────────────────────
    aspect_ratio.change(
        fn=lambda x: gr.update(visible=x == "Custom"),
        inputs=aspect_ratio,
        outputs=custom_res_row,
    )

    gen_btn_t2i.click(
        fn=txt2img,
        inputs=[
            model_selector,
            prompt_t2i, neg_prompt_t2i,
            quality_selector, add_quality,
            style_selector,
            aspect_ratio, custom_w, custom_h,
            cfg_t2i, num_steps_t2i, sampler_t2i,
            seed_t2i, rand_seed_t2i, num_images_t2i,
            use_upscaler, upscale_by, upscaler_strength,
        ],
        outputs=[output_gallery, used_seed],
    )

    gen_btn_i2i.click(
        fn=img2img,
        inputs=[
            model_selector,
            input_image,
            prompt_i2i, neg_prompt_i2i,
            quality_selector_i2i, add_quality_i2i,
            style_selector_i2i,
            strength_i2i,
            cfg_i2i, num_steps_i2i, sampler_i2i,
            seed_i2i, rand_seed_i2i,
        ],
        outputs=[output_gallery, used_seed],
    )

    load_lora_btn.click(fn=load_lora_fn, inputs=[lora_id, lora_scale], outputs=lora_status)
    clear_lora_btn.click(fn=clear_loras_fn, outputs=lora_status)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        share=IS_COLAB,
        debug=IS_COLAB,
        allowed_paths=[OUTPUT_DIR],
    )

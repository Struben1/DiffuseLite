import os
from argparse import ArgumentParser
from stablepy import (
    Model_Diffusers,
    SCHEDULE_TYPE_OPTIONS,
    SCHEDULE_PREDICTION_TYPE_OPTIONS,
    check_scheduler_compatibility,
    scheduler_names,
    PROMPT_WEIGHT_OPTIONS_PRIORITY,
)
from constants import (
    DIRECTORY_MODELS,
    DIRECTORY_LORAS,
    DIRECTORY_VAES,
    DIRECTORY_EMBEDS,
    DIRECTORY_UPSCALERS,
    DOWNLOAD_MODEL,
    DOWNLOAD_VAE,
    DOWNLOAD_LORA,
    LOAD_DIFFUSERS_FORMAT_MODEL,
    DIFFUSERS_FORMAT_LORAS,
    DOWNLOAD_EMBEDS,
    CIVITAI_API_KEY,
    HF_TOKEN,
    TASK_STABLEPY,
    TASK_MODEL_LIST,
    UPSCALER_DICT_GUI,
    UPSCALER_KEYS,
    WARNING_MSG_VAE,
    SDXL_TASK,
    MODEL_TYPE_TASK,
    POST_PROCESSING_SAMPLER,
    SUBTITLE_GUI,
    CACHE_HF_ROOT,
    CACHE_HF,
)
import torch
import re
import time
from PIL import ImageFile
from utils import (
    download_things,
    get_model_list,
    extract_parameters,
    get_my_lora,
    get_model_type,
    download_diffuser_repo,
    get_used_storage_gb,
    delete_model,
    progress_step_bar,
    html_template_message,
    escape_html,
    clear_hf_cache,
)
from datetime import datetime
import gradio as gr
import logging
import diffusers
import warnings
from stablepy import logger
import subprocess

IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
HIDE_API    = bool(os.getenv("HIDE_API"))
IS_COLAB    = bool(os.getenv("IS_COLAB", "0") == "1")

if IS_ZERO_GPU:
    subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)

IS_GPU_MODE = True if IS_ZERO_GPU else (True if torch.cuda.is_available() else False)
img_path    = "./images/"
allowed_path = os.path.abspath(img_path)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True

directories = [DIRECTORY_MODELS, DIRECTORY_LORAS, DIRECTORY_VAES, DIRECTORY_EMBEDS, DIRECTORY_UPSCALERS]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

for url in [url.strip() for url in DOWNLOAD_MODEL.split(',')]:
    download_things(DIRECTORY_MODELS, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in DOWNLOAD_VAE.split(',')]:
    download_things(DIRECTORY_VAES, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in DOWNLOAD_LORA.split(',')]:
    download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY)
for url_embed in DOWNLOAD_EMBEDS:
    download_things(DIRECTORY_EMBEDS, url_embed, HF_TOKEN, CIVITAI_API_KEY)

embed_list = get_model_list(DIRECTORY_EMBEDS)
embed_list = [(os.path.splitext(os.path.basename(emb))[0], emb) for emb in embed_list]

single_file_model_list = get_model_list(DIRECTORY_MODELS)
model_list = LOAD_DIFFUSERS_FORMAT_MODEL + single_file_model_list

lora_model_list = get_model_list(DIRECTORY_LORAS)
lora_model_list.insert(0, "None")
lora_model_list = lora_model_list + DIFFUSERS_FORMAT_LORAS

vae_model_list = get_model_list(DIRECTORY_VAES)
vae_model_list.insert(0, "BakedVAE")
vae_model_list.insert(0, "None")

print('\033[33m🏁 Download and listing of valid models completed.\033[0m')

logging.getLogger("diffusers").setLevel(logging.ERROR)
diffusers.utils.logging.set_verbosity(40)
warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")

parser = ArgumentParser(description='DiffuseLite')
parser.add_argument("--share", action="store_true", dest="share_enabled", default=False)
parser.add_argument('--theme', type=str, default="NoCrypt/miku")
parser.add_argument("--ssr", action="store_true")
parser.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args = parser.parse_args()

logger.setLevel("INFO" if IS_ZERO_GPU else getattr(logging, args.log_level.upper()))

CSS = """
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#gallery { flex-grow: 1; }
#load_model { height: 50px; }
"""


def lora_chk(lora_):
    if isinstance(lora_, str) and lora_.strip() not in ["", "None"]:
        return lora_
    return None


class GuiSD:
    def __init__(self):
        self.model          = None
        self.status_loading = False
        self.sleep_loading  = 4
        self.last_load      = datetime.now()
        self.inventory      = []

    def update_storage_models(self, storage_floor_gb=30, required_inventory_for_purge=3):
        while get_used_storage_gb() > storage_floor_gb:
            if len(self.inventory) < required_inventory_for_purge:
                break
            removal_candidate = self.inventory.pop(0)
            delete_model(removal_candidate)
        lowPrioCleanup = max((datetime.now() - self.last_load).total_seconds(), 0) > 60
        if (lowPrioCleanup and len(self.inventory) >= required_inventory_for_purge - 1
                and not self.status_loading
                and get_used_storage_gb(CACHE_HF_ROOT) > (storage_floor_gb * 2)):
            print("Cleaning up HF cache...")
            clear_hf_cache()
            self.inventory = [m for m in self.inventory if os.path.exists(m)]

    def update_inventory(self, model_name):
        if model_name not in single_file_model_list:
            self.inventory = [m for m in self.inventory if m != model_name] + [model_name]
        print(self.inventory)

    def load_new_model(self, model_name, vae_model, task, progress=gr.Progress(track_tqdm=True)):
        if model_name.startswith("http"):
            yield f"Downloading model: {model_name}"
            model_name = download_things(DIRECTORY_MODELS, model_name, HF_TOKEN, CIVITAI_API_KEY)
            if not model_name:
                raise ValueError("Error retrieving model information from URL")

        if IS_ZERO_GPU:
            self.update_storage_models()

        vae_model  = vae_model if vae_model != "None" else None
        model_type = get_model_type(model_name)
        dtype_model = torch.bfloat16 if model_type == "FLUX" else torch.float16

        if not os.path.exists(model_name):
            _ = download_diffuser_repo(repo_name=model_name, model_type=model_type,
                                       revision="main", token=True)

        self.update_inventory(model_name)

        for i in range(68):
            if not self.status_loading:
                self.status_loading = True
                if i > 0:
                    time.sleep(self.sleep_loading)
                    print("Previous model ops...")
                break
            time.sleep(0.5)
            print(f"Waiting queue {i}")
            yield "Waiting queue"

        self.status_loading = True
        yield f"Loading model: {model_name}"

        if vae_model == "BakedVAE":
            vae_model = model_name
        elif vae_model:
            vae_type = "SDXL" if "sdxl" in vae_model.lower() else "SD 1.5"
            if model_type != vae_type:
                gr.Warning(WARNING_MSG_VAE)

        try:
            start_time = time.time()
            if self.model is None:
                self.model = Model_Diffusers(
                    base_model_id=model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                    device="cpu" if IS_ZERO_GPU else None,
                )
                self.model.advanced_params(image_preprocessor_cuda_active=IS_GPU_MODE)
            else:
                if self.model.base_model_id != model_name:
                    load_now_time = datetime.now()
                    elapsed_time  = max((load_now_time - self.last_load).total_seconds(), 0)
                    if elapsed_time <= 9:
                        time.sleep(9 - elapsed_time)
                if IS_ZERO_GPU:
                    self.model.device = torch.device("cpu")
                self.model.load_pipe(
                    model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                )
            end_time = time.time()
            self.sleep_loading = max(min(int(end_time - start_time), 10), 4)
        except Exception as e:
            self.last_load      = datetime.now()
            self.status_loading = False
            self.sleep_loading  = 4
            raise e

        self.last_load      = datetime.now()
        self.status_loading = False
        yield f"Model loaded: {model_name}"

    @torch.inference_mode()
    def generate_pipeline(
        self,
        prompt, neg_prompt, num_images, steps, cfg, clip_skip, seed,
        lora1, lora_scale1, lora2, lora_scale2, lora3, lora_scale3,
        sampler, schedule_type, schedule_prediction_type,
        img_height, img_width, model_name, vae_model, task,
        image_control, strength,
        syntax_weights,
        upscaler_model_path, upscaler_increases_size,
        upscaler_tile_size, upscaler_tile_overlap,
        hires_steps, hires_denoising_strength, hires_sampler,
        hires_prompt, hires_negative_prompt,
        hires_before_adetailer, hires_after_adetailer,
        hires_schedule_type, hires_guidance_scale,
        loop_generation, leave_progress_bar, disable_progress_bar,
        image_previews, display_images,
        save_generated_images, filename_pattern, image_storage_location,
        retain_compel_previous_load, retain_hires_model_previous_load,
        generator_in_cpu, guidance_rescale,
        pag_scale,
    ):
        info_state = html_template_message("Navigating latent space...")
        yield info_state, gr.update(), gr.update()

        vae_model  = vae_model if vae_model != "None" else None
        vae_msg    = f"VAE: {vae_model}" if vae_model else ""
        msg_lora   = ""
        task       = TASK_STABLEPY[task]

        concurrency = 5
        self.model.stream_config(concurrency=concurrency, latent_resize_by=1, vae_decoding=False)

        if task != "txt2img" and not image_control:
            raise ValueError("Reference image is required for img2img.")

        if "https://" not in str(UPSCALER_DICT_GUI[upscaler_model_path]):
            upscaler_model = upscaler_model_path
        else:
            url_upscaler = UPSCALER_DICT_GUI[upscaler_model_path]
            if not os.path.exists(f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"):
                download_things(DIRECTORY_UPSCALERS, url_upscaler, HF_TOKEN)
            upscaler_model = f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"

        pipe_params = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "img_height": img_height,
            "img_width": img_width,
            "num_images": num_images,
            "num_steps": steps,
            "guidance_scale": cfg,
            "clip_skip": clip_skip,
            "pag_scale": float(pag_scale),
            "seed": seed,
            "image": image_control,
            "strength": strength,
            "lora_A": lora_chk(lora1), "lora_scale_A": lora_scale1,
            "lora_B": lora_chk(lora2), "lora_scale_B": lora_scale2,
            "lora_C": lora_chk(lora3), "lora_scale_C": lora_scale3,
            "syntax_weights": syntax_weights,
            "sampler": sampler,
            "schedule_type": schedule_type,
            "schedule_prediction_type": schedule_prediction_type,
            "gui_active": True,
            "loop_generation": loop_generation,
            "generator_in_cpu": generator_in_cpu,
            "leave_progress_bar": leave_progress_bar,
            "disable_progress_bar": disable_progress_bar,
            "image_previews": image_previews,
            "display_images": display_images,
            "save_generated_images": save_generated_images,
            "filename_pattern": filename_pattern,
            "image_storage_location": image_storage_location,
            "retain_compel_previous_load": retain_compel_previous_load,
            "retain_hires_model_previous_load": retain_hires_model_previous_load,
            "upscaler_model_path": upscaler_model,
            "upscaler_increases_size": upscaler_increases_size,
            "upscaler_tile_size": upscaler_tile_size,
            "upscaler_tile_overlap": upscaler_tile_overlap,
            "hires_steps": hires_steps,
            "hires_denoising_strength": hires_denoising_strength,
            "hires_prompt": hires_prompt,
            "hires_negative_prompt": hires_negative_prompt,
            "hires_sampler": hires_sampler,
            "hires_before_adetailer": hires_before_adetailer,
            "hires_after_adetailer": hires_after_adetailer,
            "hires_schedule_type": hires_schedule_type,
            "hires_guidance_scale": hires_guidance_scale,
        }

        if guidance_rescale:
            pipe_params["guidance_rescale"] = guidance_rescale

        if IS_ZERO_GPU:
            self.model.device = torch.device("cuda:0")

        actual_progress = 0
        info_images = gr.update()
        for img, [seed, image_path, metadata] in self.model(**pipe_params):
            info_state = progress_step_bar(actual_progress, steps)
            actual_progress += concurrency
            if image_path:
                info_images = f"Seeds: {str(seed)}"
                if vae_msg:
                    info_images += "<br>" + vae_msg
                for status, lora in zip(self.model.lora_status, self.model.lora_memory):
                    if status:
                        msg_lora += f"<br>Loaded: {lora}"
                    elif status is not None:
                        msg_lora += f"<br>Error with: {lora}"
                if msg_lora:
                    info_images += msg_lora
                info_images += "<br>GENERATION DATA:<br>" + escape_html(metadata[-1]) + "<br>-------<br>"
                download_links = "<br>".join([
                    f'<a href="{path.replace("/images/", f"/gradio_api/file={allowed_path}/")}" download="{os.path.basename(path)}">Download Image {i + 1}</a>'
                    for i, path in enumerate(image_path)
                ])
                if save_generated_images:
                    info_images += f"<br>{download_links}"
                info_state = "COMPLETE"
            yield info_state, img, info_images


sd_gen = GuiSD()

with gr.Blocks(theme=args.theme, css=CSS, fill_width=True, fill_height=False) as app:
    gr.Markdown("# 🎨 DiffuseLite")
    gr.Markdown(SUBTITLE_GUI)

    with gr.Tab("Generation"):
        with gr.Row():
            with gr.Column(scale=2):

                def update_task_options(model_name, task_name):
                    new_choices = MODEL_TYPE_TASK[get_model_type(model_name)]
                    if task_name not in new_choices:
                        task_name = "txt2img"
                    return gr.update(value=task_name, choices=new_choices)

                task_gui       = gr.Dropdown(label="Task", choices=SDXL_TASK, value=TASK_MODEL_LIST[0])
                model_name_gui = gr.Dropdown(label="Model", choices=model_list, value=model_list[0], allow_custom_value=True)
                prompt_gui     = gr.Textbox(lines=5, placeholder="Enter prompt", label="Prompt")
                neg_prompt_gui = gr.Textbox(lines=3, placeholder="Enter Neg prompt", label="Negative prompt",
                                            value="lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, worst quality, low quality, very displeasing, (bad)")
                with gr.Row(equal_height=False):
                    set_params_gui  = gr.Button(value="↙️", variant="secondary", size="sm")
                    clear_prompt_gui = gr.Button(value="🗑️", variant="secondary", size="sm")
                    set_random_seed  = gr.Button(value="🎲", variant="secondary", size="sm")
                generate_button = gr.Button(value="GENERATE IMAGE", variant="primary")

                model_name_gui.change(update_task_options, [model_name_gui, task_gui], [task_gui])

                load_model_gui = gr.HTML(elem_id="load_model", elem_classes="contain")
                result_images  = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery",
                    columns=[2], rows=[2], object_fit="contain",
                    interactive=False, preview=False, selected_index=50,
                )
                actual_task_info = gr.HTML()

                with gr.Row(equal_height=False, variant="default", visible=IS_ZERO_GPU):
                    gpu_duration_gui = gr.Number(minimum=5, maximum=240, value=59,
                                                  show_label=False, container=False,
                                                  info="GPU time duration (seconds)")
                    with gr.Column():
                        verbose_info_gui  = gr.Checkbox(value=False, container=False, label="Status info")
                        load_lora_cpu_gui = gr.Checkbox(value=False, container=False, label="Load LoRAs on CPU")

            with gr.Column(scale=1):
                steps_gui         = gr.Slider(minimum=1, maximum=100, step=1, value=28, label="Steps")
                cfg_gui           = gr.Slider(minimum=0, maximum=30, step=0.5, value=7., label="CFG")
                sampler_gui       = gr.Dropdown(label="Sampler", choices=scheduler_names, value="Euler")
                schedule_type_gui = gr.Dropdown(label="Schedule type", choices=SCHEDULE_TYPE_OPTIONS, value=SCHEDULE_TYPE_OPTIONS[0])
                img_width_gui     = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Width")
                img_height_gui    = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Height")
                seed_gui          = gr.Number(minimum=-1, maximum=9999999999, value=-1, label="Seed")
                pag_scale_gui     = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=0.0, label="PAG Scale")
                with gr.Row():
                    clip_skip_gui = gr.Checkbox(value=True, label="Layer 2 Clip Skip")

                with gr.Row(equal_height=False):
                    def run_set_params_gui(base_prompt, name_model):
                        valid_receptors = {
                            "prompt": gr.update(value=base_prompt),
                            "neg_prompt": gr.update(value=""),
                            "Steps": gr.update(value=30),
                            "width": gr.update(value=1024),
                            "height": gr.update(value=1024),
                            "Seed": gr.update(value=-1),
                            "Sampler": gr.update(value="Euler"),
                            "CFG scale": gr.update(value=7.),
                            "Clip skip": gr.update(value=True),
                            "Model": gr.update(value=name_model),
                            "Schedule type": gr.update(value="Automatic"),
                            "PAG": gr.update(value=.0),
                            "Hires upscaler": gr.update(),
                            "Hires upscale": gr.update(),
                            "Hires steps": gr.update(),
                            "Hires denoising strength": gr.update(),
                            "Hires CFG": gr.update(),
                            "Hires sampler": gr.update(),
                            "Hires schedule type": gr.update(),
                            "Strength": gr.update(),
                        }
                        for i in range(1, 4):
                            valid_receptors[f"Lora_{i}"] = gr.update()
                            valid_receptors[f"Lora_scale_{i}"] = gr.update()

                        valid_keys = list(valid_receptors.keys())
                        parameters = extract_parameters(base_prompt)

                        if "Sampler" in parameters:
                            value_sampler = parameters["Sampler"]
                            for s_type in SCHEDULE_TYPE_OPTIONS:
                                if s_type in value_sampler:
                                    value_sampler = value_sampler.replace(s_type, "").strip()
                                    parameters["Sampler"]       = value_sampler
                                    parameters["Schedule type"] = s_type

                        params_lora = []
                        if ">" in parameters["prompt"] and "<" in parameters["prompt"]:
                            params_lora = re.findall(r'<lora:[^>]+>', parameters["prompt"])

                        if params_lora:
                            parsed_params = []
                            for tag_l in params_lora:
                                try:
                                    inner   = tag_l.strip("<>")
                                    _, data = inner.split(":", 1)
                                    parts   = data.split(":")
                                    parsed_params.append((parts[0], float(parts[1]) if len(parts) > 1 else 1.0))
                                except Exception:
                                    pass
                            num_lora = 1
                            for parsed_l, parsed_s in parsed_params:
                                filtered = [m for m in lora_model_list if parsed_l in m]
                                if filtered and num_lora <= 3:
                                    parameters[f"Lora_{num_lora}"]       = filtered[0]
                                    parameters[f"Lora_scale_{num_lora}"] = parsed_s
                                    num_lora += 1

                        for key, val in parameters.items():
                            if key in valid_keys:
                                try:
                                    if key == "Sampler" and val not in scheduler_names:
                                        continue
                                    if key in ["Schedule type", "Hires schedule type"] and val not in SCHEDULE_TYPE_OPTIONS:
                                        continue
                                    if key == "Hires sampler" and val not in POST_PROCESSING_SAMPLER:
                                        continue
                                    elif key == "Clip skip":
                                        if "," in str(val):
                                            val = val.replace(",", "")
                                        if int(val) >= 2:
                                            val = True
                                    if key == "prompt":
                                        if ">" in val and "<" in val:
                                            val = re.sub(r'<[^>]+>', '', val)
                                    if key in ["prompt", "neg_prompt"]:
                                        val = re.sub(r'\s+', ' ', re.sub(r',+', ',', val)).strip()
                                    if key in ["Steps", "width", "height", "Seed", "Hires steps"]:
                                        val = int(val)
                                    if key == "PAG":
                                        val = .0
                                    if key in ["CFG scale", "Hires upscale", "Hires denoising strength", "Hires CFG", "Strength"]:
                                        val = float(val)
                                    if key == "Model":
                                        filtered = [m for m in model_list if val in m]
                                        val = filtered[0] if filtered else name_model
                                    if key == "Hires upscaler" and val not in UPSCALER_KEYS:
                                        continue
                                    if key == "Seed":
                                        continue
                                    valid_receptors[key] = gr.update(value=val)
                                except Exception as e:
                                    print(str(e))
                        return [v for v in valid_receptors.values()]

                    def run_clear_prompt_gui():
                        return gr.update(value=""), gr.update(value="")
                    clear_prompt_gui.click(run_clear_prompt_gui, [], [prompt_gui, neg_prompt_gui])

                    def run_set_random_seed():
                        return -1
                    set_random_seed.click(run_set_random_seed, [], seed_gui)

                num_images_gui  = gr.Slider(minimum=1, maximum=(8 if IS_ZERO_GPU else 20), step=1, value=1, label="Images")
                prompt_syntax_gui = gr.Dropdown(label="Prompt Syntax", choices=PROMPT_WEIGHT_OPTIONS_PRIORITY, value=PROMPT_WEIGHT_OPTIONS_PRIORITY[0])
                vae_model_gui   = gr.Dropdown(label="VAE Model", choices=vae_model_list, value=vae_model_list[0])

                with gr.Accordion("Hires fix", open=False):
                    upscaler_model_path_gui      = gr.Dropdown(label="Upscaler", choices=UPSCALER_KEYS, value=UPSCALER_KEYS[0])
                    upscaler_increases_size_gui  = gr.Slider(minimum=1.1, maximum=4., step=0.1, value=1.2, label="Upscale by")
                    upscaler_tile_size_gui       = gr.Slider(minimum=0, maximum=512, step=16, value=(0 if IS_ZERO_GPU else 192), label="Upscaler Tile Size")
                    upscaler_tile_overlap_gui    = gr.Slider(minimum=0, maximum=48, step=1, value=8, label="Upscaler Tile Overlap")
                    hires_steps_gui              = gr.Slider(minimum=0, value=30, maximum=100, step=1, label="Hires Steps")
                    hires_denoising_strength_gui = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.55, label="Hires Denoising Strength")
                    hires_sampler_gui            = gr.Dropdown(label="Hires Sampler", choices=POST_PROCESSING_SAMPLER, value=POST_PROCESSING_SAMPLER[0])
                    hires_schedule_list          = ["Use same schedule type"] + SCHEDULE_TYPE_OPTIONS
                    hires_schedule_type_gui      = gr.Dropdown(label="Hires Schedule type", choices=hires_schedule_list, value=hires_schedule_list[0])
                    hires_guidance_scale_gui     = gr.Slider(minimum=-1., maximum=30., step=0.5, value=-1., label="Hires CFG")
                    hires_prompt_gui             = gr.Textbox(label="Hires Prompt", placeholder="Main prompt will be used", lines=3)
                    hires_negative_prompt_gui    = gr.Textbox(label="Hires Negative Prompt", placeholder="Main negative prompt will be used", lines=3)

                with gr.Accordion("LoRA", open=False):
                    def lora_dropdown(label, visible=True):
                        return gr.Dropdown(label=label, choices=lora_model_list, value="None", allow_custom_value=True, visible=visible)
                    def lora_scale_slider(label, visible=True):
                        val_lora = 8 if IS_ZERO_GPU else 10
                        return gr.Slider(minimum=-val_lora, maximum=val_lora, step=0.01, value=0.33, label=label, visible=visible)

                    lora1_gui       = lora_dropdown("Lora1")
                    lora_scale_1_gui = lora_scale_slider("Lora Scale 1")
                    lora2_gui       = lora_dropdown("Lora2")
                    lora_scale_2_gui = lora_scale_slider("Lora Scale 2")
                    lora3_gui       = lora_dropdown("Lora3")
                    lora_scale_3_gui = lora_scale_slider("Lora Scale 3")

                    with gr.Accordion("From URL", open=False):
                        text_lora    = gr.Textbox(label="LoRA download URL",
                                                   placeholder="https://civitai.com/api/download/models/28907", lines=1)
                        button_lora  = gr.Button("Get and Refresh LoRA List")
                        new_lora_status = gr.HTML()
                        button_lora.click(
                            get_my_lora, [text_lora, gr.State(False)],
                            [lora1_gui, lora2_gui, lora3_gui, new_lora_status]
                        )

                with gr.Accordion("Img2img", open=False):
                    image_control = gr.Image(label="Input Image (img2img)", type="filepath")
                    strength_gui  = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.55, label="Strength")

                with gr.Accordion("Other settings", open=False):
                    schedule_prediction_type_gui = gr.Dropdown(label="Discrete Sampling Type", choices=SCHEDULE_PREDICTION_TYPE_OPTIONS, value=SCHEDULE_PREDICTION_TYPE_OPTIONS[0])
                    guidance_rescale_gui         = gr.Number(label="CFG rescale:", value=0., step=0.01, minimum=0., maximum=1.5)
                    save_generated_images_gui    = gr.Checkbox(value=True, label="Create download links for images")
                    filename_pattern_gui         = gr.Textbox(label="Filename pattern", value="model,seed", lines=1)
                    hires_before_adetailer_gui   = gr.Checkbox(value=False, label="Hires Before Adetailer", visible=False)
                    hires_after_adetailer_gui    = gr.Checkbox(value=True,  label="Hires After Adetailer",  visible=False)
                    generator_in_cpu_gui         = gr.Checkbox(value=False, label="Generator in CPU")
                    with gr.Column(visible=(not IS_ZERO_GPU)):
                        image_storage_location_gui = gr.Textbox(value=img_path, label="Image Storage Location")
                        disable_progress_bar_gui   = gr.Checkbox(value=False, label="Disable Progress Bar")
                        leave_progress_bar_gui     = gr.Checkbox(value=True,  label="Leave Progress Bar")

                with gr.Accordion("More settings", open=False, visible=False):
                    loop_generation_gui              = gr.Slider(minimum=1, value=1, label="Loop Generation")
                    display_images_gui               = gr.Checkbox(value=False, label="Display Images")
                    image_previews_gui               = gr.Checkbox(value=True,  label="Image Previews")
                    retain_compel_previous_load_gui  = gr.Checkbox(value=False, label="Retain Compel Previous Load")
                    retain_hires_model_previous_load_gui = gr.Checkbox(value=False, label="Retain Hires Model Previous Load")

                set_params_gui.click(
                    run_set_params_gui, [prompt_gui, model_name_gui], [
                        prompt_gui, neg_prompt_gui, steps_gui, img_width_gui, img_height_gui,
                        seed_gui, sampler_gui, cfg_gui, clip_skip_gui, model_name_gui,
                        schedule_type_gui, pag_scale_gui,
                        upscaler_model_path_gui, upscaler_increases_size_gui,
                        hires_steps_gui, hires_denoising_strength_gui,
                        hires_guidance_scale_gui, hires_sampler_gui, hires_schedule_type_gui,
                        strength_gui,
                        lora1_gui, lora_scale_1_gui,
                        lora2_gui, lora_scale_2_gui,
                        lora3_gui, lora_scale_3_gui,
                    ],
                )

    generate_button.click(
        fn=sd_gen.load_new_model,
        inputs=[model_name_gui, vae_model_gui, task_gui],
        outputs=[load_model_gui],
        queue=True, show_progress="minimal",
        api_name=(False if HIDE_API else None),
    ).success(
        fn=sd_gen.generate_pipeline,
        inputs=[
            prompt_gui, neg_prompt_gui, num_images_gui, steps_gui, cfg_gui,
            clip_skip_gui, seed_gui,
            lora1_gui, lora_scale_1_gui,
            lora2_gui, lora_scale_2_gui,
            lora3_gui, lora_scale_3_gui,
            sampler_gui, schedule_type_gui, schedule_prediction_type_gui,
            img_height_gui, img_width_gui, model_name_gui, vae_model_gui, task_gui,
            image_control, strength_gui,
            prompt_syntax_gui,
            upscaler_model_path_gui, upscaler_increases_size_gui,
            upscaler_tile_size_gui, upscaler_tile_overlap_gui,
            hires_steps_gui, hires_denoising_strength_gui, hires_sampler_gui,
            hires_prompt_gui, hires_negative_prompt_gui,
            hires_before_adetailer_gui, hires_after_adetailer_gui,
            hires_schedule_type_gui, hires_guidance_scale_gui,
            loop_generation_gui, leave_progress_bar_gui, disable_progress_bar_gui,
            image_previews_gui, display_images_gui,
            save_generated_images_gui, filename_pattern_gui, image_storage_location_gui,
            retain_compel_previous_load_gui, retain_hires_model_previous_load_gui,
            generator_in_cpu_gui, guidance_rescale_gui,
            pag_scale_gui,
        ],
        outputs=[load_model_gui, result_images, actual_task_info],
        queue=True, show_progress="minimal",
    )

if __name__ == "__main__":
    app.queue()
    app.launch(
        show_error=True,
        share=args.share_enabled or IS_COLAB,
        debug=True,
        ssr_mode=args.ssr,
        allowed_paths=[allowed_path],
        show_api=(not HIDE_API),
    )

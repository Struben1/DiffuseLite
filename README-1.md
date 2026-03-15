# 🎨 DiffuseLite

A lightweight, clean anime image generation app powered by Stable Diffusion XL.

Inspired by [Animagine XL 4.0](https://huggingface.co/spaces/Asahina2K/animagine-xl-4.0) — built to run on **free Google Colab T4 GPU**.

---

## ✨ Features

- **Txt2Img** — Generate images from text prompts
- **Img2Img** — Transform existing images with a prompt
- **Latent Upscaler** — Built-in upscaling after generation
- **LoRA Support** — Load any HuggingFace LoRA on the fly
- **Multiple Models** — Switch between Animagine XL 4.0, NoobAI, Pony, and more
- **Quality Tag Presets** — Auto-adds quality tags for best results
- **Style Presets** — Anime, illustration, watercolor, cinematic, and more
- **Aspect Ratio Presets** — Common portrait/landscape ratios
- **Sampler Selection** — Euler a, DPM++, DDIM, LCM, and more

---

## 🚀 Quick Start (Google Colab)

Open `DiffuseLite_Colab.ipynb` in Google Colab and run all cells.

---

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Main Gradio application |
| `requirements.txt` | Python dependencies |
| `DiffuseLite_Colab.ipynb` | Ready-to-use Colab notebook |

---

## 💡 Prompt Tips

For best results with anime models, use Danbooru-style tags:

```
1girl, character name, series name, solo, smile, looking at viewer, outdoors, masterpiece, high score, great score, absurdres
```

Negative prompt:
```
lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality
```

---

## 📜 License

This project uses models with their respective licenses. Please check each model's license on HuggingFace before use.

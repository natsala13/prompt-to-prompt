import abc

import torch
import numpy as np
import seq_aligner
from PIL import Image
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
from typing import Union, Tuple, List, Callable, Dict, Optional

import ptp_utils
from controllers import AttentionControl, AttentionStore, AttentionReplace, EmptyControl


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_id = "CompVis/ldm-text2im-large-256"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 5.
MAX_NUM_WORDS = 77
# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id).to(device)
tokenizer = ldm.tokenizer


def aggregate_attention(attention_store: AttentionStore, prompts: [str], res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, prompts: [str], res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention_comp(attention_store: AttentionStore, prompts: [str], res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def sort_by_eq(eq):
    def inner_(images):
        swap = 0
        if eq[-1] < 1:
            for i in range(len(eq)):
                if eq[i] > 1 and eq[i + 1] < 1:
                    swap = i + 2
                    break
        else:
            for i in range(len(eq)):
                if eq[i] < 1 and eq[i + 1] > 1:
                    swap = i + 2
                    break
        print(swap)
        if swap > 0:
            images = np.concatenate([images[1:swap], images[:1], images[swap:]], axis=0)

        return images

    return inner_


def run_and_display(prompts, controller, latent=None, run_baseline=True,
                    callback: Optional[Callable[[np.ndarray], np.ndarray]] = None, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False)
        print("results with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm(ldm, prompts, controller, latent=latent,
                                           num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE,
                                           generator=generator)
    if callback is not None:
        images = callback(images)
    ptp_utils.view_images(images)
    return images, x_t


def main():
    g_cpu = torch.Generator().manual_seed(888)
    prompts = ["A painting of a squirrel eating a burger"]
    controller = AttentionStore()
    images, x_t = run_and_display(prompts, controller, run_baseline=False, generator=g_cpu)
    # show_cross_attention(controller, prompts, res=16, from_where=["up", "down"])

    prompts = ["A painting of a squirrel eating a burger",
               "A painting of a lion eating a burger",
               "A painting of a cat eating a burger",
               "A painting of a deer eating a burger",
               ]
    controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=.2)
    _ = run_and_display(prompts, controller, latent=x_t, run_baseline=True)


main()

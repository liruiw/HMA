import os

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as transforms_f
from diffusers import AutoencoderKLTemporalDecoder
from einops import rearrange
from transformers import T5Tokenizer, T5Model

from external.magvit2.config import VQConfig
from external.magvit2.models.lfqgan import VQModel

vision_model = None


def get_image_encoder(encoder_type: str, encoder_name_or_path: str):
    encoder_type = encoder_type.lower()
    if encoder_type == "magvit":
        return VQModel(VQConfig(), ckpt_path=encoder_name_or_path)
    elif encoder_type == "temporalvae":
        return AutoencoderKLTemporalDecoder.from_pretrained(encoder_name_or_path, subfolder="vae")
    else:
        raise NotImplementedError(f"{encoder_type=}")


def set_seed(seed):
    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)


def mkdir_if_missing(dst_dir):
    """make destination folder if it's missing"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def resize_image(image, resize=True):
    MAX_RES = 1024

    # convert to array
    image = np.asarray(image)
    h, w = image.shape[:2]
    if h > MAX_RES or w > MAX_RES:
        if h < w:
            new_h, new_w = int(MAX_RES * w / h), MAX_RES
        else:
            new_h, new_w = MAX_RES, int(MAX_RES * h / w)
        image = cv2.resize(image, (new_w, new_h))

    if resize:
        # resize the shorter side to 256 and then do a center crop
        h, w = image.shape[:2]
        if h < w:
            new_h, new_w = 256, int(256 * w / h)
        else:
            new_h, new_w = int(256 * h / w), 256
        image = cv2.resize(image, (new_w, new_h))

        h, w = image.shape[:2]
        crop_h, crop_w = 256, 256
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        image = image[start_h : start_h + crop_h, start_w : start_w + crop_w]
    return image


def normalize_image(image, resize=True):
    """
    H x W x 3(uint8) -> imagenet normalized 3 x H x W

    Normalizes image to [-1, 1].
    Resizes the image if resize=True or if the image resolution > MAX_RES
    """
    image = resize_image(image, resize=resize)
    # normalize between -1 and 1
    image = image / 255.0
    image = image * 2 - 1.0
    return torch.from_numpy(image.transpose(2, 0, 1))


def unnormalize_image(magvit_output):
    """
    [-1, 1] -> [0, 255]

    Important: clip to [0, 255]
    """
    rescaled_output = (magvit_output.detach().cpu() + 1) * 127.5
    clipped_output = torch.clamp(rescaled_output, 0, 255).to(dtype=torch.uint8)
    return clipped_output


@torch.inference_mode()
@torch.no_grad()
def get_quantized_image_embeddings(
    image,
    encoder_type,
    encoder_name_or_path,
    keep_res=False,
    device="cuda",
):
    """
    image: (h, w, 3)
    """
    global vision_model
    DEBUG = False
    dtype = torch.bfloat16

    if vision_model is None:
        vision_model = get_image_encoder(encoder_type=encoder_type, encoder_name_or_path=encoder_name_or_path)
        vision_model = vision_model.to(device=device, dtype=dtype)
        vision_model.eval()

    batch = normalize_image(image, resize=not keep_res)[None]
    if not keep_res:
        img_h, img_w = 256, 256
    else:
        img_h, img_w = batch.shape[2:]

    h, w = img_h // 16, img_w // 16

    with vision_model.ema_scope():
        quant_, _, indices, _ = vision_model.encode(batch.to(device=device, dtype=dtype), flip=True)
    indices = rearrange(indices, "(h w) -> h w", h=h, w=w)

    # alternative way to get indices
    # indices_ = vision_model.quantize.bits_to_indices(quant_.permute(0, 2, 3, 1) > 0).cpu().numpy()
    # indices_ = rearrange(indices_, "(h w) -> h w", h=h, w=w)

    if DEBUG:
        # sanity check: decode and then visualize
        with vision_model.ema_scope():
            indices = indices[None]
            # bit representations
            quant = vision_model.quantize.get_codebook_entry(
                rearrange(indices, "b h w -> b (h w)"), bhwc=indices.shape + (vision_model.quantize.codebook_dim,)
            ).flip(1)
            ##  why is there a flip(1) needed for the codebook bits?
            decoded_img = unnormalize_image(vision_model.decode(quant.to(device=device, dtype=dtype)))
            transforms_f.to_pil_image(decoded_img[0]).save("decoded.png")
            transforms_f.to_pil_image(image).save("original.png")  # show()

    # 18 x 16 x 16 of [-1., 1.] - > 16 x 16 uint32
    indices = indices.type(torch.int32)
    indices = indices.detach().cpu().numpy().astype(np.uint32)
    return indices


@torch.inference_mode()
@torch.no_grad()
def get_vae_image_embeddings(
    image,
    encoder_type,
    encoder_name_or_path,
    keep_res: bool = False,
    device="cuda",
):
    """
    image: (h, w, 3), in [-1, 1]
    use SD VAE to encode and decode the images.
    """
    global vision_model
    DEBUG = False
    dtype = torch.bfloat16

    if vision_model is None:
        vision_model = get_image_encoder(encoder_type, encoder_name_or_path)
        vision_model = vision_model.to(device=device, dtype=dtype)
        vision_model.eval()

    # https://github.com/bytedance/IRASim/blob/main/sample/sample_autoregressive.py#L151
    # if args.use_temporal_decoder:
    #     vae = AutoencoderKLTemporalDecoder.from_pretrained(args.vae_model_path, subfolder="t2v_required_models/vae_temporal_decoder").to(device)
    # else:
    #     vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)
    #  x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor) ?

    batch = normalize_image(image, resize=not keep_res)[None]

    if isinstance(vision_model, AutoencoderKLTemporalDecoder):
        # Think SVD expects images in [-1, 1] so we don't have to change anything?
        # https://github.com/Stability-AI/generative-models/blob/1659a1c09b0953ad9cc0d480f42e4526c5575b37/scripts/demo/video_sampling.py#L182
        # https://github.com/Stability-AI/generative-models/blob/1659a1c09b0953ad9cc0d480f42e4526c5575b37/scripts/demo/streamlit_helpers.py#L894
        z = vision_model.encode(batch.to(device=device, dtype=dtype)).latent_dist.mean
    elif isinstance(vision_model, VQModel):  # vision_model should be VQModel
        # with vision_model.ema_scope():  # doesn't matter due to bugged VQModel ckpt_path arg
        z = vision_model.encode_without_quantize(batch.to(device=device, dtype=dtype))
    else:
        raise NotImplementedError(f"{vision_model=}")

    if DEBUG:
        decoded_img = unnormalize_image(vision_model.decode(z.to(device=device, dtype=dtype)))
        transforms_f.to_pil_image(decoded_img[0]).save("decoded_unquant.png")
        transforms_f.to_pil_image(image).save("original.png")

    return z[0].detach().cpu().float().numpy().astype(np.float16)

    # switch to VAE in SD
    # https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main/vae
    # https://github.com/bytedance/IRASim/blob/main/sample/sample_autoregressive.py#L151
    # from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
    # vae_model_path = 'pretrained_models/stabilityai/stable-diffusion-xl-base-1.0'
    # if args.use_temporal_decoder:
    #     vae = AutoencoderKLTemporalDecoder.from_pretrained(vae_model_path, subfolder="t2v_required_models/vae_temporal_decoder").to(device)
    # else:
    #     vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae").to(device)
    #  z = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
    # if DEBUG:
    #     decoded_img = unnormalize_image(vae.decode(z.to(device=device, dtype=dtype) / vae.config.scaling_factor))
    #     transforms_f.to_pil_image(decoded_img[0]).save("decoded_unquant.png")
    #     transforms_f.to_pil_image(image).save("original.png")


@torch.no_grad()
def get_t5_embeddings(language, per_token=True, max_length=16, device="cpu"):
    """Get T5 embedding"""
    global global_language_model, t5_tok
    if global_language_model is None:
        try:
            t5_model = T5Model.from_pretrained("t5-base")
            t5_tok = T5Tokenizer.from_pretrained("t5-base")
        except:
            t5_model = T5Model.from_pretrained("t5-base", local_files_only=True)
            t5_tok = T5Tokenizer.from_pretrained("t5-base", local_files_only=True)
        t5_model = t5_model.to(device)
        global_language_model = t5_model
        global_language_model.eval()

    # forward pass through encoder only
    enc = t5_tok(
        [language],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    output = global_language_model.encoder(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True
    )
    torch.cuda.empty_cache()
    if per_token:
        return output.last_hidden_state[0].detach().cpu().numpy()
    else:
        # get the final hidden states. average across tokens.
        emb = output.last_hidden_state[0].mean(dim=0).detach().cpu().numpy()
        return emb

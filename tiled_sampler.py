"""Spatially tiled sampling for MiniMax H3.

H3 denoises video and audio as one packed attention sequence, so its cost is
driven by the video token count (latent_t * h/2 * w/2) and its canvas is capped
at 768*1344 pixels. This sampler runs the DiT on overlapping spatial tiles of
the video stream: every tile is a full forward carrying the text, audio,
keyframe and reference rows, and the tile's video rows keep the position ids
they hold in the full packed layout, so RoPE coordinates stay absolute instead
of being renormalized to the tile. Tile velocities are blended with linear ramps
across the overlap, the audio velocity is averaged over the tiles.
"""

import logging
import math

import torch

import comfy.model_management
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import nodes
from comfy.ldm.minimax.model import PackedLayout
from comfy_api.latest import ComfyExtension, io

VAE_STRIDE = 16   # pixels per video latent cell
PATCH = 2         # DiT spatial patch, in latent cells
WRAPPER_KEY = "minimax_h3_tiled_sampler"


def tile_len(pixels, total):
    return min(max(pixels // VAE_STRIDE // PATCH * PATCH, PATCH), total)


def overlap_len(pixels, tile):
    return min(max(pixels // VAE_STRIDE // PATCH * PATCH, 0), tile - PATCH)


def tile_starts(total, tile, overlap):
    """Evenly spread tile origins covering total, aligned to the DiT patch grid."""
    count = math.ceil((total - overlap) / (tile - overlap))
    if count < 2:
        return [0]
    span = total - tile
    return sorted({i * span // (count - 1) // PATCH * PATCH for i in range(count)})


def tile_grid(video_shape, tile_width, tile_height, tile_overlap):
    lat_h, lat_w = video_shape[3], video_shape[4]
    th = tile_len(tile_height, lat_h)
    tw = tile_len(tile_width, lat_w)
    return (th, tw,
            tile_starts(lat_h, th, overlap_len(tile_overlap, th)),
            tile_starts(lat_w, tw, overlap_len(tile_overlap, tw)))


def frame_row_index(lat_w, h0, h1, w0, w1):
    """Row indices of a tile within one latent frame's 2x2-patch row order."""
    cols = lat_w // PATCH
    rows = torch.arange(h0 // PATCH, h1 // PATCH).unsqueeze(1) * cols
    return (rows + torch.arange(w0 // PATCH, w1 // PATCH)).reshape(-1)


def edge_ramp(length, head, tail, device):
    """Weights that fade in over the first head cells and out over the last tail cells."""
    w = torch.ones(length, dtype=torch.float32, device=device)
    if head > 0:
        w[:head] = torch.linspace(1.0 / (head + 1), 1.0, head, dtype=torch.float32, device=device)
    if tail > 0:
        w[length - tail:] = torch.linspace(1.0, 1.0 / (tail + 1), tail, dtype=torch.float32, device=device)
    return w


def neighbour_overlaps(starts, index, tile):
    head = starts[index - 1] + tile - starts[index] if index > 0 else 0
    tail = starts[index] + tile - starts[index + 1] if index + 1 < len(starts) else 0
    return head, tail


def blend_weights(th, tw, h_ramps, w_ramps, device):
    h = edge_ramp(th, *h_ramps, device)
    w = edge_ramp(tw, *w_ramps, device)
    return (h.unsqueeze(1) * w).view(1, 1, 1, th, tw)


def full_layout(payload, signature):
    layout = payload.get("layout")
    if layout is not None and layout.signature == signature:
        return layout
    return PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))


def tile_layout(full, full_rows, payload, signature, row_index):
    """Layout for one tile: row counts from the tile's own canvas, coordinates from the full one.

    Rebuilding the layout at tile size would renormalize the spatial grid, so the
    target rows (keyframe conditions and the video stream) instead take the slice
    of the full layout's position ids that belongs to the tile.
    """
    layout = PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    position_ids = []
    for start, stop, kind in full.segments:
        rows = full.position_ids[start:stop]
        if kind in ("cond", "video"):
            rows = rows.view(-1, full_rows, 3)[:, row_index].reshape(-1, 3)
        position_ids.append(rows)
    layout.position_ids = torch.cat(position_ids)
    return layout


def tile_payload(payload, layout, h0, h1, w0, w1):
    out = dict(payload)
    out["layout"] = layout
    cond_latents = payload.get("cond_video_latents")
    if cond_latents:
        # keyframe latents share the target canvas and are ordered ahead of the reference latents
        keyframes = sum(1 for kf in payload.get("keyframes") or () if kf.get("latent") is not None)
        out["cond_video_latents"] = [z[:, :, :, h0:h1, w0:w1] for z in cond_latents[:keyframes]] + cond_latents[keyframes:]
    return out


def tiled_diffusion_wrapper(tile_width, tile_height, tile_overlap):
    layouts = {}

    def wrapper(executor, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        video, audio = x[0], x[1]
        lat_t, lat_h, lat_w = video.shape[2], video.shape[3], video.shape[4]
        th, tw, h_starts, w_starts = tile_grid(video.shape, tile_width, tile_height, tile_overlap)
        if len(h_starts) * len(w_starts) == 1:
            return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

        payload = minimax_payload or {}
        text_len, audio_t = context.shape[1], audio.shape[-1]
        full = full_layout(payload, (text_len, lat_t, lat_h, lat_w, audio_t))
        full_rows = (lat_h // PATCH) * (lat_w // PATCH)

        video_out = torch.zeros(video.shape, dtype=torch.float32, device=video.device)
        audio_out = torch.zeros(audio.shape, dtype=torch.float32, device=audio.device)
        weights = torch.zeros((1, 1, 1, lat_h, lat_w), dtype=torch.float32, device=video.device)

        for i, h0 in enumerate(h_starts):
            for j, w0 in enumerate(w_starts):
                comfy.model_management.throw_exception_if_processing_interrupted()
                h1, w1 = h0 + th, w0 + tw
                key = (text_len, lat_t, lat_h, lat_w, audio_t, h0, w0, th, tw)
                if key not in layouts:
                    layouts[key] = tile_layout(full, full_rows, payload, (text_len, lat_t, th, tw, audio_t),
                                               frame_row_index(lat_w, h0, h1, w0, w1))
                out = executor([video[:, :, :, h0:h1, w0:w1], audio], timestep, context, transformer_options,
                               minimax_payload=tile_payload(payload, layouts[key], h0, h1, w0, w1), **kwargs)
                tile_weights = blend_weights(th, tw, neighbour_overlaps(h_starts, i, th),
                                             neighbour_overlaps(w_starts, j, tw), video.device)
                video_out[:, :, :, h0:h1, w0:w1].addcmul_(out[0].float(), tile_weights)
                weights[:, :, :, h0:h1, w0:w1] += tile_weights
                audio_out += out[1].float()

        video_out /= weights
        audio_out /= len(h_starts) * len(w_starts)
        return [video_out.to(video.dtype), audio_out.to(audio.dtype)]

    return wrapper


def prepare_sampling_wrapper(video_numel, tile_numel):
    def wrapper(executor, model, noise_shape, conds, *args, **kwargs):
        # H3 samples the flat AV pack, so budget VRAM for one tile's worth of video cells
        noise_shape = list(noise_shape)
        noise_shape[-1] += tile_numel - video_numel
        return executor(model, noise_shape, conds, *args, **kwargs)
    return wrapper


class MiniMaxH3TiledSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TiledSampler",
            display_name="MiniMax H3 Tiled Sampler",
            category="model/sampling/minimax",
            description="Denoise a MiniMax H3 AV latent in overlapping spatial tiles. Each tile is a full pass over a crop of the video stream, keeping a canvas larger than 768x1344 inside the model's native field of view and tying peak VRAM to the tile size instead of the canvas. Audio, text, keyframes and references ride along in every tile.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("tile_width", default=1344, min=64, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="Tile width in pixels, rounded down to 32. The default is the model's widest native canvas."),
                io.Int.Input("tile_height", default=768, min=64, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="Tile height in pixels, rounded down to 32. The default is the model's native short edge."),
                io.Int.Input("tile_overlap", default=192, min=0, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="Overlap between neighbouring tiles in pixels. More overlap hides tile seams and costs more tiles."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise,
                tile_width, tile_height, tile_overlap) -> io.NodeOutput:
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24:
            raise ValueError("MiniMaxH3TiledSampler expects a MiniMax H3 AV latent")
        video = samples.tensors[0]
        if video.shape[3] % PATCH or video.shape[4] % PATCH:
            raise ValueError("MiniMaxH3TiledSampler needs a latent whose width and height are multiples of 32 pixels")
        th, tw, h_starts, w_starts = tile_grid(video.shape, tile_width, tile_height, tile_overlap)

        model = model.clone()
        if len(h_starts) * len(w_starts) > 1:
            logging.info("MiniMax H3 tiled sampling: {}x{} tiles of {}x{} px".format(
                len(w_starts), len(h_starts), tw * VAE_STRIDE, th * VAE_STRIDE))
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                                       tiled_diffusion_wrapper(tile_width, tile_height, tile_overlap))
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY,
                                       prepare_sampling_wrapper(math.prod(video.shape[1:]),
                                                                math.prod(video.shape[1:3]) * th * tw))
        else:
            logging.info("MiniMax H3 tiled sampling: canvas fits in one tile, sampling it whole")

        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index", None))
        out = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                                  denoise=denoise, noise_mask=latent.get("noise_mask", None),
                                  callback=latent_preview.prepare_callback(model, steps),
                                  disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        return io.NodeOutput({**latent, "samples": out})


class MiniMaxH3TiledSamplerExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3TiledSampler]


async def comfy_entrypoint() -> MiniMaxH3TiledSamplerExtension:
    return MiniMaxH3TiledSamplerExtension()

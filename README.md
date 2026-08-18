# MiniMax H3 Tiled Sampler

## Performance

Upscaling a 15-second video to about 2 megapixels (1080p) with this tiled
sampler used roughly 75% of a 16 GB RTX 4060 Ti (~12 GB VRAM) and took 70
minutes.

A single ComfyUI node, **MiniMax H3 Tiled Sampler** (`model/sampling/minimax`), that
denoises a MiniMax H3 audio-video latent in overlapping spatial tiles.

H3 packs video and audio into one attention sequence, so its cost is driven by the
video token count (`latent_t * h/2 * w/2`) and its trained canvas stops at
768x1344 pixels. Tiling the video stream keeps every tile inside that native field
of view and ties peak VRAM to the tile size rather than to the canvas, which is
what makes larger canvases practical.

## Inputs

Same controls as `KSampler` (`seed`, `steps`, `cfg`, `sampler_name`, `scheduler`,
`denoise`), plus:

| Input | Meaning |
| --- | --- |
| `tile_width` / `tile_height` | Tile size in pixels, rounded down to 32. Defaults are the model's native canvas, 1344x768. |
| `tile_overlap` | Overlap between neighbouring tiles in pixels. Larger values hide seams and add tiles. |

`latent` must be an H3 AV latent, from `Empty MiniMax H3 AV Latent`,
`MiniMax H3 Image to Video` or `MiniMax H3 Reference to Video`. When the canvas
fits in a single tile the node samples it whole, so it is a drop-in replacement
for `KSampler` in an H3 workflow.

## How it works

Sampling itself is unchanged: the node clones the model and adds a
`DIFFUSION_MODEL` wrapper, then hands the latent to the stock sampler. Per model
call, the wrapper:

1. Cuts the video latent into overlapping tiles aligned to the DiT's 2x2 patch
   grid, pulling the last tile of each axis back so all tiles are the same size.
2. Runs one forward per tile. Text, audio, keyframe and reference rows ride along
   in every tile, so audio stays global and guides keep working.
3. Gives each tile's rows the position ids they hold in the full packed layout,
   so RoPE coordinates stay absolute. Rebuilding the layout at tile size would
   renormalize the spatial grid and make the model read a tile as a whole frame.
4. Blends tile video velocities with linear ramps over the overlap and averages
   the audio velocities.

A `PREPARE_SAMPLING` wrapper shrinks the VRAM estimate to one tile's worth of
video cells, so the model is not offloaded for a sequence length that is never
built.

## Notes

- Batch size is 1, as with H3 itself.
- Latent width and height must be multiples of 32 pixels, which every H3 latent
  node already guarantees.
- Attention cannot cross tile borders, so global spatial coherence weakens as
  tiles get smaller. Keep tiles at the native canvas size and raise
  `tile_overlap` if you see seams.
- Keyframe conditions are cropped per tile, so the 0.1% condition noise
  augmentation H3 applies by default is drawn per tile instead of once for the
  whole canvas.

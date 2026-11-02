# ComfyUI-Kreative-Drama3

A fork of [PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3) that adds **Q2_K / Q3_K experts** from [UltimateIntent](https://huggingface.co/UltimateIntent/Fish-Audio-Drama-3-Q2-Q3), so Fish Audio Drama 3 runs on a PC with **32 GB of RAM**. Everything else is Pedro's latest pack.

> ⚠️ Tested on one PC: Windows 11, RTX 5060 Ti 16 GB, 32 GB RAM, ComfyUI 0.43 (Easy-Install).

## What this fork adds

1. **Q2_K / Q3_K expert support** (`drama3/k_quants.py` and small hooks in `config.py`, `loader.py`, `model.py`, `offload.py`). The Q2_K file is 16.9 GB and needs about 10.6 GB of RAM for its experts, against 32 GB+ for the original.
2. **A launcher maker**, `tools/make_drama3_launcher.py`, that builds a Drama 3 launcher from your own (see below).
3. **A ready workflow**, `example_workflows/drama3_voice_clone_q2k.json`: Q2_K settings filled in, example text in every text box, and a node titled "Load your audio file here".
4. A `.gitignore` that keeps audio, voices and backups out of the repo.

## Download the Q2_K or Q3_K model

| File | Size | Goes in |
| --- | --- | --- |
| `drama3_q2_k_experts_bf16.safetensors` (smaller) or `drama3_q3_k_experts_bf16.safetensors` (a little better) | 16.9 GB / 20.2 GB | `models/diffusion_models` |
| `drama3_codec.safetensors` | 1.6 GB | `models/vae` |
| `qwen3_asr_0.6b_bf16.safetensors` | 1.6 GB | `models/asr` |

Q2_K / Q3_K: [UltimateIntent/Fish-Audio-Drama-3-Q2-Q3](https://huggingface.co/UltimateIntent/Fish-Audio-Drama-3-Q2-Q3). Codec: [PedroMarinhoDev/Drama-3-ComfyUI](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI). ASR: [PedroMarinhoDev/Qwen3-ASR-ComfyUI](https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI).

Q2_K / Q3_K need **experts: in RAM** and **second_gpu: none** in Load Drama 3.

## You need a separate launcher (important)

Start ComfyUI with **`--disable-pinned-memory`** when you use Drama 3.

**Why:** by default ComfyUI "pins" (locks) a large part of your system RAM, about 13 GB, to move models to the GPU faster. Drama 3 keeps its experts in system RAM too, about 10.6 GB for Q2_K. On a 32 GB PC there isn't room for both, and loading fails with:

```
RuntimeError: hostbuf_file_reader_read failed (cuMemAllocHost ... out of memory)
```

Turning pinning off fixes it. But pinning makes image and video work faster, so don't turn it off for everything. Keep two launchers: this one for Drama 3, your normal one for everything else.

**Make the launcher: the easy way (any PC)**

Run the script that comes with this pack. It finds your ComfyUI, copies *your own* normal launcher and adds the flag, so it matches how your PC is set up (portable, Easy-Install, venv, Mac or Linux). It never changes your normal launcher.

Windows portable / Easy-Install (open PowerShell in the folder that holds `python_embeded`):

```
.\python_embeded\python.exe "ComfyUI\custom_nodes\ComfyUI-Kreative-Drama3\tools\make_drama3_launcher.py"
```

Other installs (use the Python that runs your ComfyUI):

```
python ComfyUI/custom_nodes/ComfyUI-Kreative-Drama3/tools/make_drama3_launcher.py
```

If it can't find ComfyUI, add the path at the end, e.g. `... make_drama3_launcher.py "D:\AI\ComfyUI"`. It makes `Start ComfyUI Drama3.bat` (Windows) or `start_comfyui_drama3.sh` (Mac/Linux) next to your normal launcher.

**Why a script and not just a ready-made file?** Every ComfyUI install is laid out differently (different folders, Python, extra flags). A copied `.bat` only works on a PC set up exactly like the one it came from. The script builds one from yours.

**Or by hand (Windows):**

1. Open the folder with your normal launcher (`.bat`).
2. Copy it and rename the copy **`Start ComfyUI Drama3.bat`**.
3. Right-click it → **Edit**. At the end of the line that runs `main.py`, add a space and **`--disable-pinned-memory`**. Save.

The line should end like this:

```
... main.py --windows-standalone-build --disable-pinned-memory
```

Close ComfyUI fully before switching between launchers.

## Tested settings (16 GB VRAM, 32 GB RAM)

| Node | Setting |
| --- | --- |
| Load Drama 3 | `drama3_q2_k_experts_bf16.safetensors`, experts **in RAM**, expert_cache_gb **4**, second_gpu **none** |
| Load Drama 3 Codec | `drama3_codec.safetensors` |
| Drama 3 Generate | temperature 1.0, top_p 0.9, max_frames 600 |
| Qwen3-ASR Loader | `qwen3_asr_0.6b_bf16.safetensors` |

If a take goes silent before the end, queue again with a new seed.

## Credits

- **[PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3)**: the node pack this fork is built on (GPL-3.0).
- **[UltimateIntent](https://huggingface.co/UltimateIntent/Fish-Audio-Drama-3-Q2-Q3)**: Q2_K / Q3_K quantization and the K-quant code.
- **[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)**: K-quant block layouts (MIT, see `K_QUANT_NOTICE`).
- **[Fish Audio](https://huggingface.co/fishaudio/drama-3)**: Drama 3. Built with Fish Audio.
- Fork, launcher maker and workflow: Lo-res (Kreative2AI).

## Licence

Code: **GPL-3.0** (`LICENSE`), same as the original. Model weights are not in this repo and keep their own licences (Fish Audio Community License for Pedro's files, Fish Audio Research License for UltimateIntent's). Download them from the links above.

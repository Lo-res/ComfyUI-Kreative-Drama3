# ComfyUI-Kreative-Drama3

Fish Audio Drama 3 text to speech in ComfyUI, with **Q2_K / Q3_K experts** so it runs on a PC with **32 GB of RAM**. Voice cloning, dialogue and inline cues like `[laughs]` and `<whisper>`.

A fork of [PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3) with K-quant support from [UltimateIntent](https://huggingface.co/UltimateIntent/Fish-Audio-Drama-3-Q2-Q3), patched for ComfyUI 0.43.

> Tested on one PC: Windows 11, RTX 5060 Ti 16 GB, 32 GB RAM, ComfyUI 0.43 (Easy-Install).

☕ If this saved you time: [buymeacoffee.com/kreative2.ai](https://buymeacoffee.com/kreative2.ai)

## 1. Install

Put this folder in `ComfyUI/custom_nodes/` (git clone or download the zip), then restart ComfyUI.

## 2. Download the models

| File | Size | Goes in |
| --- | --- | --- |
| `drama3_q2_k_experts_bf16.safetensors` (smaller) or `drama3_q3_k_experts_bf16.safetensors` (a little better) | 16.9 GB / 20.2 GB | `models/diffusion_models` |
| `drama3_codec.safetensors` | 1.6 GB | `models/vae` |
| `qwen3_asr_0.6b_bf16.safetensors` | 1.6 GB | `models/asr` |

Q2_K / Q3_K: [UltimateIntent/Fish-Audio-Drama-3-Q2-Q3](https://huggingface.co/UltimateIntent/Fish-Audio-Drama-3-Q2-Q3) · Codec: [PedroMarinhoDev/Drama-3-ComfyUI](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI) · ASR: [PedroMarinhoDev/Qwen3-ASR-ComfyUI](https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI)

## 3. Make a Drama 3 launcher (important)

On a 32 GB PC, ComfyUI must start with **`--disable-pinned-memory`** for Drama 3, or loading fails with `cuMemAllocHost ... out of memory`. Pinning helps image and video work, so keep two launchers: this one for Drama 3, your normal one for everything else.

The included script copies your normal launcher and adds the flag. It never changes your normal launcher.

Windows portable / Easy-Install (PowerShell, in the folder that holds `python_embeded`):

```
.\python_embeded\python.exe "ComfyUI\custom_nodes\ComfyUI-Kreative-Drama3\tools\make_drama3_launcher.py"
```

Other installs (with the Python that runs your ComfyUI):

```
python ComfyUI/custom_nodes/ComfyUI-Kreative-Drama3/tools/make_drama3_launcher.py
```

If it can't find ComfyUI, add the path at the end. It creates `Start ComfyUI Drama3.bat` (Windows) or `start_comfyui_drama3.sh` (Mac/Linux).

**By hand (Windows):** copy your normal launcher `.bat`, rename it `Start ComfyUI Drama3.bat`, right-click → Edit, and add ` --disable-pinned-memory` to the end of the line that runs `main.py`.

Close ComfyUI fully before switching launchers.

## 4. Use it

Load `example_workflows/drama3_voice_clone_q2k.json`.

1. In **Load Audio**, pick a clip of 5–15 s of clean speech. **Qwen3-ASR Transcribe** fills in the transcript.
2. Type your text in **Drama 3 Generate**. Put cues where the sound should land.
3. Queue. The FLAC is saved in `output/drama3/`.
4. To keep the voice: select **Save Drama 3 Voice**, press Ctrl+M to un-mute it and queue once. Next time use **Load Drama 3 Voice** instead of the audio and ASR nodes.

If a take goes silent before the end, queue again with a new seed.

### Settings (16 GB VRAM, 32 GB RAM)

| Node | Setting |
| --- | --- |
| Load Drama 3 | experts **in RAM**, expert_cache_gb **4**, second_gpu **none** |
| Drama 3 Generate | temperature 1.0, top_p 0.9, max_frames 600 |

### Cues

| Type | Example |
| --- | --- |
| Point tag | `Well [sighs] fine, you win. [laughs]` · `[pause]` · `[gasps]` |
| Scoped tag | `Promise me <whisper>you won't tell her</whisper>.` |
| Style | `[with bright eager excitement] We won!` |
| Speakers | `<\|speaker:0\|> Ready?` then `<\|speaker:1\|> Always.` on the next line |

Any wording works. Full syntax: Appendix C of Fish Audio's [Drama 3 technical report](https://github.com/fishaudio/fish-speech/blob/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf).

## Credits and licence

- [PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3): the node pack this is built on.
- [UltimateIntent](https://huggingface.co/UltimateIntent/Fish-Audio-Drama-3-Q2-Q3): Q2_K / Q3_K quants and K-quant code.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp): K-quant block layouts.
- [Fish Audio](https://huggingface.co/fishaudio/drama-3): Drama 3. Built with Fish Audio, not affiliated with it.
- Fork, ComfyUI 0.43 patch, launcher maker and workflow: Lo-res (Kreative2AI).

Code is **GPL-3.0** (`LICENSE`). Parts from Fish Audio, Qwen and llama.cpp keep their own licences, listed in `licenses/`. Model weights are not in this repo and keep their own licences.

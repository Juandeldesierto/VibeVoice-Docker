import argparse
import os
import sys
import time
import copy
import glob
from pathlib import Path

import torch
from transformers.utils import logging

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

logging.set_verbosity_warning()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice preset file paths."""

    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        self.voice_presets = {}
        base_dir = os.path.join(os.path.dirname(__file__), "voices")
        if os.path.exists(base_dir):
            for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.pt"):
                for p in glob.glob(os.path.join(base_dir, "**", ext), recursive=True):
                    name = os.path.splitext(os.path.basename(p))[0].lower()
                    self.voice_presets[name] = os.path.abspath(p)

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path
            for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

    def get_voice_path(self, speaker_name: str) -> str:
        speaker_name = speaker_name.lower().strip()
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Partial matching
        matched_path = None
        for preset_name, path in self.voice_presets.items():
            if preset_name in speaker_name or speaker_name in preset_name:
                if matched_path is not None:
                    raise ValueError(
                        f"Multiple voice presets match '{speaker_name}', please be more specific."
                    )
                matched_path = path

        if matched_path is not None:
            return matched_path

        if not self.voice_presets:
            raise RuntimeError("No voice presets available in voices directory.")

        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', falling back to: {default_voice}")
        return default_voice


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Direct Speech Generator CLI")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text script for VibeVoice to speak out loud.",
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default=None,
        help="Path to a text file containing the script to synthesize.",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="stallone1",
        help="Name of speaker voice preset or audio sample to clone (e.g. stallone1, trump, goku, en-Carter_man).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("MODEL_PATH", "aoi-ot/VibeVoice-Large"),
        help="Path to HuggingFace or local checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        help="Device: cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="CFG scale for diffusion inference (default: 1.5)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/speech.wav",
        help="Destination path for synthesized WAV file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.text is not None:
        script_text = args.text
    elif args.txt_path is not None and os.path.exists(args.txt_path):
        with open(args.txt_path, "r", encoding="utf-8") as f:
            script_text = f.read()
    else:
        script_text = "Hello! Welcome to the presentation."

    script_text = script_text.replace("’", "'").replace('“', '"').replace('”', '"')

    device = args.device.lower()
    if device == "mpx":
        device = "mps"
    if device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS unavailable, falling back to CPU.")
        device = "cpu"

    is_streaming = (
        "realtime" in args.model_path.lower()
        or "streaming" in args.model_path.lower()
        or "0.5b" in args.model_path.lower()
    )

    print(f"[VibeVoice CLI] Target device: {device}")
    print(f"[VibeVoice CLI] Loading model: {args.model_path} (streaming={is_streaming})")

    if is_streaming:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference as ModelClass,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor as ProcessorClass,
        )
    else:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference as ModelClass,
        )
        from vibevoice.processor.vibevoice_processor import (
            VibeVoiceProcessor as ProcessorClass,
        )

    processor = ProcessorClass.from_pretrained(args.model_path)

    if device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    elif device == "mps":
        load_dtype = torch.float32
        attn_impl = "sdpa"
    else:
        load_dtype = torch.float32
        attn_impl = "sdpa"

    try:
        model = ModelClass.from_pretrained(
            args.model_path,
            torch_dtype=load_dtype,
            device_map=("cuda" if device == "cuda" else ("cpu" if device == "cpu" else None)),
            attn_implementation=attn_impl,
        )
        if device == "mps":
            model.to("mps")
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print("[VibeVoice CLI] flash_attention_2 not available, falling back to SDPA...")
            model = ModelClass.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map=("cuda" if device == "cuda" else ("cpu" if device == "cpu" else None)),
                attn_implementation="sdpa",
            )
        else:
            raise e

    model.eval()
    if is_streaming:
        model.set_ddpm_inference_steps(num_steps=5)

    voice_mapper = VoiceMapper()
    voice_sample = voice_mapper.get_voice_path(args.speaker_name)
    print(f"[VibeVoice CLI] Selected voice: {os.path.basename(voice_sample)}")

    target_device = device if device != "cpu" else "cpu"

    if is_streaming:
        all_prefilled_outputs = torch.load(voice_sample, map_location=target_device, weights_only=False)
        inputs = processor.process_input_with_cached_prompt(
            text=script_text,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    else:
        clean_text = script_text.strip()
        if not clean_text.startswith("Speaker"):
            clean_text = f"Speaker 0: {clean_text}\n"
        inputs = processor(
            text=[clean_text],
            voice_samples=[[voice_sample]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"[VibeVoice CLI] Generating speech for: \"{script_text[:60]}{'...' if len(script_text) > 60 else ''}\"")
    start_time = time.time()
    gen_kwargs = {
        **inputs,
        "max_new_tokens": None,
        "cfg_scale": args.cfg_scale,
        "tokenizer": processor.tokenizer,
        "generation_config": {"do_sample": False},
        "verbose": False,
    }
    if is_streaming:
        gen_kwargs["all_prefilled_outputs"] = copy.deepcopy(all_prefilled_outputs)

    outputs = model.generate(**gen_kwargs)
    elapsed = time.time() - start_time

    output_file = Path(args.output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processor.save_audio(
        outputs.speech_outputs[0],
        output_path=str(output_file),
    )
    print(f"[VibeVoice CLI] Audio successfully saved to: {output_file} ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()

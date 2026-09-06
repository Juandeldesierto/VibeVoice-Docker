import datetime
import builtins
import asyncio
import json
import os
import re
import threading
import time
import uuid
import shutil
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import (
    VibeVoiceProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

import copy

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp

class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.is_streaming = (
            "realtime" in self.model_path.lower()
            or "streaming" in self.model_path.lower()
            or "0.5b" in self.model_path.lower()
        )

        self.processor = None
        self.model = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor & model from {self.model_path} (is_streaming={self.is_streaming})")
        if self.is_streaming:
            self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
            ModelClass = VibeVoiceStreamingForConditionalGenerationInference
        else:
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            ModelClass = VibeVoiceForConditionalGenerationInference
        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        
        # Load model
        try:
            self.model = ModelClass.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print(f"Notice: Loading with flash_attention_2 failed ({e}). Falling back to SDPA.")
                self.model = ModelClass.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Loaded model with SDPA successfully")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        if self.is_streaming:
            self._ensure_voice_cached(self.default_voice_key)

    def _load_voice_presets(self) -> Dict[str, Path]:
        presets: Dict[str, Path] = {}
        if self.is_streaming:
            voices_dir = BASE.parent / "voices" / "streaming_model"
            if voices_dir.exists():
                for pt_path in voices_dir.rglob("*.pt"):
                    presets[pt_path.stem] = pt_path
        else:
            voices_dir = BASE.parent / "voices"
            if voices_dir.exists():
                audio_exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
                for ext in audio_exts:
                    for audio_path in voices_dir.glob(ext):
                        if audio_path.is_file():
                            presets[audio_path.stem] = audio_path
                custom_dir = voices_dir / "custom"
                if custom_dir.exists():
                    for ext in audio_exts:
                        for audio_path in custom_dir.glob(ext):
                            if audio_path.is_file():
                                presets[audio_path.stem] = audio_path

        # Fallback if empty: scan entire voices dir for any audio or .pt files
        if not presets:
            voices_dir = BASE.parent / "voices"
            for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.pt"):
                for p in voices_dir.rglob(ext):
                    presets[p.stem] = p

        print(f"[startup] Found {len(presets)} voice presets (streaming={self.is_streaming}): {list(presets.keys())[:10]}")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        for preferred in ("stallone1", "trump", "goku", "picard1", "en-Carter_man"):
            if preferred in self.voice_presets:
                return preferred

        if self.voice_presets:
            first_key = next(iter(self.voice_presets))
            print(f"[startup] Using fallback voice preset: {first_key}")
            return first_key
        return "default"

    def _ensure_voice_cached(self, key: str) -> Optional[object]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if not self.is_streaming:
            return None

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, Optional[object]]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None and self.voice_presets:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key) if key and self.is_streaming else None
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, voice_key: str, prefilled_outputs: Optional[object] = None):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        if self.is_streaming:
            processor_kwargs = {
                "text": text.strip(),
                "cached_prompt": prefilled_outputs,
                "padding": True,
                "return_tensors": "pt",
                "return_attention_mask": True,
            }
            processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)
        else:
            voice_path = str(self.voice_presets[voice_key])
            clean_text = text.strip()
            if not clean_text.startswith("Speaker"):
                clean_text = f"Speaker 0: {clean_text}\n"
            else:
                clean_text = f"{clean_text}\n"
            processed = self.processor(
                text=[clean_text],
                voice_samples=[[voice_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

        prepared = {
            k: v.to(self._torch_device) if hasattr(v, "to") else v
            for k, v in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            gen_kwargs = {
                **inputs,
                "max_new_tokens": None,
                "cfg_scale": cfg_scale,
                "tokenizer": self.processor.tokenizer,
                "generation_config": {
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                "audio_streamer": audio_streamer,
                "stop_check_fn": stop_event.is_set,
                "verbose": False,
            }
            if self.is_streaming:
                gen_kwargs["refresh_negative"] = refresh_negative
                gen_kwargs["all_prefilled_outputs"] = copy.deepcopy(prefilled_outputs)

            self.model.generate(**gen_kwargs)
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, selected_voice, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()

    def generate_wav(
        self,
        text: str,
        output_path: str,
        voice_key: Optional[str] = None,
        cfg_scale: float = 1.5,
    ) -> float:
        import scipy.io.wavfile as wavfile
        chunks = []
        for chunk in self.stream(text=text, voice_key=voice_key, cfg_scale=cfg_scale):
            chunks.append(chunk)
        if not chunks:
            raise ValueError("No audio chunks generated")
        full_audio = np.concatenate(chunks, axis=-1)
        full_audio = np.clip(full_audio, -1.0, 1.0)
        pcm16 = (full_audio * 32767.0).astype(np.int16)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(out_p), self.sample_rate, pcm16)
        return len(pcm16) / self.sample_rate


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")
    
    service = StreamingTTSService(
        model_path=model_path,
        device=device
    )
    service.load()

    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.websocket_lock = asyncio.Lock()
    print("[startup] Model ready.")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)

@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    print(f"Client connected, text={text!r}")
    cfg_param = ws.query_params.get("cfg")
    steps_param = ws.query_params.get("steps")
    voice_param = ws.query_params.get("voice")

    try:
        cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
    except ValueError:
        cfg_scale = 1.5
    if cfg_scale <= 0:
        cfg_scale = 1.5
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except ValueError:
        inference_steps = None

    service: StreamingTTSService = app.state.tts_service
    lock: asyncio.Lock = app.state.websocket_lock

    if lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Please wait for the other requests to complete."},
            "timestamp": get_timestamp(),
        }
        print("Please wait for the other requests to complete.")
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        enqueue_log(
            "backend_request_received",
            text_length=len(text or ""),
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
        )

        stop_signal = threading.Event()

        iterator = streaming_tts(
            text,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice_key=voice_param,
            log_callback=enqueue_log,
            stop_event=stop_signal,
        )
        sentinel = object()
        first_ws_send_logged = False

        await flush_logs()

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                await flush_logs()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                chunk = cast(np.ndarray, chunk)
                payload = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(payload)
                if not first_ws_send_logged:
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")
                await flush_logs()
        except WebSocketDisconnect:
            print("Client disconnected (WebSocketDisconnect)")
            enqueue_log("client_disconnected")
            stop_signal.set()
        except Exception as e:
            print(f"Error in websocket stream: {e}")
            traceback.print_exc()
            enqueue_log("backend_error", message=str(e))
            stop_signal.set()
        finally:
            stop_signal.set()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()
            except Exception:
                pass
            # clear the log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close()
            except Exception as e:
                print(f"Error closing websocket: {e}")
            print("WS handler exit")
    finally:
        if acquired:
            lock.release()


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
        "is_streaming": service.is_streaming,
        "model_path": service.model_path,
    }


@app.post("/api/voices/upload")
async def upload_custom_voice(voice_file: UploadFile = File(...)):
    if not voice_file.filename:
        raise HTTPException(status_code=400, detail="No voice file provided.")

    ext = Path(voice_file.filename).suffix.lower()
    valid_exts = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".pt")
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported voice file format '{ext}'. Supported: {', '.join(valid_exts)}"
        )

    service: StreamingTTSService = app.state.tts_service
    if ext == ".pt":
        voices_dir = BASE.parent / "voices" / "streaming_model" / "custom"
    else:
        voices_dir = BASE.parent / "voices" / "custom"
    voices_dir.mkdir(parents=True, exist_ok=True)

    clean_name = Path(voice_file.filename).stem
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', clean_name)
    target_path = voices_dir / f"{clean_name}{ext}"

    with open(target_path, "wb") as f_out:
        shutil.copyfileobj(voice_file.file, f_out)

    # Reload voice presets in service
    service.voice_presets = service._load_voice_presets()
    new_voices = sorted(service.voice_presets.keys())

    return {
        "success": True,
        "voice_name": clean_name,
        "voices": new_voices,
        "message": f"Successfully loaded custom voice '{clean_name}' ({ext})",
    }


# Directories for Avatar Pipeline
ROOT_DIR = BASE.parent.parent
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TASKS_DIR = OUTPUTS_DIR / "tasks"

INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/inputs", StaticFiles(directory=str(INPUTS_DIR)), name="inputs")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/api/media", StaticFiles(directory=str(OUTPUTS_DIR)), name="media")


@app.get("/api/avatar/presets")
def get_avatar_presets():
    presets_dir = INPUTS_DIR / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for f in presets_dir.glob(ext):
            items.append({
                "name": f.name,
                "label": f.stem.replace("_", " ").title(),
                "url": f"/inputs/presets/{f.name}",
            })
    return {"presets": sorted(items, key=lambda x: x["label"])}


@app.post("/api/avatar/generate")
async def generate_avatar(
    text: Optional[str] = Form(None),
    speaker: str = Form("en-Carter_man"),
    audio_file: Optional[UploadFile] = File(None),
    portrait_file: Optional[UploadFile] = File(None),
    preset_name: Optional[str] = Form(None),
    enhancer: str = Form("gfpgan"),
    still: bool = Form(False),
    expression_scale: float = Form(1.0),
):
    task_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # 1. Determine portrait image path
    if portrait_file and portrait_file.filename:
        file_ext = Path(portrait_file.filename).suffix or ".png"
        img_filename = f"portrait_{task_id}{file_ext}"
        target_img = INPUTS_DIR / img_filename
        with open(target_img, "wb") as f_out:
            shutil.copyfileobj(portrait_file.file, f_out)
        img_rel_path = f"inputs/{img_filename}"
        img_url = f"/inputs/{img_filename}"
    elif preset_name and (INPUTS_DIR / "presets" / preset_name).exists():
        img_rel_path = f"inputs/presets/{preset_name}"
        img_url = f"/inputs/presets/{preset_name}"
    elif (INPUTS_DIR / "portrait.png").exists():
        img_rel_path = "inputs/portrait.png"
        img_url = "/inputs/portrait.png"
    else:
        presets = list((INPUTS_DIR / "presets").glob("*.*"))
        if presets:
            img_rel_path = f"inputs/presets/{presets[0].name}"
            img_url = f"/inputs/presets/{presets[0].name}"
        else:
            raise HTTPException(status_code=400, detail="No portrait image found or provided.")

    # 2. Determine audio source (custom uploaded voice audio OR VibeVoice TTS)
    audio_filename = f"speech_{task_id}.wav"
    audio_path = OUTPUTS_DIR / audio_filename
    audio_rel_path = f"outputs/{audio_filename}"
    audio_url = f"/outputs/{audio_filename}"

    if audio_file and audio_file.filename:
        # User provided their own voice audio file!
        raw_ext = Path(audio_file.filename).suffix or ".wav"
        raw_audio_path = OUTPUTS_DIR / f"raw_audio_{task_id}{raw_ext}"
        with open(raw_audio_path, "wb") as f_out:
            shutil.copyfileobj(audio_file.file, f_out)

        # Convert to WAV with ffmpeg if available, otherwise copy
        if raw_ext.lower() == ".wav":
            shutil.copy2(raw_audio_path, audio_path)
        else:
            try:
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(raw_audio_path), "-ar", "24000", "-ac", "1", str(audio_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as conv_err:
                print(f"Warning: ffmpeg conversion failed, using raw audio: {conv_err}")
                shutil.copy2(raw_audio_path, audio_path)

        stage_msg = "Custom voice file loaded. Queued for SadTalker animation..."
    else:
        # Generate with VibeVoice TTS from text
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Please enter text or upload a custom voice audio file.")

        service: StreamingTTSService = app.state.tts_service
        try:
            service.generate_wav(
                text=text.strip(),
                output_path=str(audio_path),
                voice_key=speaker,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Speech synthesis error: {str(e)}")

        stage_msg = "Voice synthesized. Queued for avatar animation."

    # 3. Create task file for SadTalker worker
    task_data = {
        "id": task_id,
        "status": "pending",
        "stage": stage_msg,
        "progress": 25,
        "text": text.strip() if text else "",
        "speaker": speaker,
        "audio_path": audio_rel_path,
        "audio_url": audio_url,
        "image_path": img_rel_path,
        "image_url": img_url,
        "enhancer": enhancer,
        "still": still,
        "expression_scale": expression_scale,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    task_file = TASKS_DIR / f"task_{task_id}.json"
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2)

    return {
        "task_id": task_id,
        "audio_url": audio_url,
        "status": "pending",
        "stage": stage_msg,
    }


@app.get("/api/avatar/status/{task_id}")
def get_avatar_status(task_id: str):
    task_file = TASKS_DIR / f"task_{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)
    except Exception:
        return {"status": "pending", "stage": "Processing...", "progress": 50}

    # Check if target video exists
    expected_video = OUTPUTS_DIR / f"avatar_{task_id}.mp4"
    if expected_video.exists():
        task_data["video_url"] = f"/outputs/avatar_{task_id}.mp4"
        if task_data.get("status") != "completed":
            task_data["status"] = "completed"
            task_data["progress"] = 100
            task_data["stage"] = "Done"

    return task_data


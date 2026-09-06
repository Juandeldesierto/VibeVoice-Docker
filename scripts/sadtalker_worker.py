import os
import sys
import time
import json
import glob
import shutil
import subprocess
from pathlib import Path

# Paths inside container vs host
if os.path.exists("/host_dir"):
    BASE_DIR = Path("/host_dir")
else:
    BASE_DIR = Path(os.getcwd())

TASKS_DIR = BASE_DIR / "outputs" / "tasks"
OUTPUTS_DIR = BASE_DIR / "outputs"

TASKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(" SadTalker Background Queue Worker Started")
print(f" Watching for tasks in: {TASKS_DIR}")
print(f" Output directory:     {OUTPUTS_DIR}")
print("=" * 60)
sys.stdout.flush()


def find_latest_mp4(directory: Path, since_time: float) -> str:
    """Find the newest generated mp4 file in result_dir."""
    mp4_files = list(directory.rglob("*.mp4"))
    if not mp4_files:
        return ""
    # Prefer enhanced video if generated
    enhanced_files = [f for f in mp4_files if "enhanced" in f.name.lower() and f.stat().st_mtime >= since_time - 10]
    if enhanced_files:
        enhanced_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(enhanced_files[0])

    # Sort by modification time
    mp4_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for f in mp4_files:
        if f.stat().st_mtime >= since_time - 10:
            return str(f)
    return str(mp4_files[0])


def process_task(task_path: Path):
    try:
        with open(task_path, "r", encoding="utf-8") as f:
            task = json.load(f)
    except Exception as e:
        print(f"Error reading {task_path}: {e}")
        return

    if task.get("status") != "pending":
        return

    task_id = task.get("id", task_path.stem)
    print(f"\n[Worker] Picked up task {task_id}")
    task["status"] = "processing"
    task["stage"] = "SadTalker: Generating animated facial video..."
    task["progress"] = 30
    task["updated_at"] = time.time()

    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2)

    # Resolve paths (handling container vs host paths)
    audio_path = task["audio_path"]
    image_path = task["image_path"]

    if os.path.exists("/host_dir"):
        # Convert relative to container path
        if not audio_path.startswith("/host_dir"):
            audio_path = str(BASE_DIR / audio_path.replace("\\", "/").lstrip("/"))
        if not image_path.startswith("/host_dir"):
            image_path = str(BASE_DIR / image_path.replace("\\", "/").lstrip("/"))
        result_dir = str(OUTPUTS_DIR)
    else:
        result_dir = str(OUTPUTS_DIR)

    print(f"[Worker] Audio: {audio_path}")
    print(f"[Worker] Image: {image_path}")

    # Build SadTalker command
    cmd = [
        sys.executable,
        "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", result_dir,
    ]

    enhancer = task.get("enhancer", "gfpgan")
    if enhancer and enhancer.lower() not in ("none", "null", "false", ""):
        cmd.extend(["--enhancer", enhancer.lower()])

    # Avoid --still shape mismatch unless mapping checkpoint is explicitly provided
    if task.get("still", False):
        chk_mapping = BASE_DIR / "checkpoints" / "mapping_00109-model.pth.tar"
        if chk_mapping.is_file():
            cmd.append("--still")
        else:
            print("[Worker] Note: Omitting --still to prevent tensor dimension mismatch.")

    expr_scale = task.get("expression_scale", 1.0)
    cmd.extend(["--expression_scale", str(expr_scale)])

    # Only override --checkpoint_dir if landmark model actually exists in the mounted directory
    chk_dir = BASE_DIR / "checkpoints"
    if (chk_dir / "shape_predictor_68_face_landmarks.dat").is_file():
        cmd.extend(["--checkpoint_dir", str(chk_dir)])
    else:
        print("[Worker] Using SadTalker's internal pre-baked checkpoints.")

    print(f"[Worker] Running: {' '.join(cmd)}")
    sys.stdout.flush()

    start_t = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        print("[Worker] SadTalker completed successfully.")
        
        # Locate the output mp4
        target_video = OUTPUTS_DIR / f"avatar_{task_id}.mp4"
        latest_mp4 = find_latest_mp4(OUTPUTS_DIR, start_t)
        
        if latest_mp4 and os.path.exists(latest_mp4):
            if Path(latest_mp4).resolve() != target_video.resolve():
                shutil.copy2(latest_mp4, target_video)
            video_rel_path = f"outputs/avatar_{task_id}.mp4"
        else:
            video_rel_path = latest_mp4

        task["status"] = "completed"
        task["stage"] = "Done"
        task["progress"] = 100
        task["video_path"] = video_rel_path
        task["video_url"] = f"/outputs/avatar_{task_id}.mp4"
        task["completed_at"] = time.time()


    except subprocess.CalledProcessError as err:
        print(f"[Worker ERROR] SadTalker failed: {err}")
        print(err.stdout)
        task["status"] = "failed"
        task["stage"] = "Failed"
        task["error"] = f"SadTalker execution error: {err.stdout[-500:] if err.stdout else str(err)}"
    except Exception as e:
        print(f"[Worker ERROR] Unexpected error: {e}")
        task["status"] = "failed"
        task["stage"] = "Failed"
        task["error"] = str(e)

    task["updated_at"] = time.time()
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2)
    print(f"[Worker] Task {task_id} finished with status: {task['status']}")
    sys.stdout.flush()


def main():
    while True:
        task_files = list(TASKS_DIR.glob("task_*.json"))
        # Sort oldest first
        task_files.sort(key=lambda p: p.stat().st_mtime)
        for tf in task_files:
            process_task(tf)
        time.sleep(1)



if __name__ == "__main__":
    main()

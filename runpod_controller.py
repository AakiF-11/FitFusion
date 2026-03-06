"""
FitFusion — RunPod Cloud GPU Controller
========================================
Control RunPod pods from your local VS Code terminal.
Manage GPU provisioning, code sync, training, and monitoring.

Usage:
    python runpod_controller.py setup          # First-time: save API key + SSH key
    python runpod_controller.py gpus           # List available GPUs + pricing
    python runpod_controller.py create         # Create a training pod
    python runpod_controller.py status         # Show pod status + SSH info
    python runpod_controller.py ssh            # Print SSH command to connect
    python runpod_controller.py sync           # Upload code + data to pod
    python runpod_controller.py install        # Install dependencies on pod
    python runpod_controller.py train          # Start training on pod
    python runpod_controller.py logs           # Stream training logs
    python runpod_controller.py download       # Download checkpoints from pod
    python runpod_controller.py stop           # Stop pod (keeps data, stops billing)
    python runpod_controller.py resume         # Resume a stopped pod
    python runpod_controller.py terminate      # DELETE pod permanently
    python runpod_controller.py cost           # Show current session cost
"""

import os
import sys
import json
import time
import argparse
import subprocess
import getpass
from pathlib import Path
from datetime import datetime, timedelta

try:
    import runpod
except ImportError:
    print("ERROR: runpod not installed. Run: pip install runpod paramiko scp")
    sys.exit(1)

try:
    import paramiko
    from scp import SCPClient
except ImportError:
    print("ERROR: paramiko/scp not installed. Run: pip install paramiko scp")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(ROOT, ".runpod_config.json")

# Files/folders to sync to the pod
SYNC_INCLUDES = [
    "IDM-VTON/",                    # Core model code (includes train_xl_qlora.py, measurement_encoder.py, prepare_vitonhd_dataset.py)
    "src/",                         # Pipeline wrapper modules (pose, masking, size_physics, etc.)
    "fitfusion/",                   # Compositor, confidence scorer, background utils
    "data/",                        # Test images, brand catalog, VITON-HD test set
    "run_pipeline.py",              # Main inference entry point
    "requirements.txt",             # Python dependencies
    "training_data_extraction/",    # Training data
    "extract_snag_tights.py",
    "extract_universal_standard.py",
    "MODEL_SELECTION_AND_TRAINING_PLAN.md",
]

# Files to NEVER sync
SYNC_EXCLUDES = [
    "venv_ootd/",
    ".git/",
    "__pycache__/",
    "*.pyc",
    ".runpod_config.json",
]

# Default pod configuration
DEFAULT_POD_CONFIG = {
    "name": "fitfusion-training",
    "image_name": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "gpu_type_id": "NVIDIA A40",          # 48GB VRAM, ~$0.40/hr
    "cloud_type": "COMMUNITY",             # cheapest
    "gpu_count": 1,
    "volume_in_gb": 50,                    # persistent storage
    "container_disk_in_gb": 20,
    "volume_mount_path": "/workspace",
    "start_ssh": True,
    "support_public_ip": True,
    "ports": "8888/http,22/tcp",
}

# GPU preference order (tried in sequence if first choice unavailable)
GPU_PREFERENCE = [
    "NVIDIA A40",               # 48 GB, ~$0.40/hr — best value
    "NVIDIA RTX A6000",         # 48 GB, ~$0.49/hr
    "NVIDIA L40",               # 48 GB, ~$0.50/hr
    "NVIDIA GeForce RTX 4090",  # 24 GB, ~$0.59/hr (tight but works with QLoRA)
]

REMOTE_PROJECT_DIR = "/workspace/FitFusion"


# ═══════════════════════════════════════════════════════════════════════
# CONFIG MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════
def load_config():
    """Load saved configuration."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save configuration to disk."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {CONFIG_FILE}")


def ensure_api_key():
    """Ensure RunPod API key is configured."""
    config = load_config()
    api_key = config.get("api_key")
    if not api_key:
        print("ERROR: No RunPod API key configured.")
        print("Run: python runpod_controller.py setup")
        sys.exit(1)
    runpod.api_key = api_key
    return config


# ═══════════════════════════════════════════════════════════════════════
# SSH CONNECTION
# ═══════════════════════════════════════════════════════════════════════
def get_ssh_details(pod_info):
    """Extract SSH connection details from pod info."""
    runtime = pod_info.get("runtime")
    if not runtime:
        return None, None

    ports = runtime.get("ports", [])
    for port_info in ports:
        if port_info.get("privatePort") == 22:
            return port_info.get("ip"), port_info.get("publicPort")
    return None, None


def create_ssh_client(config):
    """Create an SSH client connected to the pod."""
    pod_id = config.get("pod_id")
    if not pod_id:
        print("ERROR: No pod created yet. Run: python runpod_controller.py create")
        sys.exit(1)

    ensure_api_key()
    pod_info = runpod.get_pod(pod_id)

    if not pod_info:
        print(f"ERROR: Pod {pod_id} not found.")
        sys.exit(1)

    status = pod_info.get("desiredStatus", "UNKNOWN")
    if status != "RUNNING":
        print(f"ERROR: Pod is {status}, not RUNNING.")
        print("  Run: python runpod_controller.py resume")
        sys.exit(1)

    ssh_ip, ssh_port = get_ssh_details(pod_info)
    if not ssh_ip:
        print("ERROR: Pod is running but SSH not ready yet. Wait a moment.")
        sys.exit(1)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try SSH key first, then password
    ssh_key_path = config.get("ssh_key_path")
    try:
        if ssh_key_path and os.path.exists(ssh_key_path):
            ssh.connect(ssh_ip, port=ssh_port, username="root",
                        key_filename=ssh_key_path, timeout=30)
        else:
            # RunPod pods may allow root without password if public key was set
            ssh.connect(ssh_ip, port=ssh_port, username="root", timeout=30,
                        look_for_keys=True)
    except Exception as e:
        print(f"ERROR: SSH connection failed: {e}")
        print(f"  Try manually: ssh root@{ssh_ip} -p {ssh_port}")
        sys.exit(1)

    return ssh


def ssh_exec(ssh, cmd, stream=False, timeout=600):
    """Execute command on pod via SSH."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)

    if stream:
        # Stream output line by line
        for line in iter(stdout.readline, ""):
            print(line, end="")
        err = stderr.read().decode()
        if err:
            print(err, end="")
        return stdout.channel.recv_exit_status()
    else:
        out = stdout.read().decode()
        err = stderr.read().decode()
        exit_code = stdout.channel.recv_exit_status()
        return out, err, exit_code


# ═══════════════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════════════

def cmd_setup(args):
    """First-time setup: save API key and SSH key path."""
    print("=" * 60)
    print("FitFusion RunPod Setup")
    print("=" * 60)
    print()
    print("Get your API key from: https://www.runpod.io/console/user/settings")
    print()

    config = load_config()

    api_key = input("RunPod API key: ").strip()
    if not api_key:
        print("ERROR: API key cannot be empty.")
        return

    config["api_key"] = api_key

    # Validate key
    runpod.api_key = api_key
    try:
        gpus = runpod.get_gpus()
        print(f"  API key valid! Found {len(gpus)} GPU types available.")
    except Exception as e:
        print(f"  WARNING: Could not validate API key: {e}")
        print("  Saving anyway — you can re-run setup if it's wrong.")

    # SSH key
    default_key = os.path.expanduser("~/.ssh/id_rsa")
    ed25519_key = os.path.expanduser("~/.ssh/id_ed25519")
    if os.path.exists(ed25519_key):
        default_key = ed25519_key

    print()
    print(f"SSH private key path (default: {default_key}):")
    ssh_key = input("  > ").strip() or default_key

    if os.path.exists(ssh_key):
        config["ssh_key_path"] = ssh_key
        print(f"  SSH key found: {ssh_key}")

        # Read public key for RunPod
        pub_key_path = ssh_key + ".pub"
        if os.path.exists(pub_key_path):
            with open(pub_key_path, "r") as f:
                pub_key = f.read().strip()
            config["ssh_public_key"] = pub_key
            print(f"  Public key loaded from {pub_key_path}")

            # Upload to RunPod
            try:
                runpod.update_user_settings(pubkey=pub_key)
                print("  Public key uploaded to RunPod account!")
            except Exception as e:
                print(f"  WARNING: Could not upload public key: {e}")
                print("  You may need to add it manually at runpod.io")
    else:
        print(f"  WARNING: SSH key not found at {ssh_key}")
        print("  Generate one with: ssh-keygen -t ed25519")
        print("  Then re-run setup.")

    save_config(config)
    print()
    print("Setup complete! Next: python runpod_controller.py create")


def cmd_gpus(args):
    """List available GPUs with pricing."""
    config = ensure_api_key()

    print("=" * 70)
    print("Available RunPod GPUs")
    print("=" * 70)
    print(f"{'GPU':<35} {'VRAM':>6} {'Community':>11} {'Secure':>9}")
    print("-" * 70)

    gpus = runpod.get_gpus()
    gpu_details = []

    for gpu in gpus:
        gpu_id = gpu.get("id", "")
        mem = gpu.get("memoryInGb", 0)
        if mem < 16:
            continue  # skip small GPUs

        try:
            detail = runpod.get_gpu(gpu_id)
            community_price = detail.get("communityPrice") or detail.get("lowestPrice", {}).get("uninterruptablePrice") or "-"
            secure_price = detail.get("securePrice") or "-"

            if isinstance(community_price, (int, float)):
                community_str = f"${community_price:.2f}/hr"
            else:
                community_str = str(community_price)

            if isinstance(secure_price, (int, float)):
                secure_str = f"${secure_price:.2f}/hr"
            else:
                secure_str = str(secure_price)

            recommended = " <-- RECOMMENDED" if gpu_id in GPU_PREFERENCE[:2] else ""
            print(f"{gpu_id:<35} {mem:>4} GB {community_str:>11} {secure_str:>9}{recommended}")
            gpu_details.append(detail)
        except Exception:
            print(f"{gpu_id:<35} {mem:>4} GB {'(error)':>11}")

    print()
    print("Recommended for FitFusion training:")
    print("  1. NVIDIA A40 (48 GB, ~$0.40/hr) — best balance")
    print("  2. NVIDIA RTX A6000 (48 GB, ~$0.49/hr) — premium option")
    print("  3. NVIDIA RTX 4090 (24 GB, ~$0.59/hr) — tight but works with QLoRA")


def cmd_create(args):
    """Create a new training pod."""
    config = ensure_api_key()

    if config.get("pod_id"):
        print(f"WARNING: Pod {config['pod_id']} already exists.")
        print("  Run 'terminate' first, or 'resume' to restart it.")
        choice = input("  Create a NEW pod anyway? (y/N): ").strip().lower()
        if choice != "y":
            return

    print("=" * 60)
    print("Creating FitFusion Training Pod")
    print("=" * 60)

    # Select GPU
    gpu_id = args.gpu if hasattr(args, 'gpu') and args.gpu else None

    if not gpu_id:
        print("\nTrying GPUs in preference order...")
        for gpu_pref in GPU_PREFERENCE:
            try:
                detail = runpod.get_gpu(gpu_pref)
                avail = detail.get("communityCloud") or detail.get("secureCloud")
                if avail:
                    gpu_id = gpu_pref
                    mem = detail.get("memoryInGb", "?")
                    price = detail.get("communityPrice") or detail.get("lowestPrice", {}).get("uninterruptablePrice", "?")
                    print(f"  Found: {gpu_id} ({mem}GB) @ ${price}/hr")
                    break
                else:
                    print(f"  {gpu_pref}: not available right now")
            except Exception as e:
                print(f"  {gpu_pref}: error checking — {e}")

    if not gpu_id:
        print("ERROR: No preferred GPU available. Check: python runpod_controller.py gpus")
        return

    pod_config = DEFAULT_POD_CONFIG.copy()
    pod_config["gpu_type_id"] = gpu_id

    # Add SSH public key as env var
    pub_key = config.get("ssh_public_key", "")
    if pub_key:
        pod_config["env"] = {"PUBLIC_KEY": pub_key}

    print(f"\n  GPU: {gpu_id}")
    print(f"  Image: {pod_config['image_name']}")
    print(f"  Volume: {pod_config['volume_in_gb']} GB")
    print(f"  Container disk: {pod_config['container_disk_in_gb']} GB")
    print()

    try:
        pod = runpod.create_pod(**pod_config)
        pod_id = pod.get("id")
        print(f"  Pod created! ID: {pod_id}")
        config["pod_id"] = pod_id
        config["created_at"] = datetime.now().isoformat()
        config["gpu_type"] = gpu_id
        save_config(config)

        # Wait for pod to be ready
        print("\n  Waiting for pod to start (this takes 1-3 minutes)...")
        for i in range(60):
            time.sleep(5)
            try:
                pod_info = runpod.get_pod(pod_id)
                status = pod_info.get("desiredStatus", "UNKNOWN")
                runtime = pod_info.get("runtime")

                if runtime and runtime.get("ports"):
                    ssh_ip, ssh_port = get_ssh_details(pod_info)
                    if ssh_ip:
                        print(f"\n  Pod READY!")
                        print(f"  SSH: ssh root@{ssh_ip} -p {ssh_port}")
                        config["ssh_ip"] = ssh_ip
                        config["ssh_port"] = ssh_port
                        save_config(config)
                        print(f"\n  Next: python runpod_controller.py sync")
                        return
                print(f"    [{i*5}s] Status: {status}...", end="\r")
            except Exception:
                pass

        print("\n  Pod created but SSH not ready yet.")
        print("  Run 'python runpod_controller.py status' to check.")

    except Exception as e:
        print(f"ERROR creating pod: {e}")


def cmd_status(args):
    """Show current pod status."""
    config = ensure_api_key()
    pod_id = config.get("pod_id")

    if not pod_id:
        print("No pod configured. Run: python runpod_controller.py create")
        return

    try:
        pod_info = runpod.get_pod(pod_id)
    except Exception as e:
        print(f"ERROR getting pod status: {e}")
        return

    if not pod_info:
        print(f"Pod {pod_id} not found (may have been terminated).")
        return

    status = pod_info.get("desiredStatus", "UNKNOWN")
    gpu_name = ""
    machine = pod_info.get("machine")
    if machine:
        gpu_name = machine.get("gpuDisplayName", "")

    cost_per_hr = pod_info.get("costPerHr", 0)
    uptime = pod_info.get("uptimeSeconds", 0)

    print("=" * 60)
    print("FitFusion RunPod Pod Status")
    print("=" * 60)
    print(f"  Pod ID:      {pod_id}")
    print(f"  Status:      {status}")
    print(f"  GPU:         {gpu_name or config.get('gpu_type', 'Unknown')}")
    print(f"  Cost:        ${cost_per_hr}/hr")
    print(f"  Uptime:      {timedelta(seconds=uptime)}")
    print(f"  Session cost: ${cost_per_hr * uptime / 3600:.2f}")
    print(f"  Image:       {pod_info.get('imageName', 'Unknown')}")
    print(f"  Volume:      {pod_info.get('volumeInGb', '?')} GB")

    ssh_ip, ssh_port = get_ssh_details(pod_info)
    if ssh_ip:
        print(f"\n  SSH command:")
        print(f"    ssh root@{ssh_ip} -p {ssh_port}")

        # Update cached SSH details
        config["ssh_ip"] = ssh_ip
        config["ssh_port"] = ssh_port
        save_config(config)
    elif status == "RUNNING":
        print("\n  SSH not ready yet, pod still booting...")


def cmd_ssh(args):
    """Print SSH command to connect."""
    config = ensure_api_key()
    pod_id = config.get("pod_id")
    if not pod_id:
        print("No pod. Run: python runpod_controller.py create")
        return

    pod_info = runpod.get_pod(pod_id)
    ssh_ip, ssh_port = get_ssh_details(pod_info)

    if ssh_ip:
        ssh_key = config.get("ssh_key_path", "")
        key_flag = f" -i {ssh_key}" if ssh_key else ""
        cmd = f"ssh root@{ssh_ip} -p {ssh_port}{key_flag}"
        print(f"\n  {cmd}\n")
        print("  Copy and paste into a terminal to connect.")
    else:
        print("Pod is not running or SSH not ready.")


def cmd_sync(args):
    """Upload project code and data to the pod."""
    config = ensure_api_key()

    print("=" * 60)
    print("Syncing FitFusion to RunPod")
    print("=" * 60)

    ssh = create_ssh_client(config)

    # Create project directory on pod
    ssh_exec(ssh, f"mkdir -p {REMOTE_PROJECT_DIR}")

    # Use SCP for file transfer
    scp_client = SCPClient(ssh.get_transport(), progress=_scp_progress)

    synced = 0
    for item in SYNC_INCLUDES:
        local_path = os.path.join(ROOT, item)
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {item}")
            continue

        remote_path = f"{REMOTE_PROJECT_DIR}/{item.rstrip('/')}"

        if os.path.isdir(local_path):
            print(f"\n  Uploading directory: {item}")
            # Create remote directory
            ssh_exec(ssh, f"mkdir -p {remote_path}")
            try:
                scp_client.put(local_path, remote_path=remote_path, recursive=True)
                synced += 1
            except Exception as e:
                print(f"    ERROR: {e}")
        else:
            print(f"\n  Uploading file: {item}")
            try:
                scp_client.put(local_path, remote_path=remote_path)
                synced += 1
            except Exception as e:
                print(f"    ERROR: {e}")

    scp_client.close()

    # Also upload the runpod setup script
    setup_script = os.path.join(ROOT, "runpod_setup.sh")
    if os.path.exists(setup_script):
        print(f"\n  Uploading setup script...")
        scp_client2 = SCPClient(ssh.get_transport())
        scp_client2.put(setup_script, f"{REMOTE_PROJECT_DIR}/runpod_setup.sh")
        scp_client2.close()

    ssh.close()
    print(f"\n  Synced {synced} items to {REMOTE_PROJECT_DIR}")
    print(f"  Next: python runpod_controller.py install")


def _scp_progress(filename, size, sent):
    """SCP transfer progress callback."""
    pct = sent * 100 // size if size > 0 else 100
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    name = filename.decode() if isinstance(filename, bytes) else filename
    short_name = os.path.basename(name)
    print(f"    [{bar}] {pct:3d}% {short_name} ({sent//1024}/{size//1024} KB)", end="\r")
    if sent == size:
        print()


def cmd_install(args):
    """Install dependencies on the pod."""
    config = ensure_api_key()

    print("=" * 60)
    print("Installing Dependencies on RunPod")
    print("=" * 60)

    ssh = create_ssh_client(config)

    # Upload and run setup script
    setup_script = os.path.join(ROOT, "runpod_setup.sh")
    if not os.path.exists(setup_script):
        print("ERROR: runpod_setup.sh not found. Creating it first...")
        _create_setup_script()

    print("  Uploading setup script...")
    scp_client = SCPClient(ssh.get_transport())
    scp_client.put(setup_script, f"{REMOTE_PROJECT_DIR}/runpod_setup.sh")
    scp_client.close()

    print("  Running setup (this takes 5-10 minutes)...\n")
    ssh_exec(ssh, f"chmod +x {REMOTE_PROJECT_DIR}/runpod_setup.sh")
    exit_code = ssh_exec(
        ssh,
        f"cd {REMOTE_PROJECT_DIR} && bash runpod_setup.sh 2>&1",
        stream=True,
        timeout=1200,
    )

    ssh.close()
    if exit_code == 0:
        print("\n  Installation complete!")
        print("  Next: python runpod_controller.py train")
    else:
        print(f"\n  Installation finished with exit code {exit_code}")
        print("  Check logs above for errors.")


def cmd_train(args):
    """Start training on the pod."""
    config = ensure_api_key()

    print("=" * 60)
    print("Starting FitFusion QLoRA Training on RunPod")
    print("=" * 60)

    ssh = create_ssh_client(config)

    # Check GPU info on pod
    out, _, _ = ssh_exec(ssh, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"  GPU: {out.strip()}")

    # Build training command
    train_cmd = args.cmd if hasattr(args, 'cmd') and args.cmd else None

    if not train_cmd:
        train_cmd = (
            f"cd {REMOTE_PROJECT_DIR} && "
            f"nohup python -u IDM-VTON/train_xl_qlora.py "
            f"  --pretrained_model_name_or_path /workspace/model_cache/IDM-VTON "
            f"  --pretrained_garmentnet_path /workspace/model_cache/IDM-VTON "
            f"  --pretrained_ip_adapter_path /workspace/model_cache/IDM-VTON/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin "
            f"  --image_encoder_path /workspace/model_cache/IDM-VTON/image_encoder "
            f"  --data_dir {REMOTE_PROJECT_DIR}/data/fitfusion_vitonhd "
            f"  --output_dir {REMOTE_PROJECT_DIR}/output_qlora "
            f"  --train_batch_size 2 "
            f"  --gradient_accumulation_steps 8 "
            f"  --learning_rate 2e-4 "
            f"  --num_train_epochs 50 "
            f"  --mixed_precision bf16 "
            f"  --gradient_checkpointing "
            f"  --use_8bit_adam "
            f"  --enable_xformers_memory_efficient_attention "
            f"  --lora_rank 16 "
            f"  --lora_alpha 32 "
            f"  --num_measurement_tokens 4 "
            f"  --logging_steps 500 "
            f"  --checkpointing_epoch 10 "
            f"  > {REMOTE_PROJECT_DIR}/training.log 2>&1 &"
        )

    print(f"\n  Running: {train_cmd[:120]}...")
    out, err, code = ssh_exec(ssh, train_cmd)

    # Get the background PID
    out2, _, _ = ssh_exec(ssh, "pgrep -f train_xl_qlora.py | head -1")
    pid = out2.strip()

    if pid:
        print(f"  Training started! PID: {pid}")
        config["training_pid"] = pid
        config["training_started"] = datetime.now().isoformat()
        save_config(config)
        print(f"\n  Monitor: python runpod_controller.py logs")
        print(f"  Or SSH in and run: tail -f {REMOTE_PROJECT_DIR}/training.log")
    else:
        print("  WARNING: Could not find training process.")
        print("  Check manually: python runpod_controller.py logs")

    ssh.close()


def cmd_logs(args):
    """Stream training logs from pod."""
    config = ensure_api_key()

    print("=" * 60)
    print("Training Logs (Ctrl+C to stop)")
    print("=" * 60)

    ssh = create_ssh_client(config)

    # Check if training is running
    out, _, _ = ssh_exec(ssh, "pgrep -f 'train_xl_qlora.py\\|train_xl.py' | head -1")
    pid = out.strip()
    if pid:
        print(f"  Training process active (PID: {pid})\n")
    else:
        print("  WARNING: No training process found. Showing last logs.\n")

    # Show last N lines + follow
    n_lines = args.lines if hasattr(args, 'lines') and args.lines else 50
    try:
        if pid:
            ssh_exec(ssh, f"tail -n {n_lines} -f {REMOTE_PROJECT_DIR}/training.log",
                     stream=True, timeout=0)
        else:
            ssh_exec(ssh, f"tail -n {n_lines} {REMOTE_PROJECT_DIR}/training.log",
                     stream=True)
    except KeyboardInterrupt:
        print("\n\n  Log streaming stopped.")

    ssh.close()


def cmd_download(args):
    """Download checkpoints and outputs from pod."""
    config = ensure_api_key()

    print("=" * 60)
    print("Downloading Results from RunPod")
    print("=" * 60)

    ssh = create_ssh_client(config)

    # Check what's available
    out, _, _ = ssh_exec(ssh, f"ls -la {REMOTE_PROJECT_DIR}/output/ 2>/dev/null || echo 'NO OUTPUT DIR'")
    print(f"  Remote output:\n{out}")

    local_download_dir = os.path.join(ROOT, "runpod_output")
    os.makedirs(local_download_dir, exist_ok=True)

    scp_client = SCPClient(ssh.get_transport(), progress=_scp_progress)

    # Download training log
    try:
        print(f"\n  Downloading training.log...")
        scp_client.get(f"{REMOTE_PROJECT_DIR}/training.log",
                       os.path.join(local_download_dir, "training.log"))
    except Exception as e:
        print(f"  No training.log: {e}")

    # Download output directory (checkpoints, samples)
    try:
        print(f"\n  Downloading output directory...")
        scp_client.get(f"{REMOTE_PROJECT_DIR}/output",
                       local_download_dir, recursive=True)
        print(f"\n  Downloaded to: {local_download_dir}")
    except Exception as e:
        print(f"  Error downloading output: {e}")

    scp_client.close()
    ssh.close()


def cmd_exec(args):
    """Execute an arbitrary command on the pod."""
    config = ensure_api_key()
    ssh = create_ssh_client(config)

    cmd = " ".join(args.remote_cmd) if hasattr(args, 'remote_cmd') else ""
    if not cmd:
        cmd = input("Command to run on pod: ").strip()

    if not cmd:
        print("No command specified.")
        return

    print(f"  Running: {cmd}\n")
    ssh_exec(ssh, cmd, stream=True)
    ssh.close()


def cmd_stop(args):
    """Stop the pod (preserves volume, stops billing)."""
    config = ensure_api_key()
    pod_id = config.get("pod_id")
    if not pod_id:
        print("No pod to stop.")
        return

    try:
        result = runpod.stop_pod(pod_id)
        print(f"  Pod {pod_id} stopped. Volume data preserved.")
        print(f"  Resume with: python runpod_controller.py resume")
    except Exception as e:
        print(f"ERROR stopping pod: {e}")


def cmd_resume(args):
    """Resume a stopped pod."""
    config = ensure_api_key()
    pod_id = config.get("pod_id")
    if not pod_id:
        print("No pod to resume. Run: python runpod_controller.py create")
        return

    try:
        result = runpod.resume_pod(pod_id, gpu_count=1)
        print(f"  Pod {pod_id} resuming...")
        print("  Wait 1-2 minutes, then: python runpod_controller.py status")
    except Exception as e:
        print(f"ERROR resuming pod: {e}")


def cmd_terminate(args):
    """Permanently delete the pod (IRREVERSIBLE)."""
    config = ensure_api_key()
    pod_id = config.get("pod_id")
    if not pod_id:
        print("No pod to terminate.")
        return

    print(f"  WARNING: This will permanently delete pod {pod_id}")
    print(f"  All data on the pod volume will be LOST.")
    confirm = input("  Type 'DELETE' to confirm: ").strip()

    if confirm == "DELETE":
        try:
            runpod.terminate_pod(pod_id)
            print(f"  Pod {pod_id} terminated.")
            config.pop("pod_id", None)
            config.pop("ssh_ip", None)
            config.pop("ssh_port", None)
            config.pop("training_pid", None)
            save_config(config)
        except Exception as e:
            print(f"ERROR terminating pod: {e}")
    else:
        print("  Cancelled.")


def cmd_cost(args):
    """Show current session cost."""
    config = ensure_api_key()
    pod_id = config.get("pod_id")
    if not pod_id:
        print("No active pod.")
        return

    try:
        pod_info = runpod.get_pod(pod_id)
        cost_per_hr = pod_info.get("costPerHr", 0)
        uptime = pod_info.get("uptimeSeconds", 0)
        status = pod_info.get("desiredStatus", "UNKNOWN")

        session_cost = cost_per_hr * uptime / 3600
        print(f"\n  Status: {status}")
        print(f"  Rate: ${cost_per_hr}/hr")
        print(f"  Uptime: {timedelta(seconds=uptime)}")
        print(f"  Current cost: ${session_cost:.2f}")

        if status == "RUNNING":
            print(f"\n  Projected costs:")
            for hours in [1, 2, 4, 8, 12, 24]:
                proj = cost_per_hr * hours
                print(f"    {hours:2d} hours: ${proj:.2f}")
    except Exception as e:
        print(f"ERROR: {e}")


# ═══════════════════════════════════════════════════════════════════════
# SETUP SCRIPT GENERATION
# ═══════════════════════════════════════════════════════════════════════
def _create_setup_script():
    """Generate the runpod_setup.sh script."""
    script_path = os.path.join(ROOT, "runpod_setup.sh")
    # This is generated by the create_file call below
    print(f"  Setup script should be at: {script_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="FitFusion RunPod Controller — manage cloud GPU training from VS Code"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # setup
    subparsers.add_parser("setup", help="First-time setup (API key + SSH key)")

    # gpus
    subparsers.add_parser("gpus", help="List available GPUs + pricing")

    # create
    p_create = subparsers.add_parser("create", help="Create a training pod")
    p_create.add_argument("--gpu", help="GPU type ID (e.g., 'NVIDIA A40')")

    # status
    subparsers.add_parser("status", help="Show pod status")

    # ssh
    subparsers.add_parser("ssh", help="Print SSH command")

    # sync
    subparsers.add_parser("sync", help="Upload code + data to pod")

    # install
    subparsers.add_parser("install", help="Install dependencies on pod")

    # train
    p_train = subparsers.add_parser("train", help="Start training")
    p_train.add_argument("--cmd", help="Custom training command")

    # logs
    p_logs = subparsers.add_parser("logs", help="Stream training logs")
    p_logs.add_argument("--lines", type=int, default=50, help="Lines to show")

    # download
    subparsers.add_parser("download", help="Download results from pod")

    # exec
    p_exec = subparsers.add_parser("exec", help="Run command on pod")
    p_exec.add_argument("remote_cmd", nargs="*", help="Command to execute")

    # stop
    subparsers.add_parser("stop", help="Stop pod (keeps data)")

    # resume
    subparsers.add_parser("resume", help="Resume stopped pod")

    # terminate
    subparsers.add_parser("terminate", help="DELETE pod permanently")

    # cost
    subparsers.add_parser("cost", help="Show session cost")

    args = parser.parse_args()

    commands = {
        "setup": cmd_setup,
        "gpus": cmd_gpus,
        "create": cmd_create,
        "status": cmd_status,
        "ssh": cmd_ssh,
        "sync": cmd_sync,
        "install": cmd_install,
        "train": cmd_train,
        "logs": cmd_logs,
        "download": cmd_download,
        "exec": cmd_exec,
        "stop": cmd_stop,
        "resume": cmd_resume,
        "terminate": cmd_terminate,
        "cost": cmd_cost,
    }

    if not args.command:
        parser.print_help()
        return

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

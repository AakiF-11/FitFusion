# FitFusion — Size-Aware Virtual Try-On

FitFusion is a next-generation sizing and virtual try-on architecture designed to run on top of state-of-the-art diffusion models (like IDM-VTON). Instead of standard "one-size-fits-all" image overlays, FitFusion dynamically recalculates fit tension, fabric physics, and dimensional scaling across different catalog clothing sizes to show users exactly how a specific size will drape on their specific body.

## Project Development Status

We have successfully integrated the pipeline to run end-to-end on RunPod GPU instances. The core development is currently in **Phase 5: GPU Inference and Refinement**.

### 1. Dynamic Brand Catalog System (`brand_catalog.py`)
- **Status:** Completed
- **Details:** Built a B2B catalog ingestion system that processes JSON manifests containing garments, available sizes, and reference model images.
- **Features:** 
  - Standardized sizing: A deterministic mapping system standardizes different brand sizing conventions (e.g. UK sizing, numeric sizing, S/M/L) into standard universal size labels (XS-4XL).
  - Body Matching: Employs a Height-Gated Euclidean Filter. First, a hard filter parses out reference models strictly within 5-10cm of the customer's height. Then, a weighted Euclidean distance calculation maps the visual geometric appearance based strictly on (Bust, Waist, Hips).

### 2. Size-Aware Physics Engine (`size_aware_vton.py`)
- **Status:** Completed
- **Details:** A mathematically grounded 7-layer physics controller determining warp tension parameters, localized body mapping, inpainting thresholds, and standard diffusion settings based on customer measurements vs actual garment spec.
- **Features:**
  - **Texture Preserving Mask Scaling:** Dynamic `width_ratio` calculation is applied *exclusively* to scale the agnostic boundary masks. The original garment image tensors retain pristine 1:1 aspect ratios to prevent textural hallucination and preserve high-frequency details (e.g., logos).
  - Tension parameter configurations tailored to different fabric behaviors and garment types.
  - **Prompt Stripping:** Generative prompts only carry pure garment identity structures. Text conditional hallucination of tight/loose physics (e.g., "tight fit, stretching fabric") is completely decoupled. Fit conditioning is entirely driven by explicit mask geometry scaling.

### 3. GPU Try-On End-to-End Integrations (`run_tryon.py` and `tryon_api.py`)
- **Status:** Completed
- **Details:** Configured the custom UNet extraction to accurately ingest tensor paths from the Stable Diffusion XL based IDM-VTON pipeline.
- **Features:**
  - Automated product detection and parameter passing into the neural generative pipeline.
  - Orchestrates seamless integration between the brand catalog, physics engine, and the remote IDM-VTON repository running on Linux pods.

### 4. Semantic Body Parsing & Masking
- **Status:** Completed
- **Details:** High-fidelity try-ons require precise agnostic masks so the diffusion model knows exactly where the user's body ends and the background begins. Furthermore, we must reject hallucinated mask boundaries predicting impossible arm geometries.
- **Features:**
  - Integrated DensePose extraction natively via inline script routing directly injected into the diffusion attention layers.
  - Utilizes native OpenPose (`body_pose_model.pth`) and HumanParsing (`parsing_atr.onnx`, `parsing_lip.onnx`) models for accurate segmentation without baseline edge bleeding.
  - **Mask Confidence Scoring (`confidence_scorer.py`):** An explicit heuristic layer calculates the bounding box of the user's arms using OpenPose joints (Shoulder -> Elbow -> Wrist). Any SCHP mask generation proposing garment pixels drastically outside this structural plane results in a confidence score penalty (< 0.85) to cleanly halt execution before generative waste. Includes an integrated None-Type failsafe array to prevent pipeline crashes when processing cropped photos (e.g. mirror selfies where wrists/elbows are omitted).

### 5. Production Pipeline Fortifications
- **Status:** Completed
- **Details:** Secured the API against deployment-level vulnerabilities like memory overflow, bounding hallucinogens, and generative artifacting.
- **Features:**
  - **Redis Concurrency (`run_tryon.py`):** Encapsulated the core IDM-VTON neural execution loop inside a Celery task pipeline (`tryon_tasks`) hooked to a Redis broker. This prevents simultaneous API requests from causing GPU Out-of-Memory (OOM) network crashes, and securely isolated tensor processing from standard binary Pickling failures.
  - **Background Void Prevention (`fitfusion/utils/preprocessing.py`):** Automatically standardizes customer imagery by stripping raw backgrounds via `rembg` and replacing transparent alpha zones with solid `(R:238, G:238, B:238)` studio gray before passing to IDM-VTON. This stops the generative network from hallucinating objects or limbs when we mathematically shrink down garments to smaller size matrices.
  - **Semantic Skin Compositing (`fitfusion/masking/compositor.py`):** Defends against image erasure by pulling explicit skin zone mappings (Face, Neck, Arms, Legs) out of the SCHP labels, and utilizing `cv2.addWeighted` compositing methodologies to perfectly layer original skin (and tattoos) cleanly back over the artificially rendered garment pixels.
  - **S3 Asynchronous Asset Pipeline (`run_tryon.py`):** Employs robust `boto3` and `requests` layers wrapped around the primary inference entrypoint, transitioning from fragile local SSH/SCP pushed filepath strings to secure HTTPS downloads direct to the Linux processing pod prior to IDM-VTON initialization, preventing tensor EOF file corruption faults.

---

## Architecture Stack
* **Inference Platform:** RunPod (NVIDIA RTX GPUs, PyTorch environment)
* **Underlying Generative Core:** IDM-VTON (Stable Diffusion XL fine-tuned for Try-On)
* **API Queue / Routing:** Celery workers backed by a Redis broker.
* **Pose Extraction:** Detectron2 (DensePose) & OpenPose
* **Human Semantic Mapping:** Self-Correction-Human-Parsing (SCHP)
* **Local Workspace:** Windows 10 Host syncing with Remote Linux Pods via SSH/SCP pipelines.

## Inference Usage
The core CLI process command looks like:
```bash
python run_tryon.py \
    --product_id "cropped-borg-aviator-jacket-black" \
    --brand_id "snag_tights" \
    --customer_photo "s3://fitfusion-uploads/customer_photo.jpg" \
    --bust 92 --waist 76 --hips 100 --height 170 \
    --size M \
    --output "/workspace/FitFusion/data/tryon_customer_M.png"
```
The script will return physical tension calculations and produce the rendered standard image showing the result.

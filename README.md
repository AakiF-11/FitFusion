# FitFusion QLoRA Architecture

**Size-Conditioned Virtual Try-On using QLoRA Fine-Tuning**

---

## 📦 Project Structure

```
FitFusion/
├── fitfusion/                    # Core QLoRA modules (4 files, 2160 lines)
│   ├── core/lora_bridge.py       # Trigger token mapping (620 lines)
│   ├── tools/generate_training_data.py  # Data generation (470 lines)
│   ├── train/qlora_trainer.py    # Training logic (680 lines)
│   └── pipelines/v65_optimized.py # Inference (390 lines)
│
├── qlora_training_data/          # Source data (644 examples, 86MB)
├── training_data/                # Processed data (644 pairs with captions)
├── requirements_qlora.txt        # Dependencies
└── *.md                          # Documentation
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_qlora.txt

# 2. Analyze training data
python analyze_training_data.py

# 3. Check status
# Read TRAINING_BLOCKER_STATUS.md for current options
```

---

## 📊 What's Included

### ✅ Complete Infrastructure
- **4 QLoRA Modules** - 2,160 lines of production code
- **644 Training Examples** - Perfectly balanced across 7 fit classes
- **Trigger Token System** - `[FF_TIGHT]`, `[FF_BAGGY]`, etc.
- **6GB VRAM Optimized** - Works on RTX 3050

### ⚠️ Current Status
- Training Data: ✅ Ready
- Infrastructure: ✅ Complete  
- Training: ⚠️ Blocked (OOTDiffusion not trainable with LoRA)

See **[TRAINING_BLOCKER_STATUS.md](TRAINING_BLOCKER_STATUS.md)** for options.

---

## 🎯 Fit Classes

```
[FF_BURSTING]  - H < 0.85  | Extreme tension
[FF_TIGHT]     - 0.85-0.92 | Form-fitting
[FF_SNUG]      - 0.92-0.98 | Close fit
[FF_PERFECT]   - 0.98-1.08 | Classic fit
[FF_RELAXED]   - 1.08-1.20 | Loose
[FF_BAGGY]     - 1.20-1.40 | Oversized
[FF_TENT]      - H > 1.40  | Extremely oversized
```

---

## 📚 Documentation

- **[TRAINING_BLOCKER_STATUS.md](TRAINING_BLOCKER_STATUS.md)** - Current status & options
- **[QLORA_ARCHITECTURE_GUIDE.md](QLORA_ARCHITECTURE_GUIDE.md)** - Complete technical guide
- **[TRAINING_DATA_READY.md](TRAINING_DATA_READY.md)** - Dataset documentation

---

## 💡 What You Have

✅ Production-ready QLoRA infrastructure  
✅ 644 high-quality training examples  
✅ Complete documentation  
⚠️ Need trainable base model or cloud GPU access

**Next:** See TRAINING_BLOCKER_STATUS.md for path forward.

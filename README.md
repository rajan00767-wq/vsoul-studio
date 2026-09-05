# Vsoul Studio

Local Qwen Image Edit + Qwen-VL Enhance and Uniform app. Client UI is `index.html`.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python download_models.py
python main.py
```

`download_models.py` downloads only the models used by this workspace: Qwen Image Edit 2511, Qwen2.5-VL analysis, and the optional local 2x/4x export model.

Open http://127.0.0.1:8000

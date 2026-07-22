# 🎨 Neural Style Transfer

> Blend any photo with the texture of any painting using deep learning — powered by VGG19, TensorFlow, and Streamlit.

![Neural Style Transfer Banner](assets\image.png)

---

## What it does

Upload a **content image** (your photo) and a **style image** (any painting or artwork). The app blends them together — keeping the structure of your photo while painting it in the texture and colors of the artwork.

| Content Image | Style Image |     Output      |
| :-----------: | :---------: | :-------------: |
|  Your photo   |   any art   | Stylized result |

---

## How it works

The model uses a pretrained **VGG19** CNN (trained on ImageNet) to extract:

- **Content features** from deep layers — captures shapes and structure
- **Style features** via Gram matrices from early layers — captures textures and patterns

It then optimizes a generated image to minimize:

```
Total Loss = Style Loss × 1e4  +  Content Loss × 1.0  +  TV Loss × 30.0
```

---

## Tech Stack

| Layer            | Tool                  |
| ---------------- | --------------------- |
| Deep Learning    | TensorFlow 2.x, VGG19 |
| Web App          | Streamlit             |
| Image Processing | Pillow, NumPy         |
| Deployment       | Render                |
| CI/CD            | GitHub Actions        |
| Containerization | Docker                |

---

## Project Structure

```
neural-style-transfer/
├── app.py                      # Streamlit UI
├── model.py                    # NST logic (VGG19, loss, training loop)
├── requirements.txt
├── Dockerfile
├── runtime.txt                 # python-3.11.0
├── assets/
│   └── banner.png
└── .github/
    └── workflows/
        └── deploy.yml          # CI/CD pipeline
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Deepak4053/Neural_style_transfer.git
cd neural-style-transfer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Run with Docker

```bash
# Build
docker build -t nst-app .

# Run
docker run -p 8501:8501 nst-app
```

Open `http://localhost:8501`.

---

## Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set the following:

| Field         | Value                                                               |
| ------------- | ------------------------------------------------------------------- |
| Environment   | `Python`                                                            |
| Build Command | `pip install -r requirements.txt`                                   |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Instance Type | `Free`                                                              |

5. Click **Deploy** — your app will be live at `https://your-app.onrender.com`

---

## CI/CD Pipeline

Every push to `main` automatically:

```
Push to GitHub
      ↓
Run tests (smoke test model import)
      ↓
Build Docker image → push to Docker Hub
      ↓
Trigger Render deploy hook
      ↓
App is live ✅
```

Add these secrets in GitHub → Settings → Secrets:

| Secret               | Value                    |
| -------------------- | ------------------------ |
| `DOCKER_USERNAME`    | Your Docker Hub username |
| `DOCKER_TOKEN`       | Docker Hub access token  |
| `RENDER_DEPLOY_HOOK` | Render deploy hook URL   |

---

## Tuning the Results

Edit these constants in `model.py`:

```python
STYLE_WEIGHT   = 1e4    # higher = stronger style texture
CONTENT_WEIGHT = 1.0    # higher = more like original photo
TV_WEIGHT      = 30.0   # higher = smoother, less noise
```

| Result looks like...    | Fix                               |
| ----------------------- | --------------------------------- |
| Too much noise / grainy | Increase `TV_WEIGHT` to 50-100    |
| Style too subtle        | Increase `STYLE_WEIGHT` to `1e5`  |
| Photo unrecognizable    | Increase `CONTENT_WEIGHT` to 5-10 |
| Too slow                | Reduce `IMG_SIZE` to `(224, 224)` |

---

## Requirements

```
streamlit==1.35.0
tensorflow-cpu==2.16.1
numpy==1.26.4
Pillow==10.3.0
```

---

## Author

**Deepak** — B.Tech ECE, MNNIT Allahabad  
[GitHub](https://github.com/Deepak4053) · [LinkedIn](https://linkedin.com/in/deepakgaund4053)

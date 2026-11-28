import sys
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_REPO = "Musfirah-umar/MAE-VAE-DDPM-RAG"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOTICE = "For educational use only. Not medical advice."


# ---------------------------------------------------------------- helpers
def get_file(local_rel, hf_path):
    """Use the local file if it exists, otherwise download it from Hugging Face."""
    p = ROOT / local_rel
    if p.exists():
        return str(p)
    return hf_hub_download(HF_REPO, hf_path)


def to_pil(arr01, scale=4):
    """2-D array in [0,1] -> enlarged grayscale PIL image (28x28 is tiny on screen)."""
    a = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)


# ---------------------------------------------------------------- Tab 1: MAE (partner)
def mae_predict(tile_index):
    """TODO (partner): load the EuroSAT test split, take tile `tile_index`, run her MAE
    with 75% masking, and return:
        masked_input_image, reconstruction_image, original_image, "predicted class: ..."
    Use her model from part_a_mae and her weights (mae/mae_weights.pth and
    mae/mae_classifier.pth on Hugging Face)."""
    return None, None, None, "MAE tab not wired up yet."


# ---------------------------------------------------------------- Tab 2: VAE
_vae = {}


def vae_model():
    if "model" not in _vae:
        from part_b_vae.model import VAE
        m = VAE(d=16)
        m.load_state_dict(torch.load(get_file("results/vae.pt", "vae/vae.pt"), map_location="cpu"))
        m.eval()
        _vae["model"] = m
    return _vae["model"]


def uhat_test():
    if "x" not in _vae:
        from starter.uhat import load_uhat_characters
        x, l, names = load_uhat_characters(root=str(ROOT / "starter" / "data" / "uhat"), split="test")
        _vae.update(x=x, l=l, names=names)
    return _vae["x"], _vae["l"], _vae["names"]


def vae_class_names():
    return uhat_test()[2]


def vae_random():
    with torch.no_grad():
        imgs = torch.sigmoid(vae_model().decode(torch.randn(16, 16)))     # z ~ N(0, I)
    return [to_pil(im[0].numpy()) for im in imgs]


def vae_morph(name_a, name_b, alpha):
    x, l, names = uhat_test()
    xa = x[(l == names.index(name_a)).nonzero()[0]]       # first test image of each character
    xb = x[(l == names.index(name_b)).nonzero()[0]]
    with torch.no_grad():
        mu_a, _ = vae_model().encode(xa)
        mu_b, _ = vae_model().encode(xb)
        out = torch.sigmoid(vae_model().decode((1 - alpha) * mu_a + alpha * mu_b))
    return to_pil(xa[0, 0].numpy()), to_pil(out[0, 0].numpy()), to_pil(xb[0, 0].numpy())


# ---------------------------------------------------------------- Tab 3: Diffusion
_ddpm = {}


def ddpm_model():
    if "unet" not in _ddpm:
        from diffusers import UNet2DModel
        unet = UNet2DModel(
            sample_size=28, in_channels=1, out_channels=1, layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            num_class_embeds=40)
        ck = torch.load(get_file("results/part_c_ddpm.pt", "ddpm/part_c_ddpm.pt"), map_location=DEVICE)
        unet.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        _ddpm["unet"] = unet.to(DEVICE).eval()
    return _ddpm["unet"]


@torch.no_grad()
def ddpm_generate(char_id, progress=gr.Progress()):
    unet = ddpm_model()
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    y = torch.full((8,), int(char_id), dtype=torch.long, device=DEVICE)
    x = torch.randn(8, 1, 28, 28, device=DEVICE)
    for t in range(T - 1, -1, -1):                       # 0-indexed: 999 ... 0
        tb = torch.full((8,), t, dtype=torch.long, device=DEVICE)
        eps = unet(x, tb, class_labels=y).sample
        mean = (x - betas[t] / torch.sqrt(1.0 - alpha_bars[t]) * eps) / torch.sqrt(alphas[t])
        x = mean + torch.sqrt(betas[t]) * torch.randn_like(x) if t > 0 else mean
        if t % 50 == 0:
            progress((T - t) / T, desc="Denoising")
    x = ((x.clamp(-1, 1) + 1) / 2).cpu()
    return [to_pil(im[0].numpy()) for im in x]


# ---------------------------------------------------------------- Tab 4: RAG
_rag = {}


def load_chunks():
    """Chunk texts + abstract ids: from Hugging Face if uploaded, else rebuilt from BioASQ."""
    try:
        import json
        texts = json.load(open(hf_hub_download(HF_REPO, "rag/chunks.json")))
        doc_ids = np.load(hf_hub_download(HF_REPO, "rag/chunk_doc_ids.npy"))
        return texts, doc_ids
    except Exception:
        from starter.bioasq import load_corpus
        from part_d_rag.chunking import build_chunks
        return build_chunks(load_corpus())


def rag_resources():
    if "index" not in _rag:
        from part_d_rag.embedding import load_encoder
        from part_d_rag.search import Index
        from part_d_rag.generation import load_generator
        emb = np.load(get_file("results/index/chunk_embeddings.npy", "rag/chunk_embeddings.npy"))
        texts, doc_ids = load_chunks()
        assert emb.shape[0] == len(texts), "embeddings and chunks do not match"
        _rag.update(index=Index(emb, doc_ids), texts=texts, doc_ids=doc_ids,
                    enc=load_encoder(DEVICE), gen=load_generator(DEVICE))
    return _rag


def rag_answer(question):
    if not question or not question.strip():
        return "Please type a question.", ""
    from part_d_rag.embedding import embed_texts
    from part_d_rag.generation import answer_question, cited_numbers
    r = rag_resources()
    etok, enc = r["enc"]
    gtok, gen = r["gen"]
    qv = embed_texts([question], etok, enc, DEVICE)[0]
    idx, scores = r["index"].search_chunks(qv, k=5)
    passages = [r["texts"][i] for i in idx]
    answer = answer_question(question, passages, gtok, gen, DEVICE)
    cited = set(cited_numbers(answer))
    answer_md = f"### Answer\n\n{answer}\n\n> **{NOTICE}**"
    parts = ["### Retrieved passages\n"]
    for n, (i, s) in enumerate(zip(idx, scores), start=1):
        tag = "  (cited)" if n in cited else ""
        parts.append(f"**[{n}] abstract id {int(r['doc_ids'][i])}** (score {s:.3f}){tag}\n\n{r['texts'][i]}\n")
    return answer_md, "\n".join(parts)


# ---------------------------------------------------------------- UI
with gr.Blocks(title="MAE, VAE, Diffusion and RAG") as demo:
    gr.Markdown("# Four ways to generate: MAE, VAE, Diffusion and RAG")

    with gr.Tab("MAE"):
        tile = gr.Slider(0, 5399, value=0, step=1, label="EuroSAT test tile index")
        btn_mae = gr.Button("Run MAE")
        with gr.Row():
            img_masked = gr.Image(label="Masked input (75%)")
            img_recon = gr.Image(label="Reconstruction")
            img_orig = gr.Image(label="Original")
        txt_class = gr.Textbox(label="Predicted land-cover class")
        btn_mae.click(mae_predict, tile, [img_masked, img_recon, img_orig, txt_class])

    with gr.Tab("VAE"):
        gr.Markdown("### New random characters (z ~ N(0, I))")
        btn_vae = gr.Button("Generate 16 new characters")
        gal_vae = gr.Gallery(label="Samples", columns=8, rows=2, height=300)
        btn_vae.click(vae_random, None, gal_vae)

        gr.Markdown("### Morph between two characters")
        with gr.Row():
            dd_a = gr.Dropdown(label="First character", choices=[])
            dd_b = gr.Dropdown(label="Second character", choices=[])
        alpha = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Morph amount")
        with gr.Row():
            m_a = gr.Image(label="Character A (original)")
            m_mid = gr.Image(label="Morph")
            m_b = gr.Image(label="Character B (original)")
        for comp in (dd_a, dd_b, alpha):
            comp.change(vae_morph, [dd_a, dd_b, alpha], [m_a, m_mid, m_b])

        def init_dropdowns():
            names = vae_class_names()
            return (gr.update(choices=names, value=names[0]),
                    gr.update(choices=names, value=names[1]))
        demo.load(init_dropdowns, None, [dd_a, dd_b])

    with gr.Tab("Diffusion"):
        gr.Markdown("Class ids are used because the dataset file has no names. 1000 denoising steps per click.")
        dd_char = gr.Dropdown(label="Character id", choices=[str(i) for i in range(40)], value="0")
        btn_d = gr.Button("Generate 8 samples")
        gal_d = gr.Gallery(label="Samples", columns=8, rows=1, height=200)
        btn_d.click(ddpm_generate, dd_char, gal_d)

    with gr.Tab("RAG"):
        q = gr.Textbox(label="Ask a biomedical research question",
                       placeholder="e.g. What is the mechanism of action of durvalumab?")
        btn_r = gr.Button("Answer")
        out_answer = gr.Markdown()
        out_passages = gr.Markdown()
        btn_r.click(rag_answer, q, [out_answer, out_passages])

if __name__ == "__main__":
    demo.launch()          # use demo.launch(share=True) in Colab
"""Sentence encoders. All embeddings are L2-normalised so that inner products are cosine similarities."""
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_id, pooling):
    """Return a function mapping a list of strings to an (n, d) float32 array of unit vectors."""
    if pooling == "st":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, device=DEVICE)

        def encode(texts):
            return model.encode(list(texts), batch_size=128, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        return encode

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()

    @torch.no_grad()
    def encode(texts):
        texts, out = list(texts), []
        for i in range(0, len(texts), 256):
            batch = tokenizer(texts[i:i + 256], padding=True, truncation=True, max_length=48,
                              return_tensors="pt").to(DEVICE)
            cls = model(**batch).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
    return encode

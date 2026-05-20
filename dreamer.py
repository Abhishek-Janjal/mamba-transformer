import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import List, Optional
from pyvis.network import Network
import matplotlib.pyplot as plt


class Dreamer:
    """Dreamer that builds a memory graph and provides visualization.

    - Stores token sequences as memories
    - Builds a k-NN graph between memories using mean token embeddings
    - Can 'dream' by traversing graph neighborhoods and performing short
      self-supervised updates using the model
    - Can export an interactive HTML graph (pyvis) or a PNG (matplotlib)
    """

    def __init__(self, max_mem: int = 2000, k: int = 4):
        self.max_mem = max_mem
        self.k = k
        self.memories: List[torch.Tensor] = []  # list of (L,) torch tensors on CPU
        self.graph = nx.Graph()

    def remember(self, batch_tokens: torch.Tensor):
        """Store each sequence from the batch into memory (keeps on CPU)."""
        batch = batch_tokens.detach().cpu()
        for seq in batch:
            if len(self.memories) >= self.max_mem:
                # drop oldest
                self.memories.pop(0)
            self.memories.append(seq.clone())

    def _compute_embeddings(self, embed_fn, device: torch.device):
        """Return tensor (N, D) of mean embeddings for each memory."""
        if len(self.memories) == 0:
            return None
        embs = []
        with torch.no_grad():
            for seq in self.memories:
                x = seq.unsqueeze(0).to(device)  # (1, L)
                e = embed_fn(x)  # (1, L, D)
                e_mean = e.mean(dim=1)  # (1, D)
                embs.append(e_mean.cpu())
        return torch.cat(embs, dim=0)  # (N, D)

    def build_graph(self, embed_fn, device: torch.device, top_k: Optional[int] = None):
        """Build k-NN similarity graph using embed_fn.

        embed_fn: callable that maps input_ids (B, L) -> embeddings (B, L, D)
        """
        if top_k is None:
            top_k = self.k
        self.graph.clear()
        N = len(self.memories)
        if N == 0:
            return self.graph

        embs = self._compute_embeddings(embed_fn, device)  # (N, D)
        embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
        sims = embs @ embs.t()  # (N, N)
        sims = sims.numpy()

        for i in range(N):
            self.graph.add_node(i, label=self._short_label(self.memories[i]))

        for i in range(N):
            row = sims[i].copy()
            row[i] = -1.0
            neighs = np.argpartition(-row, range(min(top_k, N - 1)))[:min(top_k, N - 1)]
            for j in neighs:
                weight = float(sims[i, j])
                if weight > 0.0:
                    self.graph.add_edge(i, int(j), weight=weight)
        return self.graph

    def _short_label(self, seq: torch.Tensor, max_chars: int = 60):
        # represent as hex of first tokens or truncated text if tokenizer provided later
        tokens = seq.tolist()
        text = " ".join(str(t) for t in tokens[:16])
        if len(text) > max_chars:
            text = text[:max_chars - 3] + '...'
        return text

    def visualize(self, out_html: str = 'dream_graph.html', notebook: bool = False, bgcolor: str = '#ffffff'):
        """Export interactive HTML using pyvis. Also saves a PNG fallback."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            raise RuntimeError('Graph is empty. Call build_graph() first with an embed function.')

        net = Network(height='800px', width='100%', bgcolor=bgcolor)
        net.force_atlas_2based()

        for n, data in self.graph.nodes(data=True):
            lbl = data.get('label', str(n))
            net.add_node(n, label=lbl)

        for u, v, data in self.graph.edges(data=True):
            w = data.get('weight', 1.0)
            net.add_edge(int(u), int(v), value=w)

        net.show(out_html)

    def dream(self, model, tokenizer, optimizer, device: torch.device, steps: int = 4, samples: int = 8):
        """Perform dreaming by sampling graph neighborhoods and training briefly."""
        if len(self.memories) < 2:
            return None

        # ensure graph is built
        def embed_fn(x: torch.Tensor):
            emb_layer = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else model.embed
            return emb_layer(x)

        self.build_graph(embed_fn, device=device)

        # sample some central nodes
        nodes = list(self.graph.nodes())
        if len(nodes) == 0:
            return None
        sampled = np.random.choice(nodes, size=min(samples, len(nodes)), replace=False)

        model.train()
        loss_accum = 0.0
        for _ in range(steps):
            for n in sampled:
                # build a prompt by concatenating neighbor sequences
                nbrs = list(self.graph.neighbors(n))[: self.k]
                parts = [self.memories[n]] + [self.memories[m] for m in nbrs]
                prompt = torch.cat(parts, dim=0).unsqueeze(0).to(device)  # (1, L')
                labels = prompt.clone()
                outputs = model(prompt, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_accum += float(loss.item())
        return loss_accum / (steps * max(1, len(sampled)))

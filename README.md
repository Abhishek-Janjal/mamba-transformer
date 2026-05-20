# Transformer-Mamba LLM with Dream-Based Memory Graph

A hybrid Mamba-Transformer architecture that incorporates:
- **Hybrid model**: alternating state-space (Mamba) and attention blocks
- **Dreaming**: periodic memory consolidation via k-NN graph discovery
- **Ollama-style chat**: interactive terminal-based inference
- **Graph visualization**: inspect memory connections and concepts over training

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Authenticate (if using gated Qwen model)
```bash
huggingface-cli login
# or set env var
export HUGGINGFACE_HUB_TOKEN="hf_...your_token..."
```

### 3. Train (defaults to safe small settings)
```bash
# Ensure memory optimizations:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python hybrid_llm.py
```

This trains a small hybrid model on the Qwen tokenizer + medical QA dataset (200 docs, 20 steps, batch=1). Training outputs:
- `model.pt` — trained model state
- `dream_graph.html` — interactive k-NN memory graph from training

### 4. Chat with the Trained Model
```bash
python inference.py --interactive --model_path model.pt
```

Commands in chat:
- `/system <text>` — set system prompt
- `/temp <value>` — set sampling temperature (default 0.7)
- `/clear` — reset conversation
- `/exit` or `quit` — exit

### 5. Build Graph from Saved Conversation (Optional)
```bash
# Save conversation to file (e.g., conversation.txt)
python build_dream_graph_from_history.py \
  --model_path model.pt \
  --conversation conversation.txt \
  --out dream_graph_from_chat.html

# Open in browser
xdg-open dream_graph_from_chat.html
```

## Architecture Overview

### Model Components

**`hybrid_llm.py`** — Training script with Mamba-Transformer hybrid model.

- **HybridConfig**: configurable model/training parameters
  - `hidden_size=128, num_layers=2` (small defaults for safe GPU runs)
  - `max_seq_len=64, batch_size=1` (reduces memory)
  - `num_steps=20` (short training for quick testing)
  
- **SimpleSSM**: state-space (Mamba-like) block with 1D convolution + parallel SSM dynamics
- **SimpleAttention**: standard multi-head attention
- **HybridBlock**: combines SSM or attention based on `layer_pattern` (e.g., "MMAMAMAM")
- **HybridModel**: full model with embeddings, hybrid layers, and language modeling head

**Training loop**:
1. Tokenize medical QA dataset (streaming, Qwen tokenizer)
2. Mixed-precision training with gradient accumulation
3. Periodic "dreaming": Dreamer builds k-NN graph of past memories, performs synthetic updates
4. Saves final model + dream graph visualization

---

### Memory & Dreaming

**`dreamer.py`** — Memory graph and dreaming logic.

**Dreamer class**:
- `remember(batch)` — stores token sequences from training
- `build_graph(embed_fn, device)` — computes similarity graph (k-NN based on token embeddings)
- `dream(model, tokenizer, optimizer, device)` — samples graph neighborhoods, synthesizes prompts, performs brief self-supervised updates
- `visualize(out_html)` — exports interactive pyvis graph showing memory connections

**Graph interpretation**:
- Nodes = past token sequences (memories)
- Edges = similarity (cosine distance between mean embeddings)
- Edge weight = similarity strength
- Dreaming = walking the graph and re-training on synthetic data

---

### Inference

**`inference.py`** — Ollama-style terminal chat interface.

Features:
- Streaming token generation (token-by-token output)
- System prompt customization (`/system`)
- Temperature control (`/temp`)
- Conversation history tracking
- Graceful error handling

---

### Dream Graph from Conversation

**`build_dream_graph_from_history.py`** — Builds k-NN memory graph from a saved conversation.

1. Reads conversation file (one turn per line)
2. Tokenizes each line
3. Builds k-NN graph using trained model embeddings
4. Exports interactive HTML visualization

---

## Usage Examples

### Train with Different Settings

```bash
# Larger model (more memory required)
python hybrid_llm.py --hidden-size 256 --num-layers 4 --batch-size 2 --max-seq-len 128

# Different dataset
python hybrid_llm.py \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --num-documents 500

# Local HybridModel (not Qwen)
python hybrid_llm.py --no-use-qwen --hidden-size 256 --num-layers 4
```

### Memory Optimization Tips

```bash
# Disable torch.compile (reduces overhead)
export USE_TORCH_COMPILE=0

# Enable GPU memory expansion
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Gradient accumulation (keep batch_size small, accumulate steps)
python hybrid_llm.py --grad-accum 4

# Run inference on CPU (slower but no OOM)
python inference.py --interactive --model_path model.pt --device cpu
```

---

## File Structure

```
.
├── hybrid_llm.py                      # Training with dream integration
├── dreamer.py                         # Memory graph + visualization
├── inference.py                       # Ollama-style chat CLI
├── build_dream_graph_from_history.py  # Graph from saved conversation
├── requirements.txt                   # Dependencies
├── gpu_monitor.py                     # (Optional) GPU monitoring
└── README.md                          # This file
```

---

## Customization

### CLI Flags (see `python hybrid_llm.py --help`)

**Model**:
- `--model-name` — HF model id (default: `Qwen/Qwen3-0.6B`)
- `--use-qwen` — use HF model instead of local HybridModel (default: enabled)
- `--hidden-size` — override hidden dimension
- `--num-layers` — override number of layers
- `--batch-size` — override batch size
- `--max-seq-len` — override sequence length

**Data**:
- `--dataset` — HF dataset id (default: medical QA)
- `--dataset-config` — dataset config name
- `--num-documents` — number of documents to tokenize

**Training**:
- `--dream-every` — steps between dream phases
- `--dream-steps` — optimizer steps per dream
- `--grad-accum` — gradient accumulation steps

---

## Troubleshooting

**CUDA Out of Memory**:
1. Set env vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. Reduce `--batch-size`, `--max-seq-len`, `--hidden-size`, `--num-layers`
3. Enable `--grad-accum` (accumulate gradients over smaller batches)
4. Disable `torch.compile`: `export USE_TORCH_COMPILE=0`
5. Use CPU: `--device cpu` (very slow, but works)

**HF Model Access (401/404)**:
1. Ensure you have access to the gated model on HuggingFace
2. Run `huggingface-cli login` and accept terms if needed
3. Verify token: `huggingface-cli whoami`
4. Or pass a local folder: `--model-name /path/to/local/qwen-folder`

**Deprecated `autocast()` warning**:
- This is fixed; uses `torch.amp.autocast(device_type='cuda')` now

---

## Example Workflow

```bash
# 1. Train (auto-generates dream_graph.html)
export HUGGINGFACE_HUB_TOKEN="hf_..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python hybrid_llm.py

# 2. Chat
python inference.py --interactive --model_path model.pt
# (save conversation to conversation.txt manually or via /save if implemented)

# 3. Visualize memory from chat
python build_dream_graph_from_history.py \
  --model_path model.pt \
  --conversation conversation.txt \
  --out dream_graph_from_chat.html

# 4. Open graphs
xdg-open dream_graph.html           # from training
xdg-open dream_graph_from_chat.html # from conversation
```

---

## References

- **Mamba**: [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.08956)
- **Hybrid architectures**: Combining SSMs with attention for complementary strengths
- **Dreaming**: Inspired by sleep-based memory consolidation in neuroscience; implemented as periodic k-NN graph walks
- **Karpathy's wiki-LLM idea**: Building associative memory graphs from sequential data

---

## License

See [LICENSE](LICENSE) file.
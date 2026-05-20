import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import argparse
from typing import Optional, List, Dict
from dataclasses import dataclass
from dreamer import Dreamer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from torch import amp

@dataclass
class HybridConfig:
    vocab_size: int = 50257
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 8
    ssm_state_size: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
    layer_pattern: str = "MMAMAMAM"
    
    # Training (smaller defaults for safe local runs)
    max_seq_len: int = 64
    batch_size: int = 1
    num_documents: int = 200
    learning_rate: float = 5e-4
    num_steps: int = 20
    
    dropout: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    
    def __post_init__(self):
        self.intermediate_size = self.expand_factor * self.hidden_size


class SimpleSSM(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.ssm_state_size
        
        self.in_proj = nn.Linear(config.hidden_size, self.intermediate_size * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.intermediate_size, self.intermediate_size,
            kernel_size=config.conv_kernel, groups=self.intermediate_size,
            padding=config.conv_kernel - 1, bias=False
        )
        self.x_proj = nn.Linear(self.intermediate_size, config.ssm_state_size * 2 + 1, bias=False)
        self.A = nn.Parameter(torch.randn(self.intermediate_size, self.ssm_state_size))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x = F.silu(x)
        
        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([1, self.ssm_state_size, self.ssm_state_size], dim=-1)
        delta = F.softplus(delta)
        
        # Simplified parallel SSM
        A = -torch.exp(self.A)
        decay = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        states = x.unsqueeze(-1) * B.unsqueeze(2) * decay
        y = (states * C.unsqueeze(2)).sum(dim=-1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return self.out_proj(y * F.silu(z))


class SimpleAttention(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout
        
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.out_proj(attn.transpose(1, 2).reshape(B, L, -1))


class HybridBlock(nn.Module):
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.mixer = SimpleSSM(config) if config.layer_pattern[layer_idx] == 'M' else SimpleAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


# Allow disabling torch.compile via environment (useful to reduce memory)
compile_fn = torch.compile if os.getenv('USE_TORCH_COMPILE', '0') == '1' else (lambda x: x)

@compile_fn
class HybridModel(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HybridBlock(config, i) for i in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Tie weights
        
        # Initialize
        self.apply(lambda m: torch.nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)
    
    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, self.config.vocab_size), labels[..., 1:].reshape(-1))
        return logits, loss


class TextDataset(Dataset):
    def __init__(self, tokens, max_length):
        self.tokens = tokens
        self.max_length = max_length
        
    def __len__(self):
        return len(self.tokens) // self.max_length
    
    def __getitem__(self, idx):
        start = idx * self.max_length
        return torch.tensor(self.tokens[start:start + self.max_length], dtype=torch.long)


class HFModelWrapper(nn.Module):
    """Wraps HF model to return (logits, loss) tuple like our HybridModel"""
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    
    def forward(self, input_ids, labels=None):
        output = self.model(input_ids=input_ids, labels=labels)
        return output.logits, output.loss
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def train(self, mode=True):
        self.model.train(mode)
        return self
    
    def eval(self):
        self.model.eval()
        return self



def main():
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere GPUs
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-0.6B', help='HF model id to fine-tune (optional)')
    parser.add_argument('--dataset', type=str,  default="Malikeh1375/medical-question-answering-datasets", help='Hugging Face dataset id to use')
    parser.add_argument('--dataset-config', type=str, default='medical_meadow_health_advice', help='Config name for HF datasets with multiple configs')
    parser.add_argument('--use-qwen', action='store_true', help='Use HF Qwen-style model path instead of local HybridModel')
    parser.add_argument('--batch-size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--max-seq-len', type=int, default=None, help='Override max sequence length')
    parser.add_argument('--hidden-size', type=int, default=None, help='Override model hidden size')
    parser.add_argument('--num-layers', type=int, default=None, help='Override number of layers')
    parser.add_argument('--dream-every', type=int, default=200, help='Run dream phase every N steps')
    parser.add_argument('--dream-steps', type=int, default=4, help='Number of mini-steps during dream')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps to reduce memory')
    parser.add_argument('--num-documents', type=int, default=200, help='Number of documents to sample/tokenize')
    # default to using Qwen model unless explicitly disabled
    parser.set_defaults(use_qwen=True)
    args = parser.parse_args()

    # Config
    config = HybridConfig()
    config.num_documents = args.num_documents

    # apply runtime overrides to keep quick experiments small
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_seq_len:
        config.max_seq_len = args.max_seq_len
    if args.hidden_size:
        config.hidden_size = args.hidden_size
        config.intermediate_size = config.expand_factor * config.hidden_size
    if args.num_layers:
        config.num_layers = args.num_layers

    # Load data (streaming)
    print("Loading data...")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split='train', streaming=True)
    else:
        dataset = load_dataset(args.dataset, split='train', streaming=True)

    # tokenizer: will be set differently depending on model selection
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    # Tokenize into token stream
    all_tokens = []
    for i, item in enumerate(tqdm(dataset, total=config.num_documents, desc="Tokenizing")):
        if i >= config.num_documents:
            break
        # try some common text fields
        text = None
        for key in ("text", "article", "context", "passage", "question", "snippet"):
            if key in item:
                text = item[key]
                break
        if text is None:
            text = str(item)
        tokens = tokenizer.encode(text[:3000], add_special_tokens=False)
        all_tokens.extend(tokens)

    config.vocab_size = getattr(tokenizer, 'vocab_size', config.vocab_size)
    
    # Create dataset
    train_dataset = TextDataset(all_tokens, config.max_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,  # More workers for faster data loading
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2
    )
    
    # Create model - either HF model (qwen) or local HybridModel
    if args.use_qwen:
        print(f"Loading HF model {args.model_name}...")
        hf_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = HFModelWrapper(hf_model)
    else:
        model = HybridModel(config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters, {config.num_layers} layers ({config.layer_pattern})")
    
    # Optimizer and AMP
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Skip GradScaler for models with BFloat16 (e.g., Qwen)
    # Just use autocast for mixed precision
    scaler = None

    # Dreamer: collect past batches and occasionally 'dream'
    dreamer = Dreamer(max_mem=500, k=4)
    
    # Training loop with gradient accumulation and amp.autocast
    model.train()
    step = 0  # counts optimizer steps (updates)
    accum = max(1, args.grad_accum)
    pbar = tqdm(total=config.num_steps, desc="Training")

    micro_step = 0
    for epoch_batch in iter(lambda: True, False):
        # iterate over dataloader until we reach num_steps
        for batch in train_loader:
            if step >= config.num_steps:
                break

            batch = batch.to(device, non_blocking=True)  # Async transfer

            # Remember for dream
            try:
                dreamer.remember(batch.cpu())
            except Exception:
                pass

            # Mixed precision training using recommended API
            with amp.autocast(device_type='cuda'):
                _, loss = model(batch, labels=batch)

            # Backward with accumulation
            loss = loss / accum
            loss.backward()
            micro_step += 1

            # optimizer step when enough micro-batches accumulated
            if (micro_step % accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1

                # Dream phase
                if step % args.dream_every == 0:
                    try:
                        dloss = dreamer.dream(model, tokenizer, optimizer, device, steps=args.dream_steps)
                        if dloss is not None:
                            pbar.set_postfix({'loss': f'{loss.item() * accum:.4f}', 'dream_loss': f'{dloss:.4f}'})
                    except Exception:
                        pass

                if step % config.log_every == 0:
                    pbar.set_postfix({'loss': f'{(loss.item() * accum):.4f}'})
                pbar.update(1)
        if step >= config.num_steps:
            break
    
    pbar.close()
    
    # Build and save dream graph (visualization)
    try:
        def _embed_fn(x: torch.Tensor):
            emb_layer = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else model.embed
            return emb_layer(x)

        print("Building dream graph (this may take a moment)...")
        dreamer.build_graph(_embed_fn, device=device)
        dreamer.visualize(out_html='dream_graph.html')
        print("Dream graph exported to dream_graph.html")
    except Exception:
        pass

    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # For Qwen, save the inner model's state dict; for Hybrid, save the model itself
    if args.use_qwen:
        # HFModelWrapper wraps the actual HF model
        state_dict_to_save = model_to_save.model.state_dict()
    else:
        state_dict_to_save = model_to_save.state_dict()
    
    # Save as checkpoint with metadata
    checkpoint = {
        'model_type': 'qwen' if args.use_qwen else 'hybrid',
        'model_name': args.model_name if args.use_qwen else None,
        'state_dict': state_dict_to_save,
        'config': config if not args.use_qwen else None,
    }
    torch.save(checkpoint, "model.pt")
    print("Model saved to model.pt")
    
    # Quick generation test
    model.eval()
    with torch.no_grad():
        prompt = tokenizer.encode("The future of AI is", return_tensors="pt").to(device)
        with amp.autocast(device_type='cuda'):
            for _ in range(30):
                logits, _ = model(prompt)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token], dim=1)
        
        print("\nGenerated:", tokenizer.decode(prompt[0]))


if __name__ == "__main__":
    main()
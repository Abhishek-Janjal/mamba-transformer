"""
Usage:
  # Install requirements
  pip install -r requirements.txt

  # After training (produces model.pt), and after having a conversation saved as 'conversation.txt':
  python build_dream_graph_from_history.py --model_path model.pt --conversation conversation.txt --out dream_graph_from_chat.html

This script:
 - loads tokenizer and trained model (HybridModel)
 - tokenizes each non-empty line in the conversation file as a memory sequence
 - uses `Dreamer` to remember sequences, builds k-NN graph, and exports an interactive HTML graph
"""

import argparse
import torch
from transformers import AutoTokenizer
from hybrid_llm import HybridConfig, HybridModel
from dreamer import Dreamer


def load_tokenizer(model_name_or_path: str = 'gpt2'):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    tok.pad_token = tok.eos_token or tok.pad_token
    return tok


def load_trained_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location='cpu')

    # New checkpoint format: dict with metadata
    if isinstance(checkpoint, dict) and 'model_type' in checkpoint:
        model_type = checkpoint.get('model_type')
        state_dict = checkpoint.get('state_dict')

        if model_type == 'qwen':
            model_name = checkpoint.get('model_name', 'Qwen/Qwen3-0.6B')
            print(f"Loading HF model {model_name}...")
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            try:
                hf_model.load_state_dict(state_dict)
            except RuntimeError:
                # try remapping common wrapper prefixes
                remapped = {}
                for k, v in state_dict.items():
                    new_k = k
                    if new_k.startswith('model.'):
                        new_k = new_k[len('model.'):]
                    if new_k.startswith('model.'):
                        new_k = new_k[len('model.'):]
                    remapped[new_k] = v
                hf_model.load_state_dict(remapped, strict=False)
            model = hf_model.to(device)
            model.eval()
            return model

        else:
            # hybrid
            config = checkpoint.get('config') or HybridConfig()
            if config is None:
                config = HybridConfig()
            model = HybridModel(config)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            return model

    # Legacy: assume plain state_dict for HybridModel
    if 'embed.weight' in checkpoint:
        vocab_size = checkpoint['embed.weight'].shape[0]
        config = HybridConfig()
        config.vocab_size = vocab_size
        model = HybridModel(config)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        return model

    raise RuntimeError('Unrecognized checkpoint format for ' + model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--conversation', type=str, required=True)
    parser.add_argument('--out', type=str, default='dream_graph_from_chat.html')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--max_mem', type=int, default=2000)
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device)
    print('Using device:', device)

    print('Loading tokenizer...')
    tokenizer = load_tokenizer(args.tokenizer)

    print('Loading trained model...')
    model = load_trained_model(args.model_path, device)

    dreamer = Dreamer(max_mem=args.max_mem, k=args.top_k)

    # Read conversation file and tokenize lines
    with open(args.conversation, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        print('No conversation lines found in', args.conversation)
        return

    # Tokenize each line and store as memory
    for line in lines:
        toks = tokenizer.encode(line, add_special_tokens=False)
        if len(toks) == 0:
            continue
        seq = torch.tensor(toks, dtype=torch.long).unsqueeze(0)  # (1, L)
        dreamer.remember(seq)

    print(f'Remembered {len(dreamer.memories)} sequences from conversation')

    # Build graph and export
    def embed_fn(x: torch.Tensor):
        emb_layer = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else model.embed
        return emb_layer(x)

    print('Building graph...')
    dreamer.build_graph(embed_fn, device=device)
    print('Visualizing to', args.out)
    dreamer.visualize(out_html=args.out)
    print('Done')


if __name__ == '__main__':
    main()

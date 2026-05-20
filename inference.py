import sys
import time
import torch
import torch.nn.functional as F
import colorama
colorama.init()
from hybrid_llm import HybridModel, HybridConfig, HFModelWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def load_model(model_path, device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint (may contain metadata)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is the new format with metadata
    if isinstance(checkpoint, dict) and 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
        state_dict = checkpoint['state_dict']
        
        if model_type == 'qwen':
            # Load HF Qwen model
            model_name = checkpoint.get('model_name', 'Qwen/Qwen3-0.6B')
            print(f"Loading Qwen model: {model_name}...")
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            model = HFModelWrapper(hf_model)
            
            # Unwrap to load state dict directly into inner model
            try:
                model.model.load_state_dict(state_dict)
            except RuntimeError as e:
                # Try remapping common wrapper prefixes (e.g., 'model.' or 'model.model.')
                remapped = {}
                for k, v in state_dict.items():
                    new_k = k
                    if new_k.startswith('model.'):
                        new_k = new_k[len('model.'):]
                    if new_k.startswith('model.'):
                        new_k = new_k[len('model.'):]
                    remapped[new_k] = v
                try:
                    model.model.load_state_dict(remapped)
                except Exception:
                    # Re-raise original for clarity
                    raise e
        else:
            # Load HybridModel
            config = checkpoint.get('config')
            if config is None:
                config = HybridConfig()
                vocab_size = state_dict.get('embed.weight', torch.zeros(50257, 128)).shape[0]
                config.vocab_size = vocab_size
            
            model = HybridModel(config)
            model.load_state_dict(state_dict)
    else:
        # Old format: assume it's HybridModel state dict
        print("Loading legacy checkpoint (assumes HybridModel)...")
        vocab_size = checkpoint['embed.weight'].shape[0]
        print(f"Detected vocabulary size: {vocab_size}")
        
        config = HybridConfig()
        config.vocab_size = vocab_size
        
        model = HybridModel(config)
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, device='cuda'):
    """Generate text using the loaded model"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Generating...")
    
    with torch.no_grad():
        for i in range(max_length):
            # Get model predictions
            logits, _ = model(input_ids)

            # Get next token logits
            if temperature <= 0:
                temperature = 1e-6
            next_token_logits = logits[:, -1, :] / temperature

            # Sanitize logits: replace NaN/Inf and clamp to reasonable range
            if not torch.isfinite(next_token_logits).all():
                next_token_logits = torch.nan_to_num(next_token_logits, neginf=-1e9, posinf=1e9, nan=-1e9)
            next_token_logits = torch.clamp(next_token_logits, -1e4, 1e4)
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample from the distribution
            try:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                print(f"Sampling error at step {i}: {e}")
                print("Falling back to CPU sampling for debugging...")
                cpu_logits = next_token_logits.detach().to('cpu')
                cpu_probs = F.softmax(cpu_logits, dim=-1)
                next_token = torch.multinomial(cpu_probs, num_samples=1).to(device)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode and print progress
            if i % 10 == 0:
                current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                print(f"Step {i}: {current_text[-50:]}...")
    
    # Decode final result
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def generate_stream(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, device='cuda'):
    """Stream generation token-by-token (yields each decoded token)."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for i in range(max_length):
            logits, _ = model(input_ids)
            if temperature <= 0:
                temperature = 1e-6
            next_token_logits = logits[:, -1, :] / temperature

            # Sanitize logits before top-k / softmax
            if not torch.isfinite(next_token_logits).all():
                next_token_logits = torch.nan_to_num(next_token_logits, neginf=-1e9, posinf=1e9, nan=-1e9)
            next_token_logits = torch.clamp(next_token_logits, -1e4, 1e4)

            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                filtered = torch.full_like(next_token_logits, float('-inf'))
                filtered.scatter_(1, top_k_indices, top_k_logits)
                next_token_logits = filtered

            try:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                print(f"Sampling error at step {i}: {e}")
                print("Falling back to CPU sampling for debugging...")
                cpu_logits = next_token_logits.detach().to('cpu')
                cpu_probs = F.softmax(cpu_logits, dim=-1)
                next_token = torch.multinomial(cpu_probs, num_samples=1).to(device)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            token_id = int(next_token[0, 0].item())
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_text

    return
# ANSI colours
R    = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
CY   = "\033[36m"
GR   = "\033[32m"
YL   = "\033[33m"
BL   = "\033[34m"
RD   = "\033[31m"

BANNER = f"""
{BOLD}{CY}╔══════════════════════════════════════════════════════════════╗
║      Qwen-3-0.6b  ·  Mamba  ·  Dream                         ║
╚══════════════════════════════════════════════════════════════╝{R}
{DIM}  /help   /clear   /memory   /info   /exit{R}
"""
def interactive_mode(model, tokenizer, device):
    print(BANNER)
    print(f"{GR}Entering interactive mode. Type your messages below.{R}")
    system_prompt = "You are a helpful assistant."
    history = []  # list of (role, text)
    temperature = 0.7
    top_k = 40

    while True:
        try:
            # Print colored prompt separately so some terminals/REPLs render ANSI correctly
            print(f"{BOLD}{GR}You > {R}", end="", flush=True)
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')
            break

        if not user_input:
            continue
        if user_input.lower() in ('/exit', 'quit'):
            break
        if user_input.lower().startswith('/system '):
            system_prompt = user_input[len('/system '):].strip()
            print(f"System prompt set to: {system_prompt}")
            continue
        if user_input.lower().startswith('/temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature set to {temperature}")
            except Exception:
                print("Invalid temperature value")
            continue
        if user_input.lower() == '/clear':
            history = []
            print("Conversation cleared")
            continue

        # Append user message
        history.append(('user', user_input))

        # Build prompt: system + history (very simple concatenation)
        prompt_parts = [f"System: {system_prompt}", ""]
        for role, text in history:
            if role == 'user':
                prompt_parts.append(f"User: {text}")
            else:
                prompt_parts.append(f"Assistant: {text}")
        prompt_parts.append("Assistant:")
        full_prompt = "\n".join(prompt_parts)

        # Stream generation
        print(f"\n{BOLD}{BL}Qwen > {R}", end=" ", flush=True)
        try:
            for token in generate_stream(model, tokenizer, full_prompt, max_length=200, temperature=temperature, top_k=top_k, device=device):
                # Some tokenizers return empty strings for whitespace-only tokens
                sys.stdout.write(token)
                sys.stdout.flush()
                time.sleep(0.01)
            print()  # newline after generation

            # Save assistant reply in history by decoding recent tokens
            # Rebuild full text quickly
            # Note: This is a simple approach — for large models prefer streaming into a buffer
            generated_full = generate_text(model, tokenizer, full_prompt, max_length=200, temperature=temperature, top_k=top_k, device=device)
            assistant_reply = generated_full[len(full_prompt):]
            history.append(('assistant', assistant_reply.strip()))

        except Exception as e:
            print(f"\nError during generation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inference script for Hybrid Transformer-Mamba model")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    else:
        # Single generation
        try:
            generated_text = generate_text(
                model, tokenizer, args.prompt, 
                max_length=args.max_length, 
                temperature=args.temperature, 
                top_k=args.top_k, 
                device=device
            )
            
            print("\n" + "="*50)
            print("FINAL GENERATED TEXT:")
            print("="*50)
            print(generated_text)
            
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()

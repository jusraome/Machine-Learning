import argparse
from pathlib import Path
import torch 

from src.device import (
    DEVICE_CHOICES,
    is_directml,
    print_device_info,
    resolve_device,
)
from src.gan import Generator, SEQUENCE_LENGTH

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "gan_vanilla_last.pt"

def decode_sequence(probabilities, itos, temperature =1.0): 
    """
    Select a character per position and converts the sequence to text
    p<1 most common characters 
    p>1 different kind of combinations
    """
    probabilities = probabilities.clamp_min(1e-8).pow(1.0 / temperature)
    probabilities = probabilities / probabilities.sum(dim= -1, keepdim= True)
    # show a character by the distribution of each position 
    ids = torch.multinomial(probabilities, num_samples=1).squeeze(-1).tolist()

    return "".join(itos[idx] for idx in ids)

@torch.no_grad()
def generate(checkpoint_path, number_of_samples, temperature, device_name="auto"): 
    device, device_desc = resolve_device(device_name)
    print_device_info(device, device_desc)

    checkpoint = torch.load( 
        checkpoint_path, 
        map_location="cpu", 
        weights_only= False
    )

    generator = Generator( 
        vocab_size= checkpoint["vocab_size"], 
        noise_dim= checkpoint["noise_dim"]
    )
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device)
    generator.eval()

    itos = checkpoint["vocab"]["itos"]

    noise = torch.randn(number_of_samples, checkpoint["noise_dim"], device = device)
    probabilities = generator(noise) # generation of random noise values
    if is_directml(device):
        probabilities = probabilities.cpu()

    print(
        f"\nChekpoint: {checkpoint_path}"
        f"\nSaved epoch: {checkpoint['epoch']}", 
        f"\nTemperature: {temperature}\n"
        )

    for idx , sequence_probs in enumerate(probabilities, start = 1): 
        text = decode_sequence(sequence_probs,itos, temperature)
        print(f"\nSample {idx}: \n{text}")

def main(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-samples", type = int, default=5)
    parser.add_argument("--temperature", type=float, default= 1.0)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"THERE IS NO CHECKPOINTS ON THE ROUTE {args.checkpoint}")

    if args.num_samples <1: 
        raise ValueError("number of samples muts be positive")
    if args.temperature <= 0:
        raise ValueError("temperatures must be positive")
    generate(args.checkpoint, args.num_samples, args.temperature, args.device)


if __name__ == "__main__": 
    main() 


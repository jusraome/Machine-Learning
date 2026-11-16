import argparse
from pathlib import Path
from collections import Counter

import pandas as pd 
import torch

from src.gan import Generator, SEQUENCE_LENGTH, loadTrainingData

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT/"checkpoints"
REPORT_DIR  = ROOT/ "reports" / "metrics"
SAMPLE_DIR = ROOT/ "reports" / "generated_samples"
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def sample_sequences(generator,count, noise_dim, device, batch_size =128):
    #Generate all the ids by sampling per position 
    all_ids = [] 
    generator.eval()

    with torch.no_grad(): 
        remaining = count

        while remaining > 0: 
            current_batch = min(batch_size, remaining)
            noise = torch.randn(current_batch, noise_dim, device=device)
            probabilities = generator(noise)

            ids = torch.multinomial(
                probabilities.reshape(-1, probabilities.size(-1)), 
                num_samples=1
            )
            ids = ids.reshape(current_batch,SEQUENCE_LENGTH)
            all_ids.extend(ids.cpu().tolist())
            remaining -= current_batch

    return all_ids

def distribution(counter :torch.tensor):
    """ normalzied counter to compare distributions""" 
    total = sum(counter.values())
    if total == 0: 
        return {}
    return {key: value/total for key, value in counter.items()}

def total_variation_distance(counter_a, counter_b): 
    """Comparation between two distributions, closest to 0 --> higher similarity"""
    dist_a = distribution(counter_a)
    dist_b = distribution(counter_b)
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(key, 0) - dist_b.get(key, 0)) for key in keys)

def get_metrics(sequences, real_sequences, unk_token = "<UNK>"): 
    # metrics thata show correlation with real sequences
    unique_sequences = len({tuple(sequence) for sequence in sequences})
    total_characters = sum(len(sequence) for sequence in sequences)

    generated_chars = Counter(char for sequence in sequences for char in sequence)
    real_chars = Counter(char for sequence in real_sequences for char in sequence)

    generated_bigrams = Counter(
        (sequence[i], sequence[i+1])
        for sequence in sequences
        for i in range(len(sequence)-1)
    )

    real_bigrams = Counter(
        (sequence[i], sequence[i+1])
        for sequence in real_sequences
        for i in range(len(sequence)-1)
    )
    unknown_count = generated_chars.get(unk_token,0)

    return {
        "unique_sequence_ratio": unique_sequences/len(sequences), 
        "mean_unique_characters": sum(
            len(set(sequence))for sequence in sequences
        )/ len(sequences), 
        "character_distribution_tv" : total_variation_distance(generated_chars, real_chars),
        "bigram_distribution_tv": total_variation_distance(generated_bigrams,real_bigrams),
        "unknown_charater_ratio": (unknown_count / total_characters if total_characters else 0)
    }
def load_generator(checkpoint, device): 
    checkpoint =torch.load(
        checkpoint, 
        map_location=device, 
        weights_only= False
    )
    generator = Generator( 
        vocab_size=checkpoint["vocab_size"],
        noise_dim=checkpoint["noise_dim"], 
    ).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    return generator, checkpoint

def ids_to_text(sequence_ids,itos): 
    return "".join(
        "\ufffd" if itos[index] == "<UNK>" else itos[index] #uffd unrecognized chars
        for index in sequence_ids
    )

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int ,default=1000)
    parser.add_argument("--seed",type= int, default=SEED)
    args = parser.parse_args()

    if args.num_samples<1: 
        raise ValueError("number of samples must be positive")

    torch.manual_seed(seed=args.seed)
    device = DEVICE

    dataset, vocab = loadTrainingData()
    real_count = min(args.num_samples,len(dataset))
    real_idx = torch.randperm(len(dataset))[:real_count].tolist()
    real_sequences = [dataset[index].tolist() for index in real_idx]

    models = {
        "vanilla": CHECKPOINT_DIR / "gan_vanilla_last.pt",
        "wgan_gp": CHECKPOINT_DIR/ "gan_wgan_last.pt"
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name, checkpoint_path in models.items():
        if not checkpoint_path.exists(): 
            raise FileNotFoundError(f"{model_name} doesn't exists")

        generator, checkpoint = load_generator(checkpoint_path, device)
        generated_ids = sample_sequences(
            generator=generator,
            count = args.num_samples, 
            noise_dim= checkpoint["noise_dim"], 
            device = device
        ) 
        generated_tokens = [
            [vocab["itos"][index] for index in sequence]
            for sequence in generated_ids
        ]
        real_tokens = [
            [vocab["itos"][index] for index in sequence]
            for sequence in real_sequences
        ]

        metrics = get_metrics(generated_tokens, real_tokens)
        rows.append({"model": model_name, **metrics})

        sample_path = SAMPLE_DIR / f"{model_name}_samples.txt"
        with sample_path.open("w", encoding="utf-8") as file:
            for index, sequence in enumerate(generated_ids[:32], start=1):
                file.write(f"sample {index}:\n")
                file.write(ids_to_text(sequence, vocab["itos"]))
                file.write("\n\n")

        print(f"\n{model_name} — checkpoint epoch {checkpoint['epoch']}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print(f" samples saved on: {sample_path}")

    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / "gan_quality.csv", index=False)

    print(f"\n resume saved on {REPORT_DIR/ 'gan.quality.csv'}")
    print(results.round(4).to_string(index=False))

if __name__ =="__main__": 
    main()

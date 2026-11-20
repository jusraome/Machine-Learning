"""GAN training fro synthetic data generation"""
import argparse
from pathlib import Path

import torch 
from torch import nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F 

from src.device import (
    DEVICE_CHOICES,
    cpu_state_dict,
    dataloader_kwargs,
    print_device_info,
    resolve_device,
    seed_everything,
)
from src.gan import Discriminator, Generator, loadTrainingData

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
SEED = 42

def train(args): 

    seed_everything(args.seed)
    device, device_desc = resolve_device(args.device)
    print_device_info(device, device_desc)
    dataset, vocab = loadTrainingData()

    loader = DataLoader(
        dataset=dataset, 
        batch_size= args.batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=True,
        **dataloader_kwargs(device),
    )

    vocab_size = vocab["size"]
    generator = Generator(
        vocab_size=vocab_size, 
        noise_dim=args.noise_dim,
    ).to(device)
    discriminator = Discriminator(
        vocab_size=vocab_size
    ).to(device)

    optimizer_ge= torch.optim.Adam(
        params=generator.parameters(),
        lr = args.learning_rate, 
        betas=(0.5,0.999)
    )
    optimizer_di = torch.optim.Adam(
        params=discriminator.parameters(), 
        lr=args.learning_rate, 
        betas=(0.5,0.999)
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1,args.epochs+1): 
        total_g_loss = 0.0
        total_d_loss = 0.0 
        batches_seen = 0 

        generator.train()
        discriminator.train()

        for batch_idx,ids in enumerate(loader,start=1): 
            if args.max_batches and batch_idx > args.max_batches: 
                break

            ids = ids.to(device, non_blocking=(device.type == "cuda"))

            real = F.one_hot(ids,num_classes=vocab_size).float()
            batch_size = real.size(0)

            #! Discriminator update 

            noise = torch.randn(batch_size,args.noise_dim, device=device) # tensor size
            fake= generator(noise)

            real_logits = discriminator(real)
            fake_logits = discriminator(fake.detach()) # detach to disconect from the main graphs

            # Unilatreal softened 
            real_labels = torch.full_like(real_logits, 0.9)
            fake_labels = torch.zeros_like(fake_logits)

            d_loss_real = criterion(real_logits,real_labels)
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_di.zero_grad()
            d_loss.backward()
            optimizer_di.step()

            #! Generator update 

            noise = torch.randn(batch_size, args.noise_dim, device = device) 
            generated = generator(noise)
            generated_logits = discriminator(generated)

            #real target 
            target = torch.ones_like(generated_logits)
            g_loss = criterion(generated_logits, target)

            optimizer_ge.zero_grad()
            g_loss.backward()
            optimizer_ge.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            batches_seen +=1 


        mean_g_loss = total_g_loss / batches_seen
        mean_d_loss = total_d_loss / batches_seen

        print(
            f"Epoch {epoch:03d}/{args.epochs} |"
            f"Generator loss {mean_g_loss:.4f} | "
            f"Discriminator loss {mean_d_loss:.4f} | "
            f"lotes {batches_seen}"
        )

        CHECKPOINT_DIR.mkdir(parents= True,exist_ok=True)
        torch.save( 
            {
                "epoch": epoch, 
                "generator_state_dict" : cpu_state_dict(generator), 
                "discriminator_state_dict" : cpu_state_dict(discriminator), 
                "vocab": vocab, 
                "noise_dim": args.noise_dim, 
                "vocab_size": vocab_size 
            }, 
            CHECKPOINT_DIR / "gan_vanilla_last.pt"
        )

def main (): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default= 2e-4)
    parser.add_argument("--seed", type=int, default= 42)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)

    # just for testing , the 0 value = non limit
    parser.add_argument("--max-batches", type= int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__": 
    main()



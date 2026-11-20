"Re-train of the GAN using gradient penalty"
import argparse
from pathlib import Path 
import torch 

from torch.utils.data import DataLoader

from src.device import (
    DEVICE_CHOICES,
    cpu_state_dict,
    dataloader_kwargs,
    print_device_info,
    resolve_device,
    seed_everything,
)
from src.gan import Critic, Generator, loadTrainingData

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT /"checkpoints"
SEED = 42 

#the critic applies the penalty 

def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor, device: torch.device): 

    batch_size = real.size(0)
    alpha = torch.rand(batch_size,1,1,device=device)
    interpolated = alpha * real+(1-alpha) * fake.detach()
    interpolated.requires_grad_(True)

    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores, 
        inputs=interpolated, 
        grad_outputs=torch.ones_like(scores), 
        create_graph=True,
        retain_graph=True 
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = torch.sqrt(gradients.pow(2).sum(dim=1) + 1e-12)
    return ((gradient_norm - 1) ** 2).mean()

def train(args): 
    seed_everything(args.seed)
    device, device_desc = resolve_device(args.device)
    print_device_info(device, device_desc)

    dataset, vocab = loadTrainingData()

    loader = DataLoader( 
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        drop_last= True,
        **dataloader_kwargs(device),
    )

    vocab_size = vocab["size"]
    generator = Generator(
        vocab_size=vocab_size,
        noise_dim=args.noise_dim
        ).to(device)
    critic = Critic(vocab_size).to(device)

    optimizer_g = torch.optim.Adam( 
        generator.parameters(), 
        lr= args.learning_rate, 
        betas=(0.0,0.9)
    )
    optimizer_c = torch.optim.Adam( 
        critic.parameters(), 
        lr = args.learning_rate, 
        betas= (0.0,0.9)
    )

    for epoch in range(1, args.epochs + 1):
        total_g_loss = 0.0
        total_c_loss = 0.0
        batches_seen = 0
        g_steps = 0

        generator.train()
        critic.train()

        for batch_idx, real_ids in enumerate(loader, start=1):
            if args.max_batches and batch_idx > args.max_batches:
                break

            real_ids = real_ids.to(device, non_blocking=(device.type == "cuda"))
            real = torch.nn.functional.one_hot(
                real_ids,
                num_classes=vocab_size
            ).float()

            batch_size = real.size(0)

            #! critic update
            noise = torch.randn(batch_size, args.noise_dim, device=device)
            with torch.no_grad():
                fake = generator(noise)

            real_scores = critic(real)
            fake_scores = critic(fake)

            gp = gradient_penalty(
                critic=critic,
                real=real,
                fake=fake,
                device=device
            )

            critic_loss = (
                fake_scores.mean() - real_scores.mean()
                + args.lambda_gp * gp
            )

            optimizer_c.zero_grad(set_to_none=True)
            critic_loss.backward()
            optimizer_c.step()

            total_c_loss += critic_loss.item()
            batches_seen += 1

            #! generator update
            if batch_idx % args.critic_steps == 0:
                noise = torch.randn(batch_size, args.noise_dim, device=device)
                generated = generator(noise)
                generator_loss = -critic(generated).mean()

                optimizer_g.zero_grad(set_to_none=True)
                generator_loss.backward()
                optimizer_g.step()

                total_g_loss += generator_loss.item()
                g_steps += 1

        if batches_seen == 0:
            raise ValueError("check the max-batches input or the dataset structure")

        mean_c_loss = total_c_loss / batches_seen
        mean_g_loss = total_g_loss / g_steps if g_steps > 0 else float("nan")

        print(
            f"epoch: {epoch:03d}/{args.epochs} | "
            f"critic loss: {mean_c_loss:.4f} | "
            f"generator loss: {mean_g_loss:.4f} | "
            f"batches: {batches_seen} | g_steps: {g_steps}"
        )

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": cpu_state_dict(generator),
                "critic_state_dict": cpu_state_dict(critic),
                "vocab": vocab,
                "noise_dim": args.noise_dim,
                "vocab_size": vocab_size
            },
            CHECKPOINT_DIR / "gan_wgan_last.pt"
        )

def main (): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim",type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-steps", type= int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-batches", type= int, default=0)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    args = parser.parse_args()

    train(args=args)


if __name__ == "__main__":
    main()
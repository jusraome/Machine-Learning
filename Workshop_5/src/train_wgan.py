"Re-train of the GAN using gradient penalty"
import argparse
from pathlib import Path 
import torch 

from torch.utils.data import DataLoader

from src.gan import Critic, Generator, loadTrainingData

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT /"checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42 

#the critic applies the penalty 

def gradient_penalty(critic, real:torch.tensor,fake,device): 

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
    gradient_norm = gradients.norm(2,dim = 1)
    return ((gradient_norm - 1) ** 2).mean()

def train(args): 
    torch.manual_seed(args.seed)
    device = DEVICE

    dataset, vocab = loadTrainingData()

    loader = DataLoader( 
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        drop_last= True
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

    for epoch in range (1, args.epochs +1): 
        total_g_loss = 0.0
        total_c_loss = 0.0 
        batches_seen = 0

        generator.train()
        critic.train()

        for batch_idx, real_ids in enumerate(loader, start=1): 
            if args.max_batches and batch_idx > args.max_batches: 
                break

            real_ids = real_ids.to(device)
            real = torch.nn.functional.one_hot(
                real_ids, 
                num_classes=vocab_size
            ).float()

            batch_size = real.size(0)

            #! critic update
            critic_loss_sum = 0.0 

            for _ in range(args.critic_steps): 
                noise = torch.randn(batch_size, args.noise_dim, device=device)
                fake = generator(noise)

                real_scores = critic(real)
                fake_scores = critic(fake.detach())

                gp = gradient_penalty(
                    critic=critic,
                    real=real,
                    fake=fake, 
                    device = device)

                critic_loss = (fake_scores.mean()-real_scores.mean()
                               +args.lambda_gp * gp)

                optimizer_c.zero_grad(set_to_none=True)
                critic_loss.backward()
                optimizer_c.step()

                critic_loss_sum += critic_loss.item()

            #! generator update 

            noise = torch.randn(batch_size,args.noise_dim, device=device)
            generated = generator(noise)
            generator_loss = -critic(generated).mean()

            optimizer_g.zero_grad(set_to_none=True)
            generator_loss.backward()
            optimizer_g.step()

            total_c_loss += critic_loss_sum / args.critic_steps
            total_g_loss += generator_loss.item()
            batches_seen += 1

            if batches_seen ==0: 
                raise ValueError("chek the max-batches input or the dataset strcture")

            print(
                f"epoch: {epoch:03d}/ {args.epochs} | "
                f"critic loss: {total_c_loss/ batches_seen:.4f} | "
                f"generator loss: {total_g_loss / batches_seen:.4f} | "
                f"batch: {batches_seen}" 
            )

            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch, 
                    "generator_state_dict": generator.state_dict(), 
                    "critic_state_dict": critic.state_dict(), 
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
    args = parser.parse_args()

    train(args=args)


if __name__ == "__main__":
    main()
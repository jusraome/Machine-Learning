from pathlib import Path
import torch
from model import CharTransformer
from tokenizer import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "model_best.pt"
CONTEXT_LENGTH = 80


def main(): 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    model = CharTransformer(vocab_size=tokenizer.vocab_size()).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = input("\nWrite an exaple to predict the next character: ")

    if not prompt: 
        return

    input_ids = tokenizer.encode(prompt)[-CONTEXT_LENGTH:]
    X = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits[0], dim = 0) 
        #top 5 predicted characters 
        top_prob, top_ids = torch.topk(probabilities,k =5) 

        predicted_id = int((top_ids[0].item()))
        predicted_char = tokenizer.idxToChar.get(predicted_id, "<UNK>")

        print(f"\n predicted character {predicted_char!r} | results {prompt+predicted_char}")
        print("\n Top 5 predicted characters: ")

        for char_id, prob in zip(top_ids.tolist(), top_prob.tolist()):
            char = tokenizer.idxToChar.get(char_id,"<UNK>")
            print(f"{char!r}: {prob:.4f}")

if __name__ == "__main__": 
    main()
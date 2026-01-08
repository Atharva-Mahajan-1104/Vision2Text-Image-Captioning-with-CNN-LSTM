import argparse, json, torch
from PIL import Image
from torchvision import transforms
from models import EncoderCNN, DecoderLSTM
from utils import Vocabulary
"""
    Load a saved vocabulary from a JSON file.

    This function reconstructs the Vocabulary object by
    restoring the word-to-index and index-to-word mappings
    used during training.
    """
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    v = Vocabulary(min_freq=obj.get("min_freq", 3))
    v.word2id = obj["word2id"]
    v.id2word = {int(i): w for w, i in v.word2id.items()}
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-len", type=int, default=20)
    args = ap.parse_args()

   # Select GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model checkpoint onto the selected device
ckpt = torch.load(args.checkpoint, map_location=device)

# Load the saved vocabulary used during training
vocab = load_vocab(args.vocab)


    enc = EncoderCNN(embed_dim=ckpt["embed_dim"]).to(device)
    dec = DecoderLSTM(vocab_size=ckpt["vocab_size"], embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"]).to(device)

    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = enc(img)
        ids = dec.sample(feats, max_len=args.max_len, bos_id=vocab.word2id["<bos>"], eos_id=vocab.word2id["<eos>"])
    caption = vocab.decode(ids[0].cpu().numpy())
    print(caption)

if __name__ == "__main__":
    main()

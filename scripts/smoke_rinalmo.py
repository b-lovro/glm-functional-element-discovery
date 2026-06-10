import argparse
import torch
from rinalmo.pretrained import get_pretrained_model

from rinalmo.model.model import RiNALMo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--model-name", type=str, default="giga-v1")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("RiNALMo smoke test")
    print("=" * 80)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("device:", device)
    print("model:", args.model_name)

    seq = "AUGGCUACGUAGCUAGCUAGCUAGCUAGCUA"

    print("sequence length:", len(seq))
    print("sequence preview:", seq[:100])

    print("\nLoading model...")
    model, alphabet = get_pretrained_model(model_name=args.model_name)

    model = model.to(device)
    model.eval()

    print("Tokenizing...")
    tokens = torch.tensor(
        alphabet.batch_tokenize([seq]),
        dtype=torch.int64,
        device=device,
    )

    print("tokens shape:", tuple(tokens.shape))

    print("Running forward pass...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            outputs = model(tokens)

    print("output keys:", list(outputs.keys()))

    representations = outputs["representation"]
    pooled = representations.mean(dim=1)

    print("representation shape:", tuple(representations.shape))
    print("pooled embedding shape:", tuple(pooled.shape))
    print("representation dtype:", representations.dtype)


    print("\nDone.")


if __name__ == "__main__":
    main()
"""Write a trained model's robot motions from a saved checkpoint.

Training already writes the motions it produced, and saves the trained model in
`<data_dir>/checkpoints/`. This step writes those motions again from a saved model,
without training: it rebuilds the network the checkpoint describes, loads its weights and
the two scales saved with them, and writes one motion file per human clip, exactly as
training did. No trained models are distributed with this repository; the checkpoint is
one you trained yourself with train.py or train_adversarial.py.

Runs in the main environment, on a GPU when one is available:
    python -m robots.human_to_g1.generate \
        --checkpoint data/robots/human_to_g1/checkpoints/true_pair_model.pt \
        --out_dir data/robots/human_to_g1/generated_again/true_pair_model
"""
import argparse
from pathlib import Path

import torch

from robots.human_to_g1.data import DEFAULT_DATA, load_corpus
from robots.human_to_g1.train import HUMAN_JOINTS, ROBOT_POINTS, write_motions
from robots.models import AutoencoderPair, DirectMap


def load_model(checkpoint_path, device):
    """Rebuild a trained model and return it with its two normalisation scales."""
    saved = torch.load(checkpoint_path, map_location=device)
    name = saved["model"]
    model = (AutoencoderPair(HUMAN_JOINTS, ROBOT_POINTS) if name == "unpaired_objective"
             else DirectMap(HUMAN_JOINTS, ROBOT_POINTS))
    model.load_state_dict(saved["state_dict"])
    return model.to(device), name, saved["human_scale"], saved["robot_scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name, human_scale, robot_scale = load_model(args.checkpoint, device)
    clips, _ = load_corpus(Path(args.data_dir))
    write_motions(model, name, clips, human_scale, robot_scale, device, args.out_dir)
    print(f"{name}: {len(clips)} motions written to {args.out_dir}")


if __name__ == "__main__":
    main()

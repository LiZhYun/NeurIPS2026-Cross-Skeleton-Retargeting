"""Turn the raw Truebones Zoo BVH files into the processed dataset every method reads.

Reads datasets/truebones/zoo/Truebone_Z-OO and writes dataset/truebones/zoo/truebones_processed
(motions/, bvhs/, animations/, cond.npy, metadata.txt, positions_error_rate.txt). Both paths are
relative to the repository root, so run this from there:

    python -m core.truebones.create_dataset
"""
from core.truebones.motion_process import create_data_samples

def main():
    create_data_samples()

if __name__=="__main__":
    main()

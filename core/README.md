# Core

The code in this folder is shared by every method and by the Truebones evaluation. It comes
from [AnyTop](https://github.com/Anytop2025/Anytop) by Gat et al., a diffusion model that
generates motion for skeletons of any shape.

`anytop/` is AnyTop's own model: the network, the diffusion process that trains and samples
it, the loader that feeds it Truebones clips, and the training and sampling loops. The
source-conditioned AnyTop in `methods/anytop_source/` builds on it, and a few other parts of
the repository reuse pieces of it. It is AnyTop's code under its MIT licence, kept in
AnyTop's folder layout so each file can be compared with the original. The licence text is
in `anytop/LICENSE`. The file `anytop/utils/rotation_conversions.py` comes from PyTorch3D
and keeps its BSD licence, in `anytop/utils/PYTORCH3D_LICENSE`. We made a few changes to
AnyTop's files, mainly so that they run as part of this repository; each one is listed in
[THIRD_PARTY.md](../THIRD_PARTY.md).

`truebones/` turns the Truebones Zoo into the form the code reads, and describes its
skeletons. It is based on AnyTop's own Truebones code.

- `create_dataset.py` converts the downloaded zoo into the processed dataset
  (`python -m core.truebones.create_dataset`, see [docs/data.md](../docs/data.md)).
- `use_paper_clip_names.py` then gives the processed clips the names the paper uses, with
  the list in `paper_clip_names.json` (`python -m core.truebones.use_paper_clip_names`).
- `process_new_skeleton.py` converts the motions of a single skeleton, from the zoo or from
  elsewhere, into a folder of your choice.
- `motion_process.py` does the conversion itself, and also turns processed values back into
  joint positions.
- `param_utils.py` holds the settings that describe the zoo: where the data lives, which
  joints show which way each animal faces, and the ten animals held out of training so that
  a model first meets them at evaluation.
- `get_opt.py` gathers these settings for AnyTop's data loader, and `plot_script.py` draws
  the short video of each clip that the conversion writes.

Everything here runs as `python -m core.<...>` from the repository root.

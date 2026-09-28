# Methods

The fourteen methods the paper evaluates on the Truebones Zoo, one folder per family. Each
folder holds the code to train the method (if it is trained), the settings the paper used, and
a script that answers the queries of an evaluation set. How to run each one, and how long it
takes, is in [docs/methods.md](../docs/methods.md).

| Folder | Methods | Trained |
|---|---|---|
| `tokenizer/` | The per-animal tokenizers shared by ACE, MoReFlow, AL-Flow and DPG-SB-v3 | yes |
| `anytop_source/` | AnyTop, with a motion encoder so it generates from the source clip | yes |
| `ace/` | ACE-T and ACE-I, and the runs that take the source away | yes |
| `moreflow/` | MoReFlow-T and MoReFlow-I, and the step that encodes the training clips | yes |
| `alflow/` | AL-Flow, AL-Flow-Src and AL-Flow-Src-G | yes |
| `dpg_sb/` | DPG-SB-v3 | yes |
| `motion2motion/` | Motion2Motion-Direct and Motion2Motion-BVH | no |
| `anchor/` | ANCHOR | no |
| `random_label/` | random-same-cluster and random-same-exact-action | no |
| `common/` | Code shared by several methods | |

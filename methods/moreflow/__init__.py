"""MoReFlow, one of the retargeting methods the paper compares.

Like ACE, it works on the short sequences of numbers a frozen per-skeleton tokenizer
produces from motion, rather than on motion directly. Where ACE predicts the target
directly, MoReFlow learns a path: it starts from the source animal's numbers and moves them
step by step until they read as the target animal's. What it learns is the direction to
move at each point along that path.

During training the two sides are matched up by a small set of measurements taken from each
clip, such as how fast the body travels or where the feet are. Those same measurements can
then be asked for at generation time, which is how a particular motion is requested.

The paper reports two trained models: MoReFlow-T, trained on all 70 skeletons, and
MoReFlow-I, trained on 60 so the other 10 are new to it when it is measured.

This package also holds build_cache, which prepares the tokenized clips that ACE, AL-Flow
and DPG-SB train on as well.
"""

"""DPG-SB-v3, one of the retargeting methods the paper compares.

It starts generation from a real clip rather than from nothing. Given a query, it finds a
clip of the same action already performed by the target animal, adds noise to it, and then
learns to move that starting point towards what the source motion asks for. The clips the
query will be scored against are excluded from that search, so the answer cannot simply be
copied.

Everything happens in the tokenized space the per-skeleton tokenizers provide, and the
result is turned back into motion by the target animal's tokenizer.
"""

# latents

This folder looks at the numeric code each trained model settles on before it turns that code
into a motion. That is how the paper tells a model that never took note of the source clip
apart from one that took note of it and then lost it, and how it checks that two training runs
of one model differ by a turn of their latent space. The full guide, including which trained
models each check needs, is [docs/latents.md](../docs/latents.md).

Save the codes of one trained model, then run the checks on the saved folders:

```bash
python -m latents.dump_latents --method ace \
    --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/latent_codes/ace_t_seed42/set37
python -m latents.rotation_test --run ACE-T-seed42 --codes outputs/latent_codes/ace_t_seed42/set37
```

| File | What it does |
|---|---|
| `dump_latents.py` | Runs one trained model (ACE, MoReFlow, AL-Flow or AnyTop) over a benchmark set and saves its code for each query. |
| `read_codes.py` | Reads a folder of saved codes; the other scripts start here. |
| `effective_rank.py` | How many directions of its code each model actually uses. |
| `cross_seed_alignment.py` | How much of the difference between two training runs one rotation explains. |
| `rotation_test.py` | Whether turning a code changes the decoded motion more than noise of the same size does. |
| `rotation_significance.py` | Rank tests on the results of `rotation_test.py`. |
| `latent_sif.py` | Source-Instance Fidelity computed on the codes, next to the same score on the motions. |

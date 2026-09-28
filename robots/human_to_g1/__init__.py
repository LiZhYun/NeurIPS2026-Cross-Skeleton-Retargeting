"""Human motion capture retargeted to one Unitree G1 humanoid.

Run the steps in this order, each as `python -m robots.human_to_g1.<step>` from the
repository root. docs/robots.md says which environment each step runs in.

    select_clips                   choose the actions and clips, and write clips.json
    extract_clips                  pull those clips out of the two motion archives
    convert_to_tensors             turn them into joint positions for both skeletons
    train                          train the unpaired, averaging and true-pair models
    train_adversarial              train the adversarial model
    train_adversarial_second_setup train it again with different training settings, three
                                   seeds, and score each run
    generate                       write a trained model's motions again from its checkpoint
    score                          Source-Instance Fidelity (SIF) and variation on their own
    score_four_measures            SIF, variation, action-level AUC and realism
    random_clip_auc                the action-level AUC of the random same-action clip,
                                   scored fairly

human_positions and g1_positions read one recording of each side; the conversion step
uses them.
"""

"""One motion capture set retargeted to six different humanoid robots.

The human-to-G1 study asks the paper's question of one robot. This one asks it of six, so
the answer cannot be a quirk of a single machine. The six robots differ in height, in how
many joints they have and in how their limbs are arranged.

Run the steps in this order, each as `python -m robots.lafan1_to_six_robots.<step>` from
the repository root. docs/robots.md says which environment each step runs in.

    choose_windows      check the recordings and choose the windows to use
    retarget            retarget every recording to one robot
    retarget_retry      redo any recording that crashed, one process at a time
    build_tensors       turn the results into joint positions and write the clip list
    train               train the three models for every robot and setting
    score               Source-Instance Fidelity (SIF) and variation for every robot and
                        setting
    score_four_measures action-level AUC and realism for every robot and setting
    random_clip_auc     the action-level AUC of the random same-group clip, scored fairly

retarget_one retargets a single recording; retarget_retry calls it. The retargeting itself
is done by General Motion Retargeting, which is fetched by `robots/fetch_gmr.sh`.
"""

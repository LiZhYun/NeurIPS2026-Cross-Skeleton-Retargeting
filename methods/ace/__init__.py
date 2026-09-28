"""ACE, one of the retargeting methods the paper compares.

Retargeting means taking a motion performed by one animal skeleton and producing the same
motion on another one, which may have a different number of limbs and a different body
plan. ACE does this in two steps. A tokenizer, trained once per skeleton and then frozen,
turns short pieces of that skeleton's motion into a short sequence of numbers, and turns
them back again. ACE works only on those numbers: it reads the source animal's numbers and
writes the target animal's.

It is trained like a generative adversarial network. A second network, the discriminator,
looks at decoded target motion and says whether it looks like that animal really moving,
and the generator is trained partly to convince it. A second term compares a small set of
measurements of the source motion against the same measurements of what was generated, so
the output is pulled towards the source's timing and footfalls.

The paper reports two trained models: ACE-T, trained on all 70 skeletons, and ACE-I,
trained on 60 of them so the other 10 are new to it when it is measured. It also reports
what happens when the source motion is taken away, both during training and at the moment
of generation.
"""

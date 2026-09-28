"""AL-Flow, a family of comparison methods that generate from an action label.

These models test how much of the task is solved by the action label alone. Each Truebones
clip carries an action name, such as walk or attack, and those names are grouped into ten
coarse clusters. An AL-Flow model is told the target animal and the action, and generates a
motion for that animal; whether it is also told anything about the source clip is what the
three variants differ in.

  AL-Flow        the action labels and the target animal only.
  AL-Flow-Src    also the source clip and which animal performed it.
  AL-Flow-Src-G  the same, but the model never looks an animal up by name: both animals
                 reach it only as a description of their skeleton's shape, so it can also
                 be asked about animals it never trained on.

All three generate in the same tokenized space as ACE and MoReFlow.
"""

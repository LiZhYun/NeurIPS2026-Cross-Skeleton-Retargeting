"""Pictures and videos of the robots, and the figure they are laid out into.

    select_clips          measure which clips suit the figure, and record the measurements
    panel_six_robots      one dance shown on all six robots
    panel_two_performers  two performers of one routine, on the Unitree G1
    compose               lay the two halves out as one figure and add the labels
    merge_report          gather the record of what the figure is made of
    run_panels            all of the above in this order
    run_second_panel      only the second half, then the layout and the record

These steps run in the robot environment described by `robots/environment.yml`, read the
robot descriptions from the retargeting tool in external/GMR, and draw on a graphics card:

    python -m robots.render.run_panels
"""

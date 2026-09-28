"""Redraw only the second half of the figure, then lay it out and gather the record.

Useful while adjusting the second half, since the first half takes much longer to draw.
It reuses the first half and the clip measurements from an earlier run_panels run.

Runs in the robot environment on a machine with a graphics card:
    python -m robots.render.run_second_panel
"""
from robots.render.run_panels import LAYOUT, parse, run

STEPS = ["robots.render.panel_two_performers"] + LAYOUT

if __name__ == "__main__":
    run(STEPS, parse())

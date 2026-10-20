"""Root entrypoint for launching the PokéGrade AI Streamlit Card Grading UI."""

from __future__ import annotations

import importlib

import app.tools.vision_inspector
import app.ui

importlib.reload(app.tools.vision_inspector)
importlib.reload(app.ui)

if __name__ == "__main__":
    app.ui.render_pokegrade_ui()


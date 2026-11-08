"""Skills for local models (D-62).

A skill is a folder with a ``SKILL.md`` (front matter with ``name`` and ``description``, then markdown)
and optional extra text files: the same layout other agents use, so those skills import unchanged.
``importer`` validates a folder or ``.zip`` the user picked, ``store`` keeps imported skills under
``~/.chaosengine/skills``, and ``chat`` gives an enabled skill's name and description to the chat model
and its full text only when the model asks. Nothing in a skill is ever executed.
"""

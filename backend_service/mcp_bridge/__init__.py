"""Agent bridge: MCP tools for the Image, Video and Voice APIs.

``protocol`` answers JSON-RPC at ``/mcp``, ``tools`` defines the fixed tool set, ``jobs`` runs the
work on the shared heavy-job queue, ``models`` lists and recommends models, ``access`` holds the
on/off switch and the agent token. Nothing heavy is imported at module load.
"""

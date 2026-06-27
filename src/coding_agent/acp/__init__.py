"""
ACP server: exposes the coding agent over the Agent Client Protocol.

The ACP server communicates with ACP-aware editors
(Zed, JetBrains, Neovim, Emacs, etc.) via the stdio transport.
"""

from coding_agent.acp.server import CodingAgentServer

__all__ = [
    "CodingAgentServer",
]

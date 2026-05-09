"""
Local RDKit tool runner (CLI).

This script provides a fast local interface for the first scientific-agent extension
without requiring Streamlit integration.

Examples:
  python scripts/run_rdkit_agent.py --list-tools
  python scripts/run_rdkit_agent.py --tool canonicalize_smiles --args-json '{"smiles":"CCO"}'
"""

import argparse
import json
import sys
from pathlib import Path

# Allow importing src package when script is run from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rdkit_tools import execute_tool, get_tool_schemas  # noqa: E402


def main() -> None:
    """Parse CLI arguments and execute the requested RDKit tool."""
    parser = argparse.ArgumentParser(description="Run local RDKit tools")
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available RDKit tools and exit",
    )
    parser.add_argument(
        "--tool",
        type=str,
        help="Tool name to execute",
    )
    parser.add_argument(
        "--args-json",
        type=str,
        default="{}",
        help="Tool arguments as JSON object string",
    )

    args = parser.parse_args()

    if args.list_tools:
        tools = get_tool_schemas()
        print(json.dumps(tools, indent=2))
        return

    if not args.tool:
        parser.error("--tool is required unless --list-tools is used")

    try:
        tool_args = json.loads(args.args_json)
    except json.JSONDecodeError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "tool": args.tool,
                    "error": f"Invalid JSON in --args-json: {str(exc)}",
                },
                indent=2,
            )
        )
        sys.exit(1)

    result = execute_tool(args.tool, tool_args)
    print(json.dumps(result, indent=2))

    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()

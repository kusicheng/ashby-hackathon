"""Kevin-C: launch the desktop tool or run reproducible JSON workflows."""

import argparse
import json
import sys
from pathlib import Path

from kusic import workbench


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kevin-C pathology research workbench")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("gui", help="Open the local desktop workspace (default)")
    demo = subparsers.add_parser("demo", help="Evaluate all four synthetic prototypes")
    demo.add_argument("--output", type=Path, help="Save evaluation.json and comparison.csv here")
    template = subparsers.add_parser("template", help="Print editable synthetic input JSON")
    template.add_argument("--idea", choices=workbench.IDEAS, required=True)
    run = subparsers.add_parser("run", help="Evaluate a local JSON request")
    run.add_argument("--idea", choices=workbench.IDEAS, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in (None, "gui"):
            from kusic.desktop import launch

            launch()
            return 0
        if arguments.command == "template":
            result = workbench.module_for(arguments.idea).demo_payload()
        elif arguments.command == "demo":
            result = workbench.portfolio()
            if arguments.output:
                workbench.write_json(arguments.output / "evaluation.json", result)
                workbench.write_comparison_csv(arguments.output / "comparison.csv", result)
        else:
            result = workbench.run_request(arguments.idea, workbench.read_request(arguments.input))
            if arguments.output:
                workbench.write_json(arguments.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError) as error:
        print(f"Input/action error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

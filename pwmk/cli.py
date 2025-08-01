"""Command-line interface for PWMK."""

import argparse
import sys
from typing import Optional

from . import __version__


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Perspective World Model Kit - Neuro-symbolic AI toolkit"
    )
    
    parser.add_argument(
        "--version",
        action="version", 
        version=f"pwmk {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train world model")
    train_parser.add_argument("--config", help="Configuration file")
    train_parser.add_argument("--data", help="Training data path")
    
    # Evaluate command  
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--model", required=True, help="Model checkpoint path")
    eval_parser.add_argument("--env", help="Environment name")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    demo_parser.add_argument("--scenario", help="Demo scenario")
    
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 1
        
    try:
        if parsed_args.command == "train":
            return _train_command(parsed_args)
        elif parsed_args.command == "eval":
            return _eval_command(parsed_args) 
        elif parsed_args.command == "demo":
            return _demo_command(parsed_args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _train_command(args) -> int:
    """Handle train command."""
    try:
        print("Training world model...")
        print(f"Config: {args.config}")
        print(f"Data: {args.data}")
        return 0
    except Exception as e:
        print(f"Error during training: {e}")
        return 1


def _eval_command(args) -> int:
    """Handle evaluation command."""
    try:
        print("Evaluating model...")
        print(f"Model: {args.model}")
        print(f"Environment: {args.env}")
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


def _demo_command(args) -> int:
    """Handle demo command."""
    print("Running demonstration...")
    print(f"Scenario: {args.scenario}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
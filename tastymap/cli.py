import argparse

import panel as pn

from .ui import TastyKitchen


def main():
    parser = argparse.ArgumentParser(description="Serve the TastyKitchen UI")
    parser.add_argument("command", help="The command to run", choices=["ui"])
    args = parser.parse_args()

    if args.command == "ui":
        pn.serve(TastyKitchen(), port=8888, show=True)


if __name__ == "__main__":
    main()

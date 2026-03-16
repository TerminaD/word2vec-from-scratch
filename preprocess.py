import argparse

parser = argparse.ArgumentParser(description="Dataset preprocessing script")
parser.add_argument("name", help="Person's name")
args = parser.parse_args()

print(f"Hello, {args.name}!")
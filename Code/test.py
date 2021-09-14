import argparse
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data', nargs='+', type=float)
args = parser.parse_args()
print(args.data)
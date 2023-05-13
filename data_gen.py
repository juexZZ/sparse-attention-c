# python data_gen.py 100 10 random
# python data_gen.py 100 10 ordered
import argparse
import numpy as np

# take arguments N and d from command line
parser = argparse.ArgumentParser()
parser.add_argument("N", type=int, help="number of data points")
parser.add_argument("d", type=int, help="number of dimensions")
parser.add_argument("mode", type=str, help="mode of data generation", choices=["random", "ordered"], default="random")
args = parser.parse_args()
# generate numpy random array of size N*d
if args.mode == "ordered":
    query = np.arange(args.N*args.d).reshape(args.N, args.d)
    key = np.arange(args.N*args.d).reshape(args.N, args.d)
    value = np.arange(args.N*args.d).reshape(args.N, args.d)
else: # default is random
    query = np.random.rand(args.N, args.d)
    key = np.random.rand(args.N, args.d)
    value = np.random.rand(args.N, args.d)
# save the data to file in txt format, spaced by \t and \n, each row has dim elements, num rows in total.
np.savetxt("query.txt", query, fmt="%.4f", delimiter="\t", newline="\n")
np.savetxt("key.txt", key, fmt="%.4f", delimiter="\t", newline="\n")
np.savetxt("value.txt", value, fmt="%.4f", delimiter="\t", newline="\n")




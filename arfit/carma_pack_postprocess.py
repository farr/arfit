#!/usr/bin/env python

# Must happen first!
import matplotlib
matplotlib.use('PDF')

import argparse
from arfit.run_carma_pack_posterior import LL, LP
import arfit.utils as u

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', required=True, help='output directory')

    args = parser.parse_args()

    u.process_output_dir(args.dir)

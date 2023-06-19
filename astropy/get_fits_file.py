# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Generates a bash script for downloading light curves.
The input to this script is a CSV file of Kepler targets, for example the DR24
TCE table, which can be downloaded in CSV format from the NASA Exoplanet Archive
at:
  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce
Example usage:
  python generate_download_script.py \
    --kepler_csv_file=dr24_tce.csv \
    --download_dir=${HOME}/astronet/kepler
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import os
import sys
import pandas as pd

all_nums = [x+1 for x in range(20)]

download_dir = 'E:/DR_24_LC/'

kepler_csv_file = ['kep_data/kepData{}.csv'.format(num) for num in all_nums]
print(kepler_csv_file)

output_file = ['shell_new_17052022/get_kepler_{}.sh'.format(num) for num in all_nums]


_WGET_CMD = ("wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -erobots=off "
             "-R 'index*' -A _llc.fits")
_BASE_URL = "http://archive.stsci.edu/pub/kepler/lightcurves"


def main(kep_file, out_file):
  # del argv  # Unused.

  # Read Kepler targets.
  # kepids = set()
  kepids = []
  # with open(FLAGS.kepler_csv_file) as f:
  #   reader = csv.DictReader(row for row in f if not row.startswith("#"))
  #   for row in reader:
  #     kepids.add(row["kepid"])

  # df = pd.read_csv(kepler_csv_file, comment='#', header=0)
  df = pd.read_csv(kep_file, comment='#', header=0)
  

  for i, row in df.iterrows():
    # kepids.add(row['kepid'])
    kepids.append(row['kepid'])
    # print(row['kepid'])

  num_kepids = len(kepids)
  

  # Write wget commands to script file.
  with open(out_file, "w") as f:
    # download_dir = 'E:/DR_24_LC/'
    f.write("#!/bin/sh\n")
    print("HERE")
    f.write("echo 'Downloading {} Kepler targets to {}'\n".format(num_kepids, download_dir))

    
    for i, kepid in enumerate(kepids):
      # download_dir = 'E:/DR_24_LC/'
      if i and not i % 10:
        f.write("echo 'Downloaded {}/{}'\n".format(i, num_kepids))
      kepid = "{0:09d}".format(int(kepid))  # Pad with zeros.
      subdir = "{}/{}".format(kepid[0:4], kepid)
      # print(subdir)
      download_dir_n = os.path.join(download_dir, subdir)
      print(download_dir_n)
      url = "{}/{}/".format(_BASE_URL, subdir)
      f.write("{} -P {} {}\n".format(_WGET_CMD, download_dir_n, url))

    f.write("echo 'Finished downloading {} Kepler targets to {}'\n".format(
        num_kepids, download_dir))
    print("echo 'Finished downloading {}'\n".format(num_kepids))

  os.chmod(out_file, 0o744)  # Make the download script executable.
  print("{} Kepler targets will be downloaded to {}".format(
      num_kepids, out_file))
  print("To start download, run:\n  {}".format("./" + out_file
                                               if "/" not in out_file
                                               else out_file))


if __name__ == "__main__":
  for file, out in zip(kepler_csv_file, output_file):
    print("{} produces output {}".format(file, out))


    main(file, out)
#!/usr/bin/env bash

DNA3SPNGRO_BIN_PATH=~/Workspace/genesis_input_cg_gro/DNA_3SPN2C
export PATH=$PATH:~/Workspace/x3dna-v2.4/bin
export X3DNA=~/Workspace/x3dna-v2.4

# This tool is originally built by de Pablo's group and modified to generate
# input files for Genesis.

if [ $# -ne 1 ]; then
    echo "Usage: $0 <sequence file>"
    exit 1
fi

echo "================================================================================"
echo "Making DNA curvature parameter file..."
$DNA3SPNGRO_BIN_PATH/seq2curv_DNA2c.py $1

echo "--------------------------------------------------------------------------------"
echo "Running X3DNA..."
x3dna_utils cp_std BDNA
rebuild -atomic dna2c.curv bdna_aa.pdb
rm -f Atomic*
rm -f ref_frames.dat
rm -f dna2c.curv

echo "--------------------------------------------------------------------------------"
echo "Making GROMACS itp files for GENESIS..."
$DNA3SPNGRO_BIN_PATH/pdb2gro_3spn2c.py bdna_aa.pdb
echo " DONE!"
echo "================================================================================"

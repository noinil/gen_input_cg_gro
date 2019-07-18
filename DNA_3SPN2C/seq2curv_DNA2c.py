#!/usr/bin/env python3

def read_DNA_sequence(file_name):
    with open(file_name, 'r') as fin:
        for line in fin:
            if line.startswith('>'):
                continue
            seq = line.strip()
            for b in seq:
                if b not in ('A', 'C', 'G', 'T'):
                    break
            else:            
                return seq

def main(DNA_sequence_file_name):
    base_pair_parms = {
        #        a-b,  shear, stretch, stagger, buckle, propeller, opening,  
        'A' : ("A-T",   0.07,   -0.19,    0.07,    1.8,     -15.0,     1.5),
        'T' : ("T-A",  -0.07,   -0.19,    0.07,   -1.8,     -15.0,     1.5),
        'C' : ("C-G",   0.16,   -0.17,    0.15,   -4.9,      -8.7,    -0.6),
        'G' : ("G-C",  -0.16,   -0.17,    0.15,    4.9,      -8.7,    -0.6)
    }
    base_step_parms = {
        #        shift, slide,  rise,  tilt,  roll,  twist
        "AA" : ( -0.05, -0.21,  3.27, -1.84,  0.76,  35.31),
        "AT" : (  0.00, -0.56,  3.39,  0.00, -1.39,  31.21),
        "AC" : (  0.21, -0.54,  3.39, -0.64,  0.91,  31.52),
        "AG" : (  0.12, -0.27,  3.38, -1.48,  3.15,  33.05),
        "TA" : (  0.00,  0.03,  3.34,  0.00,  5.25,  36.20),
        "TT" : (  0.05, -0.21,  3.27,  1.84,  0.91,  35.31),
        "TC" : (  0.27, -0.03,  3.35,  1.52,  3.87,  34.80),
        "TG" : (  0.16,  0.18,  3.38,  0.05,  5.95,  35.02),
        "CA" : ( -0.16,  0.18,  3.38, -0.05,  5.95,  35.02),
        "CT" : ( -0.12, -0.27,  3.38,  1.48,  3.15,  33.05),
        "CC" : (  0.02, -0.47,  3.28,  0.40,  3.86,  33.17),
        "CG" : (  0.00,  0.57,  3.49,  0.00,  4.29,  34.38),
        "GA" : ( -0.27, -0.03,  3.35, -1.52,  3.87,  34.80),
        "GT" : ( -0.21, -0.54,  3.39,  0.64,  0.91,  31.52),
        "GC" : (  0.00, -0.07,  3.38,  0.00,  0.67,  34.38),
        "GG" : ( -0.02, -0.47,  3.28, -0.40,  3.86,  33.17),
        "00" : (  0.00,  0.00,  0.00,  0.00,  0.00,   0.00),
    }
    base_type = ['A', 'T', 'C', 'G']

    # read in DNA sequence:
    seq_DNA = read_DNA_sequence(DNA_sequence_file_name)
    len_DNA = len(seq_DNA)
    print(" DNA lenth: ", len_DNA, " bp.")

    # prepare the output file:
    out_file = open('dna2c.curv', 'w')
    out_file.write("{0:>4d} # bps \n".format(len_DNA))
    out_file.write("   0 # local base-pairing and base-step parameters \n")
    out_file.write("#        Shear    Stretch   Stagger   Buckle   Prop-Tw   Opening     Shift     Slide     Rise      Tilt      Roll      Twist\n")
    out_parm_str = "{bp[0]}  {bp[1]:>9.3f} {bp[2]:>9.3f} {bp[3]:>9.3f} {bp[4]:>9.3f} {bp[5]:>9.3f} {bp[6]:>9.3f} {bs[0]:>9.3f} {bs[1]:>9.3f} {bs[2]:>9.3f} {bs[3]:>9.3f} {bs[4]:>9.3f} {bs[5]:>9.3f}\n"

    # parameter output loop
    for i, b in enumerate(seq_DNA):
        bp_parm = base_pair_parms[b]
        base_step = "00" if i == 0 else seq_DNA[i-1:i+1]
        bs_parm = base_step_parms[base_step]
        out_file.write(out_parm_str.format(bp=bp_parm, bs=bs_parm))

    out_file.close()

if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Generate DNA curvature parameter file from DNA sequence.')
        parser.add_argument('fasta', type=str, help="DNA sequence file name.")
        return parser.parse_args()
    args = parse_arguments()
    main(args.fasta)

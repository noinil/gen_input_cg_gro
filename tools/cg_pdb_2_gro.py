#!/usr/bin/env python3

def main(pdb_name):
    # HEAD: time in the unit of ps
    GRO_HEAD_STR  = "{system_info}, t= {time0:>16.3f} \n"
    # ATOM NUM: free format int
    GRO_ATOM_NUM  = "{atom_num:>12d} \n"
    # XYZ: in the unit of nm!!!
    GRO_ATOM_LINE = "{res_num:>5d}{res_name:>5}{atm_name:>5}{atm_num:>5d}{x:>8.4f}{y:>8.4f}{z:>8.4f}{vx:>8.4f}{vy:>8.4f}{vz:>8.4f} \n"
    GRO_BOX_LINE  = "{box_v1x:>15.4f}{box_v2y:>15.4f}{box_v3z:>15.4f} \n\n"


    # ================
    # Read in PDB info
    # ================
    pdb_lines = []
    with open(pdb_name, 'r') as fin:
        for line in fin:
            if line.startswith('ATOM'):
                if len(line) < 80:
                    line += " "
                pdb_lines.append(line)


    # ==================
    # Output to gro file
    # ==================
    pdb_name_stem = pdb_name[:4]
    gro_name = pdb_name_stem + ".gro"
    gro_file = open(gro_name, 'w')
    gro_file.write(GRO_HEAD_STR.format(system_info="Gro from CG PDB", time0=0))
    gro_file.write(GRO_ATOM_NUM.format(atom_num = len(pdb_lines)))
    for line in pdb_lines:
        atom_serial    = int(line[6:11])
        atom_name      = line[12:16]
        residue_name   = line[17:21]
        chain_id       = line[21]
        residue_serial = int(line[22:26])
        coor_x         = float(line[30:38])
        coor_y         = float(line[38:46])
        coor_z         = float(line[46:54])
        gro_file.write(GRO_ATOM_LINE.format(res_num=residue_serial,
                                            res_name=residue_name,
                                            atm_name=atom_name,
                                            atm_num=atom_serial,
                                            x=coor_x,
                                            y=coor_y,
                                            z=coor_z,
                                            vx=0.0,
                                            vy=0.0,
                                            vz=0.0))
    gro_file.close()

if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Transfer CG PDB into gro.')
        parser.add_argument('pdb', type=str, help="PDB file.")
        return parser.parse_args()
    args = parse_arguments()
    main(args.pdb)

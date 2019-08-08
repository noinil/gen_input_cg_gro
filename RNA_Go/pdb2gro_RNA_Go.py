#!/usr/bin/env python3

import numpy as np
import MDAnalysis 
from tqdm import tqdm

def main(PDB_name, flag_head_phos, flag_psf_output):
    ###########################################################################
    #                         Variables and constants                         #
    ###########################################################################

    # ============================================
    # Constants for CG particle masses and charges
    # ============================================
    std_base_mass = {'A': 134.1, 'G': 150.1, 'C': 110.1, 'U': 111.1}
    std_rPhos_mass = 62.97
    std_rSuga_mass = 131.11
    # k for bond force constants
    bond_k = {'PS': 26.5, 'SR': 40.3, 'SY': 62.9, 'SP': 84.1}
    # k for angle force constants
    angl_k = {'PSR': 18.0, 'PSY': 22.8, 'PSP': 22.1, 'SPS': 47.8}
    # k for dihedral force constants
    dihe_k = {'PSPS': 1.64, 'SPSR': 1.88, 'SPSY': 2.82, 'SPSP': 2.98}
    # e for ST contact:
    epsilon_stack = 2.06
    # e for BP contact:
    epsilon_bpair_2hb = 2.94
    epsilon_bpair_3hb = 5.37
    # e for other contact:
    epsilon_other = { 'SS': 1.48, 'SB': 0.98, 'BB': 0.93 }

    # =============================
    # Other undetermined parameters
    # =============================
    # "NREXCL" in "[moleculetype]"
    MOL_NR_EXCL = 3
    # "CGNR" in "[atoms]"
    CG_ATOM_FUNC_NR = 1
    # "f" in "[bonds]"
    CG_BOND_FUNC_NR = 1
    # "f" in "[angles]"
    CG_ANGL_FUNC_NR = 1
    # "f" in "[dihedral]" for Gaussian
    CG_DIHE_FUNC_NR = 1

    ###########################################################################
    #                     Read in structural info from PDB                    #
    ###########################################################################
    print("> Step 1: open PDB file.")
    u = MDAnalysis.Universe(PDB_name)

    selstr_RNA = "nucleic"

    selstr_P = "(resid {0} and (name P or name OP*))"
    selstr_S = "resid {0} and (name *') and not (name H*)"
    selstr_B = "resid {0} and not (name *' or name OP* or name P or name H*)"
    selstr_RP_cg = "resid {0} and (name P)"
    selstr_RS_cg = "resid {0} and (name *') and not (name H*)"
    selstr_RR_cg = "resid {0} and (name N1)"
    selstr_RY_cg = "resid {0} and (name N3)"

    print("> Step 2: find out CG particles.")
    sel_rna = u.select_atoms(selstr_RNA)

    cg_rna_coors     = []
    cg_rna_p_resname = []
    cg_rna_p_name    = []
    cg_rna_p_charge  = []
    cg_rna_p_mass    = []
    cg_rna_r_ID      = []
    cg_rna_p_ID      = []
    cg_rna_base_type = []

    num_rna_cg_particle = 0
    resid_list = list(sel_rna.residues.resids)
    for i, j in enumerate(tqdm( resid_list )):
        tmp_resname = sel_rna.residues[i].resname
        # Phosphate
        if i > 0 or flag_head_phos == 1:
            num_rna_cg_particle += 1
            res_P = sel_rna.select_atoms(selstr_P.format(j))
            cg_P  = sel_rna.select_atoms(selstr_RP_cg.format(j))
            cg_rna_coors.append(cg_P[0].position)
            cg_rna_p_name.append('RP')
            cg_rna_p_charge.append(-1.0)
            cg_rna_p_mass.append(std_rPhos_mass)
            cg_rna_p_resname.append(tmp_resname)
            cg_rna_p_ID.append(num_rna_cg_particle)
            cg_rna_r_ID.append(i + 1)
            cg_rna_base_type.append['P']
        # Sugar
        num_rna_cg_particle += 1
        res_S = sel_rna.select_atoms(selstr_S.format(j))
        cg_S  = sel_rna.select_atoms(selstr_RS_cg.format(j))
        cg_rna_coors.append( cg_S.center_of_mass() )
        cg_rna_p_name.append('RS')
        cg_rna_p_charge.append(0.0)
        cg_rna_p_mass.append(std_rSuga_mass)
        cg_rna_p_resname.append(tmp_resname)
        cg_rna_p_ID.append(num_rna_cg_particle)
        cg_rna_r_ID.append(i + 1)
        cg_rna_base_type.append['S']
        # Base
        num_rna_cg_particle += 1
        res_B = sel_rna.select_atoms(selstr_B.format(j))
        if tmp_resname[-1] in ["A", "G"]:
            cg_B  = sel_rna.select_atoms(selstr_RR_cg.format(j))
            cg_rna_base_type.append['R']
        else:
            cg_B  = sel_rna.select_atoms(selstr_RY_cg.format(j))
            cg_rna_base_type.append['Y']
        cg_rna_coors.append(cg_B[0].position)
        cg_rna_p_name.append('RB')
        cg_rna_p_charge.append(0.0)
        cg_rna_p_mass.append(std_base_mass[tmp_resname[-1]])
        cg_rna_p_resname.append(tmp_resname)
        cg_rna_p_ID.append(num_rna_cg_particle)
        cg_rna_r_ID.append(i + 1)
    cg_rna_p_num = num_rna_cg_particle


    ###########################################################################
    #                             Output .psf file                            #
    ###########################################################################
    def output_psf():
        """Output psf file for protein-RNA complex.
        """
        PSF_HEAD_STR = "PSF CMAP \n\n"
        PSF_TITLE_STR0 = "      3 !NTITLE \n"
        PSF_TITLE_STR1 = "REMARKS PSF file created with GENESIS CG tools. \n"
        PSF_TITLE_STR2 = "REMARKS RNA: {0:>5d} bases. \n"
        PSF_TITLE_STR5 = "REMARKS ======================================== \n"
        PSF_TITLE_STR6 = "       \n"
        PSF_TITLE_STR = PSF_TITLE_STR0 + PSF_TITLE_STR1 + PSF_TITLE_STR2 + PSF_TITLE_STR5 + PSF_TITLE_STR6
        PSF_ATOM_TITLE = " {atom_num:>6d} !NATOM \n"
        PSF_ATOM_LINE = " {atom_ser:>6d} {seg_id:>3} {res_ser:>5d} {res_name:>3} {atom_name:>3} {atom_type:>5}  {charge:>10.6f}  {mass:>10.6f}          0 \n"

        psf_file_name = "rna_cg.psf"
        psf_file = open(psf_file_name, 'w')
        psf_file.write(PSF_HEAD_STR)
        psf_file.write(PSF_TITLE_STR.format(cg_rna_r_num))
        psf_file.write(PSF_ATOM_TITLE.format(atom_num = cg_rna_p_num))

        for i in range(cg_rna_p_num):
            psf_file.write(PSF_ATOM_LINE.format(atom_ser  = cg_rna_p_ID[i],
                                                seg_id    = 'a',
                                                res_ser   = cg_rna_r_ID[i],
                                                res_name  = cg_rna_p_resname[i],
                                                atom_name = cg_rna_p_name[i],
                                                atom_type = cg_rna_p_name[i][-1],
                                                charge    = cg_rna_p_charge[i],
                                                mass      = cg_rna_p_mass[i]))
        psf_file.close()

    if flag_psf_output:
        output_psf()


    ###########################################################################
    #                       Determine .itp parameters                       #
    ###########################################################################
    def compute_bond(coor1, coor2):
        vec = coor1 - coor2
        return np.linalg.norm(vec)
    def compute_angle(coor1, coor2, coor3):
        vec1 = coor1 - coor2
        vec2 = coor3 - coor2
        n_v1 = np.linalg.norm(vec1)
        n_v2 = np.linalg.norm(vec2)
        return np.arccos(np.clip(np.dot(vec1, vec2) / n_v1 / n_v2, -1.0, 1.0)) / np.pi * 180.0
    def compute_dihedral(coor1, coor2, coor3, coor4):
        v_12 = coor2 - coor1
        v_23 = coor3 - coor2
        v_34 = coor4 - coor3
        n123 = np.cross(v_12, v_23)
        n234 = np.cross(v_23, v_34)
        norm_n123 = np.linalg.norm(n123)
        norm_n234 = np.linalg.norm(n234)
        dih = np.arccos(np.clip(np.dot(n123, n234) / norm_n123 / norm_n234, -1.0, 1.0))
        # determine sign of dih
        n1234 = np.cross(n123, n234)
        zajiao = np.dot(n1234, v_23)
        if zajiao < 0:
            dih = - dih
        return (dih - np.pi) / np.pi * 180.0

    rna_atm_list = []
    rna_bnd_list = []
    rna_ang_list = []
    rna_dih_list = []
    rna_contact_hb_list = []
    rna_contact_st_list = []
    rna_contact_nn_list = []

    print("> Step 3. Determine bond/angle/dihedral/contacts: ")
    for i_rna in tqdm( range(cg_rna_p_num) ):
        if cg_rna_p_name[i_rna] == "RS":
            # atom S
            rna_atm_list.append((cg_rna_p_ID      [i_rna],
                                    cg_rna_p_name    [i_rna],
                                    cg_rna_r_ID      [i_rna],
                                    cg_rna_p_resname [i_rna],
                                    cg_rna_p_name    [i_rna],
                                    cg_rna_p_charge  [i_rna],
                                    cg_rna_p_mass    [i_rna]))
            # bond S--B
            coor_s = cg_rna_coors[i_rna]
            coor_b = cg_rna_coors[i_rna + 1]
            r_sb = compute_bond(coor_s, coor_b)
            bond_type = "S" + cg_rna_base_type[i_rna + 1]
            k = bond_k[bond_type]
            rna_bnd_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 1], r_sb / 10, k * 2))

            # bond S--P+1
            if i_rna + 2 < cg_rna_p_num:
                coor_p3 = cg_rna_coors[i_rna + 2]
                r_sp3 = compute_bond(coor_s, coor_p3)
                k = bond_k["SP"]
                rna_bnd_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 2], r_sp3 / 10, k * 2))
            if i_rna + 4 < cg_rna_p_num:
                # Angle S--P+1--S+1
                coor_p3 = cg_rna_coors[i_rna + 2]
                coor_s3 = cg_rna_coors[i_rna + 3]
                ang_sp3s3 = compute_angle(coor_s, coor_p3, coor_s3)
                k = angl_k["SPS"]
                rna_ang_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 2], cg_rna_p_ID[i_rna + 3], ang_sp3s3, k * 2))
                # Dihedral S--P+1--S+1--B+1
                coor_b3 = cg_rna_coors[i_rna + 4]
                dih_sp3s3b3 = compute_dihedral(coor_s, coor_p3, coor_s3, coor_b3)
                dihe_type = "SPS" + cg_rna_base_type[i_rna + 4]
                k = dihe_k[dihe_type]
                rna_dih_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 2], cg_rna_p_ID[i_rna + 3], cg_rna_p_ID[i_rna + 4], dih_sp3s3b3, k))
                # Dihedral S--P+1--S+1--P+2
                if i_rna + 5 < cg_rna_p_num:
                    coor_p33 = cg_rna_coors[i_rna + 5]
                    dih_sp3s3p33 = compute_dihedral(coor_s, coor_p3, coor_s3, coor_p33)
                    k = dihe_k["SPSP"]
                    rna_dih_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 2], cg_rna_p_ID[i_rna + 3], cg_rna_p_ID[i_rna + 5], dih_sp3s3p33, k))
        elif cg_rna_p_name[i_rna] == "DP":
            # atom P
            rna_atm_list.append((cg_rna_p_ID      [i_rna],
                                    cg_rna_p_name    [i_rna],
                                    cg_rna_r_ID      [i_rna],
                                    cg_rna_p_resname [i_rna],
                                    cg_rna_p_name    [i_rna],
                                    cg_rna_p_charge  [i_rna],
                                    cg_rna_p_mass    [i_rna]))
            # bond P--S
            coor_p = cg_rna_coors[i_rna]
            coor_s = cg_rna_coors[i_rna + 1]
            r_ps = compute_bond(coor_p, coor_s)
            k = bond_k["PS"]
            rna_bnd_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 1], r_ps / 10, k * 2))
            # angle P--S--B
            coor_b = cg_rna_coors[i_rna + 2]
            ang_psb = compute_angle(coor_p, coor_s, coor_b)
            angl_type = "PS" + cg_rna_base_type[i_rna + 2]
            k = angl_k[angl_type]
            rna_ang_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 1], cg_rna_p_ID[i_rna + 2], ang_psb, k * 2))
            if i_rna + 3 < cg_rna_p_num:
                # angle P--S--P+1
                coor_p3 = cg_rna_coors[i_rna + 3]
                ang_psp3 = compute_angle(coor_p, coor_s, coor_p3)
                k = angl_k["PSP"]
                rna_ang_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 1], cg_rna_p_ID[i_rna + 3], ang_psp3, k * 2))
                # Dihedral P--S--P+1--S+1
                coor_s3 = cg_rna_coors[i_rna + 4]
                dih_psp3s3 = compute_dihedral(coor_p, coor_s, coor_p3, coor_s3)
                k = dihe_k["PSPS"]
                rna_dih_P_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna + 1], cg_rna_p_ID[i_rna + 3], cg_rna_p_ID[i_rna + 4], dih_psp3s3, k))
        elif cg_rna_p_name[i_rna] == "DB":
            # atom Base
            rna_atm_list.append((cg_rna_p_ID[i_rna],
                                    cg_rna_p_resname[i_rna],
                                    cg_rna_r_ID[i_rna],
                                    cg_rna_p_resname[i_rna],
                                    cg_rna_p_name[i_rna],
                                    cg_rna_p_charge[i_rna],
                                    cg_rna_p_mass[i_rna]))
            # if i_rna + 3 < cg_rna_p_num:
            #     # angle B--S--P+1
            #     resname5 = cg_rna_p_resname[i_rna][-1]
            #     resname3 = cg_rna_p_resname[i_rna + 1][-1]
            #     coor_b = cg_rna_coors[i_rna]
            #     coor_s = cg_rna_coors[i_rna - 1]
            #     coor_p3 = cg_rna_coors[i_rna + 1]
            #     ang_bsp3 = compute_angle(coor_b, coor_s, coor_p3)
            #     k = get_angle_param("BSP", resname5 + resname3)
            #     rna_ang_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna - 1], cg_rna_p_ID[i_rna + 1], ang_bsp3, k * 2))
            #     # Dihedral B--S--P+1--S+1
            #     coor_s3 = cg_rna_coors[i_rna + 2]
            #     dih_bsp3s3 = compute_dihedral(coor_b, coor_s, coor_p3, coor_s3)
            #     rna_dih_P_list.append((cg_rna_p_ID[i_rna], cg_rna_p_ID[i_rna - 1], cg_rna_p_ID[i_rna + 1], cg_rna_p_ID[i_rna + 2], dih_bsp3s3))
        else:
            print("Error! Wrong RNA type...")
                

    # ================
    # Output .itp file
    # ================
    def output_itp():
        itp_mol_head = "[ moleculetype ] \n"
        itp_mol_comm = ";{0:15} {1:>6} \n".format("name", "nrexcl")
        itp_mol_line = "{0:16} {1:>6d} \n"

        itp_atm_head = "[ atoms ] \n"
        itp_atm_comm = ";{0:>9}{1:>5}{2:>10}{3:>5}{4:>5}{5:>5}{6:>8}{7:>8}\n".format("nr", "type", "resnr", "res", "atom", "cg", "charge", "mass")
        itp_atm_line = "{atm[0]:>10d}{atm[1]:>5}{atm[2]:>10d}{atm[3]:>5}{atm[4]:>5}{cgnr:>5d}{atm[5]:>8.3f}{atm[6]:>8.3f} \n"

        itp_bnd_head = "[ bonds ] \n"
        itp_bnd_comm = ";{0:>9}{1:>10}{2:>5}{3:>18}{4:>18} \n".format("i", "j", "f", "eq", "k2")
        itp_bnd_line = "{bond[0]:>10d}{bond[1]:>10d}{functype:>5d}{bond[2]:>18.4E}{bond[3]:>18.4E} \n"

        itp_ang_head = "[ angles ] \n"
        itp_ang_comm = ";{0:>9}{1:>10}{2:>10}{3:>5}{4:>18}{5:>18} \n".format("i", "j", "k", "f", "eq", "k")
        itp_ang_line = "{ang[0]:>10d}{ang[1]:>10d}{ang[2]:>10d}{functype:>5d}{ang[3]:>18.4E}{ang[4]:>18.4E} \n"

        itp_dih_head = "[ dihedrals ] \n"
        itp_dih_comm = ";{0:>9}{1:>10}{2:>10}{3:>10}{4:>5}{5:>18}{6:>18}{7:>18} \n".format("i", "j", "k", "l", "f", "eq", "k", "w")
        itp_dih_line = "{dih[0]:>10d}{dih[1]:>10d}{dih[2]:>10d}{dih[3]:>10d}{functype:>5d}{dih[4]:>18.4E}{dih[5]:>18.4E} \n"


        itp_name = "rna_strand{0}.itp".format(j + 1)
        itp_file = open(itp_name, 'w')
        # write molecule type information
        itp_strand_name = "rna_strand_{0}".format(j + 1)
        itp_file.write(itp_mol_head)
        itp_file.write(itp_mol_comm)
        itp_file.write(itp_mol_line.format(itp_strand_name, MOL_NR_EXCL))
        itp_file.write("\n")
        # write atoms information
        itp_file.write(itp_atm_head)
        itp_file.write(itp_atm_comm)
        for i, a in enumerate(rna_atm_list):
            itp_file.write(itp_atm_line.format(atm=a, cgnr=CG_ATOM_FUNC_NR))
        itp_file.write("\n")
        # write bond information
        itp_file.write(itp_bnd_head)
        itp_file.write(itp_bnd_com1)
        for i, b in enumerate(rna_bnd_list):
            itp_file.write(itp_bnd_line.format(bond=b, functype=CG_BOND_FUNC4_NR))
        itp_file.write("\n")
        # write angle information
        itp_file.write(itp_ang_head)
        itp_file.write(itp_ang_comm)
        for i, a in enumerate(rna_ang_list):
            itp_file.write(itp_ang_line.format(ang=a, functype=CG_ANG_FUNC_NR))
        itp_file.write("\n")
        # write dihedral information
        itp_file.write(itp_dih_head)
        itp_file.write(itp_dih_comm)
        for i, d in enumerate(rna_dih_list):
            itp_file.write(itp_dih_line.format(dih=d, functype=CG_DIH_GAUSS_FUNC_NR))
        itp_file.write("\n")

        itp_file.close()

    print("> Step 4: output topology information to itp.")
    output_itp()

    # ================
    # Output .gro file
    # ================
    def output_gro():
        """Output .gro file for RNA complex.
        """
        # HEAD: time in the unit of ps
        GRO_HEAD_STR  = "{system_info}, t= {time0:>16.3f} \n" 
        # ATOM NUM: free format int
        GRO_ATOM_NUM  = "{atom_num:>12d} \n"
        # XYZ: in the unit of nm!!!
        GRO_ATOM_LINE = "{res_num:>5d}{res_name:>5}{atm_name:>5}{atm_num:>5d} {x:>8.4f} {y:>8.4f} {z:>8.4f} {vx:>8.4f} {vy:>8.4f} {vz:>8.4f} \n"
        GRO_BOX_LINE  = "{box_v1x:>15.4f}{box_v2y:>15.4f}{box_v3z:>15.4f} \n\n"

        gro_file_name = "rna_cg.gro"
        gro_file = open(gro_file_name, 'w')
        gro_file.write(GRO_HEAD_STR.format(system_info="RNA 3SPN.2C model", time0=0))
        gro_file.write(GRO_ATOM_NUM.format(atom_num = cg_rna_p_num))

        for i in range(cg_rna_p_num):
            gro_file.write(GRO_ATOM_LINE.format(atm_num  = cg_rna_p_ID[i],
                                                res_num  = cg_rna_r_ID[i],
                                                res_name = cg_rna_p_resname[i],
                                                atm_name = cg_rna_p_name[i],
                                                x        = cg_rna_coors[i][0] / 10,  
                                                y        = cg_rna_coors[i][1] / 10,
                                                z        = cg_rna_coors[i][2] / 10,
                                                vx       = 0.0,
                                                vy       = 0.0,
                                                vz       = 0.0))
        gro_file.write(GRO_BOX_LINE.format(box_v1x =0.0, box_v2y =0.0, box_v3z =0.0))
        gro_file.close()
    print("> Step 5: output coordinate information to gro.")
    output_gro()

    print("[1;32m DONE! [0m ")
    print(" Please check the .itp and .gro files.")


if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Generate 3SPN.2C .itp and .gro files from RNA PDB.')
        parser.add_argument('pdb', type=str, help="PDB file name.")
        parser.add_argument('-n', '--psf', help="Output a simple psf file.", action="store_true")
        parser.add_argument('-P', '--headPHOS', help="Specify whether the 5' phosphate group will be used or not.", action="store_true")
        return parser.parse_args()
    args = parse_arguments()
    flag_head_phos = 1 if args.headPHOS else 0

    print("> Welcome!")
    print("> This tool helps you prepare CG RNA files for MD simulations in Genesis.")
    print("> ------ ")
    main(args.pdb, flag_head_phos, args.psf)

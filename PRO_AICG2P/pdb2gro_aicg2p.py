#!/usr/bin/env python3

import numpy as np
import MDAnalysis

def main(PDB_name, is_scaled):
    ###########################################################################
    #                         Variables and constants                         #
    ###########################################################################

    protein_name = PDB_name[:-4]

    # ==========================
    # Protein general parameters
    # ==========================
    aa_mass_dict = {
        'ALA' :   71.09,
        'ARG' :  156.19,
        'ASN' :  114.11,
        'ASP' :  115.09,
        'CYS' :  103.15,
        'GLN' :  128.14,
        'GLU' :  129.12,
        'GLY' :   57.05,
        'HIS' :  137.14,
        'ILE' :  113.16,
        'LEU' :  113.16,
        'LYS' :  128.17,
        'MET' :  131.19,
        'PHE' :  147.18,
        'PRO' :   97.12,
        'SER' :   87.08,
        'THR' :  101.11,
        'TRP' :  186.21,
        'TYR' :  163.18,
        'VAL' :   99.14
    }

    aa_charge_dict = {
        'ALA' :  0.0,
        'ARG' :  1.0,
        'ASN' :  0.0,
        'ASP' : -1.0,
        'CYS' :  0.0,
        'GLN' :  0.0,
        'GLU' : -1.0,
        'GLY' :  0.0,
        'HIS' :  0.0,
        'ILE' :  0.0,
        'LEU' :  0.0,
        'LYS' :  1.0,
        'MET' :  0.0,
        'PHE' :  0.0,
        'PRO' :  0.0,
        'SER' :  0.0,
        'THR' :  0.0,
        'TRP' :  0.0,
        'TYR' :  0.0,
        'VAL' :  0.0
    }

    # =============================
    # AICG2+ Force Field Parameters
    # =============================
    CAL2JOU     = 4.184
    # bond force constant
    bond_k        = 110.40 * CAL2JOU * 100 * 2  # kcal * mol^-1 * nm^-2
    # sigma for Gaussian angle
    ang_gauss_sigma = 0.15 * 0.1  # nm
    # sigma for Gaussian dihedral
    dih_gauss_sigma = 0.15      # Rad ??
    # atomistic contact cutoff
    dfcontact = 6.5
    # AICG2+ pairwise interaction cutoff
    con_atm_cutoff = 5.0
    # AICG2+ hydrogen bond cutoff
    hb_cutoff = 3.2
    # AICG2+ salt bridge cutoff
    sb_cutoff = 3.5
    # AICG2+ energy cutoffs
    ene_local_upper_lim = -0.5
    ene_local_lower_lim = -5.0

    # AICG2+ pairwise interaction pairs
    ITYPE_BB_HB = 1  # B-B hydrogen bonds
    ITYPE_BB_DA = 2  # B-B donor-accetor contacts
    ITYPE_BB_CX = 3  # B-B carbon-X contacts
    ITYPE_BB_xx = 4  # B-B other
    ITYPE_SS_HB = 5  # S-S hydrogen bonds
    ITYPE_SS_SB = 6  # S-S salty bridge
    ITYPE_SS_DA = 7  # S-S donor-accetor contacts
    ITYPE_SS_CX = 8  # S-S carbon-X contacts
    ITYPE_SS_QX = 9  # S-S charge-X contacts
    ITYPE_SS_xx = 10 # S-S other
    ITYPE_SB_HB = 11 # S-B hydrogen bonds
    ITYPE_SB_DA = 12 # S-B donor-accetor contacts
    ITYPE_SB_CX = 13 # S-B carbon-X contacts
    ITYPE_SB_QX = 14 # S-B charge-X contacts
    ITYPE_SB_xx = 15 # S-B other
    ITYPE_LR_CT = 16 # long range contacts
    ITYPE_offst = 0  # offset

    pairwise_energy = [0 for i in range(17)]
    pairwise_energy[ITYPE_BB_HB] = - 1.4247   # B-B hydrogen bonds
    pairwise_energy[ITYPE_BB_DA] = - 0.4921   # B-B donor-accetor contacts
    pairwise_energy[ITYPE_BB_CX] = - 0.2404   # B-B carbon-X contacts
    pairwise_energy[ITYPE_BB_xx] = - 0.1035   # B-B other
    pairwise_energy[ITYPE_SS_HB] = - 5.7267   # S-S hydrogen bonds
    pairwise_energy[ITYPE_SS_SB] = -12.4878   # S-S salty bridge
    pairwise_energy[ITYPE_SS_DA] = - 0.0308   # S-S donor-accetor contacts
    pairwise_energy[ITYPE_SS_CX] = - 0.1113   # S-S carbon-X contacts
    pairwise_energy[ITYPE_SS_QX] = - 0.2168   # S-S charge-X contacts
    pairwise_energy[ITYPE_SS_xx] =   0.2306   # S-S other
    pairwise_energy[ITYPE_SB_HB] = - 3.4819   # S-B hydrogen bonds
    pairwise_energy[ITYPE_SB_DA] = - 0.1809   # S-B donor-accetor contacts
    pairwise_energy[ITYPE_SB_CX] = - 0.1209   # S-B carbon-X contacts
    pairwise_energy[ITYPE_SB_QX] = - 0.2984   # S-B charge-X contacts
    pairwise_energy[ITYPE_SB_xx] = - 0.0487   # S-B other
    pairwise_energy[ITYPE_LR_CT] = - 0.0395   # long range contacts
    pairwise_energy[ITYPE_offst] = - 0.1051   # offset


    # =============================
    # Other undetermined parameters
    # =============================
    # "NREXCL" in "[moleculetype]"
    MOL_NR_EXCL          = 3
    # "CGNR" in "[atoms]"
    CG_ATOM_FUNC_NR      = 1
    # "f" in "[bonds]"
    CG_BOND_FUNC_TYPE    = 1
    # "f" in AICG-type "[angles]"
    CG_ANG_G_FUNC_TYPE   = 21
    # "f" in Flexible-type "[angles]"
    CG_ANG_F_FUNC_TYPE   = 22
    # "f" in AICG-type "[dihedral]"
    CG_DIH_G_FUNC_TYPE   = 21
    # "f" in Flexible-type "[dihedral]"
    CG_DIH_F_FUNC_TYPE   = 22
    # "f" in Go-contacts "[pairs]"
    CG_CONTACT_FUNC_TYPE = 22


    ###########################################################################
    #                           Basic Math Functions                          #
    ###########################################################################
    def compute_distance(coor1, coor2):
        vec = coor1 - coor2
        return np.linalg.norm(vec)
    def compute_angle(coor1, coor2, coor3):
        vec1 = coor1 - coor2
        vec2 = coor3 - coor2
        n_v1 = np.linalg.norm(vec1)
        n_v2 = np.linalg.norm(vec2)
        return np.arccos(np.clip(np.dot(vec1, vec2) / n_v1 / n_v2, -1.0, 1.0)) / np.pi * 180.0
    def compute_dihedral(coor1, coor2, coor3, coor4):
        v_12      = coor2 - coor1
        v_23      = coor3 - coor2
        v_34      = coor4 - coor3
        n123      = np.cross(v_12, v_23)
        n234      = np.cross(v_23, v_34)
        norm_n123 = np.linalg.norm(n123)
        norm_n234 = np.linalg.norm(n234)
        dih       = np.arccos(np.clip(np.dot(n123, n234) / norm_n123 / norm_n234, -1.0, 1.0))
        n1234     = np.cross(n123, n234)
        zajiao    = np.dot(n1234, v_23)
        if zajiao < 0:
            dih   = - dih
        return (dih - np.pi) / np.pi * 180.0
    def is_backbone(atom_name):
        if atom_name in ['N', 'C', 'O', 'OXT', 'CA']:
            return True
        return False
    def is_hb_donor(atom_name, res_name):
        if atom_name.startswith( 'N' ):
            return True
        if atom_name.startswith( 'S' ) and res_name == 'CYS':
            return True
        if atom_name.startswith( 'O' ):
            if ( res_name == 'SER' and atom_name == 'OG'  ) or \
               ( res_name == 'THR' and atom_name == 'OG1' ) or \
               ( res_name == 'TYR' and atom_name == 'OH'  ):
                return True
        return False
    def is_hb_acceptor(atom_name):
        if atom_name.startswith('O') or atom_name.startwith('S'):
            return True
        return False
    def is_cation(atom_name, res_name):
        if atom_name.startswith( 'N' ):
            if ( res_name == 'ARG' and atom_name == 'NH1' ) or \
               ( res_name == 'ARG' and atom_name == 'NH2' ) or \
               ( res_name == 'LYS' and atom_name == 'NZ'  ):
                return True
        return False
    def is_anion(atom_name, res_name):
        if atom_name.startswith( 'O' ):
            if ( res_name == 'GLU' and atom_name == 'OE1' ) or \
               ( res_name == 'GLU' and atom_name == 'OE2' ) or \
               ( res_name == 'ASP' and atom_name == 'OD1' ) or \
               ( res_name == 'ASP' and atom_name == 'OD2' ):
                return True
        return False
    def is_hb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2):
        if is_hb_acceptor(atom_name_1, res_name_1) and is_hb_donor(atom_name_2, res_name_2):
            return True
        if is_hb_acceptor(atom_name_2, res_name_2) and is_hb_donor(atom_name_1, res_name_1):
            return True
        return False
    def is_sb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2):
        if is_cation(atom_name_1, res_name_1) and is_anion(atom_name_2, res_name_2):
            return True
        if is_cation(atom_name_2, res_name_2) and is_anion(atom_name_1, res_name_1):
            return True
        return False
    def is_anti_sb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2):
        if is_cation(atom_name_1, res_name_1) and is_cation(atom_name_2, res_name_2):
            return True
        if is_anion(atom_name_2, res_name_2)  and is_anion(atom_name_1, res_name_1):
            return True
        return False

    def count_atomic_contact(resid1, resid2):
        """
        Keyword Arguments:
        resid1 -- residue 1
        resid2 -- residue 2
        Return:
        contact_count -- array of contact counts in each itype
        """
        contact_count = [0 for i in range(17)]
        res_name_1 = resid1.resname
        res_name_2 = resid2.resname
        num_short_range_contact = 0
        for atom1 in resid1.atoms:
            atom_name_1 = atom1.name
            if atom_name_1.startswith('H'):
                continue
            coor_1 = atom1.position
            for atom2 in resid2.atoms:
                atom_name_2 = atom2.name
                if atom_name_2.startswith('H'):
                    continue
                coor_2 = atom2.position
                dist_12 = compute_distance(coor_1, coor_2)

                is_hb = is_hb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2)
                is_sb = is_sb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2)
                is_anti_sb = is_anti_sb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2)
                is_1_backbone = is_backbone(atom_name_1)
                is_2_backbone = is_backbone(atom_name_2)
                if dist_12 < dfcontact:
                    contact_count[ITYPE_LR_CT] += 1
                if dist_12 < con_atm_cutoff:
                    num_short_range_contact += 1
                    if is_1_backbone and is_2_backbone:
                        if is_hb:
                            if dist_12 < hb_cutoff:
                                contact_count[ITYPE_BB_HB] += 1
                            else:
                                contact_count[ITYPE_BB_DA] += 1
                        elif atom_name_1.startswith('C') or atom_name_2.startswith('C'):
                            contact_count[ITYPE_BB_CX] += 1
                        else:
                            contact_count[ITYPE_BB_xx] += 1
                    elif ( not is_1_backbone ) and ( not is_2_backbone ):
                        if is_hb:
                            if is_sb:
                                if dist_12 < sb_cutoff:
                                    contact_count[ITYPE_SS_SB] += 1
                                else:
                                    contact_count[ITYPE_SS_QX] += 1
                            elif dist_12 < hb_cutoff:
                                contact_count[ITYPE_SS_HB] += 1
                            elif is_anti_sb:
                                contact_count[ITYPE_SS_QX] += 1
                            else:
                                contact_count[ITYPE_SS_DA] += 1
                        elif is_anti_sb:
                            contact_count[ITYPE_SS_QX] += 1
                        elif atom_name_1.startswith('C') or atom_name_2.startswith('C'):
                            contact_count[ITYPE_SS_CX] += 1
                        else:
                            contact_count[ITYPE_SS_xx] += 1
                    elif ( is_1_backbone and ( not is_2_backbone ) ) or \
                         ( is_2_backbone and ( not is_1_backbone ) ):
                        if is_hb:
                            if dist_12 < hb_cutoff:
                                contact_count[ITYPE_SB_HB] += 1
                            elif is_anti_sb:
                                contact_count[ITYPE_SB_QX] += 1
                            else:
                                contact_count[ITYPE_SB_DA] += 1
                        elif is_anti_sb:
                            contact_count[ITYPE_SB_QX] += 1
                        elif atom_name_1.startswith('C') or atom_name_2.startswith('C'):
                            contact_count[ITYPE_SB_CX] += 1
                        else:
                            contact_count[ITYPE_SB_xx] += 1
        # control the number of long-range contacts
        if dfcontact > con_atm_cutoff:
            contact_count[ITYPE_LR_CT] -= num_short_range_contact
        else:
            contact_count[ITYPE_LR_CT] = 0

        # control the number of salty bridge
        if contact_count[ITYPE_SS_SB] >= 2:
            contact_count[ITYPE_SS_QX] += contact_count[ITYPE_SS_SB] - 1
            contact_count[ITYPE_SS_SB] = 1
        return contact_count
            

    ###########################################################################
    #  ___ _   _ ____  _   _ _____ 
    # |_ _| \ | |  _ \| | | |_   _|
    #  | ||  \| | |_) | | | | | |  
    #  | || |\  |  __/| |_| | | |  
    # |___|_| \_|_|    \___/  |_|  
    # 
    ###########################################################################
    u = MDAnalysis.Universe(PDB_name)
    pro_atom_group = u.select_atoms("protein")

    # Number of CG particles
    cg_pro_num     = len( pro_atom_group.residues )

    # ===============
    # Core structures
    # ===============
    cg_pro_coors    = np.empty([cg_pro_num, 3])
    top_cg_pro_atoms = []
    top_cg_pro_bonds = []
    top_cg_pro_angles = []
    top_cg_pro_dihedrals = []
    top_cg_pro_13 = []
    top_cg_pro_14 = []
    param_cg_pro_e_13 = []
    param_cg_pro_e_14 = []

    # ====================================
    # Get coordinates and basic properties
    # ====================================
    calpha_list  = pro_atom_group.select_atoms("name CA")

    # check the number
    if len(calpha_list) != cg_pro_num:
        print(" ERROR! Number of C_alpha is not equal to number of residues!")
        exit()

    for i, ca in enumerate(calpha_list):
        cg_pro_coors[i] = ca.position
        charge = aa_charge_dict[ca.resname]
        mass = aa_mass_dict[ca.resname]
        top_cg_pro_atoms.append((i + 1,      # residue index
                                 ca.resname, # residue name
                                 i + 1,      # atom index
                                 "CA",       # atom name
                                 charge,      
                                 mass))
    
    ###########################################################################
    #                           AICG2+ Calculations                           #
    ###########################################################################

    # =====
    # bonds
    # =====
    for i in range(cg_pro_num - 1):
        cai = calpha_list[i]
        caj = calpha_list[i + 1]
        segi = cai.segid
        segj = caj.segid
        if segi != segj:
            continue
        coori = cai.position
        coorj = caj.position
        dist_ij = compute_distance(coori, coorj)
        top_cg_pro_bonds.append((i, i + 1, dist_ij))

    # ======
    # angles
    # ======
    e_ground_local = 0.0
    e_ground_13 = 0.0
    mba = 0
    for i in range(cg_pro_num - 2):
        cai = calpha_list[i]
        cak = calpha_list[i + 2]
        segi = cai.segid
        segk = cak.segid
        if segk != segi:
            continue
        coori = cai.position
        coork = cak.position
        dist_ik = compute_distance(coori, coork)
        top_cg_pro_angles.append(i)
        top_cg_pro_13.append((i, dist_ik))

        # count AICG2+ atomic contact
        ri = pro_atom_group.residues[i]
        rj = pro_atom_group.residues[i + 2]
        contact_counts = count_atomic_contact(ri, rj)

        # calculate AICG2+ pairwise energy
        e_local = pairwise_energy[ITYPE_offst]
        for k, w in enumerate( contact_counts ):
            e_local += w * pairwise_energy[k]
        if e_local > ene_local_upper_lim:
            e_local = ene_local_upper_lim
        if e_local < ene_local_lower_lim:
            e_local = ene_local_lower_lim
        e_ground_local += e_local
        e_ground_13 += e_local
        mba += mba
        param_cg_pro_e_13.append(e_local)

    # =========
    # dihedrals
    # =========
    for i in range(cg_pro_num - 3):
        cai = calpha_list[i]
        caj = calpha_list[i + 1]
        cak = calpha_list[i + 2]
        cal = calpha_list[i + 3]
        segi = cai.segid
        segl = cal.segid
        if segl != segi:
            continue
        coori = cai.position
        coorj = caj.position
        coork = cak.position
        coorl = cal.position
        dihed = compute_dihedral(coori, coorj, coork, coorl)
        top_cg_pro_dihedrals.append(i)
        top_cg_pro_14.append((i, dihed))

    ###########################################################################
    #   ___  _   _ _____ ____  _   _ _____ 
    #  / _ \| | | |_   _|  _ \| | | |_   _|
    # | | | | | | | | | | |_) | | | | | |  
    # | |_| | |_| | | | |  __/| |_| | | |  
    #  \___/ \___/  |_| |_|    \___/  |_|  
    # 
    ###########################################################################

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
        itp_bnd_com1 = ";{0:>9}{1:>10}{2:>5}{3:>18}{4:>18} \n".format("i", "j", "f", "eq", "k2")
        itp_bnd_com2 = ";{0:>9}{1:>10}{2:>5}{3:>18}{4:>18} \n".format("i", "j", "f", "eq", "k4")
        itp_bnd_line = "{bond[0]:>10d}{bond[1]:>10d}{functype:>5d}{bond[2]:>18.4E}{k:>18.4E} \n"

        itp_ang_head = "[ angles ] \n"
        itp_ang_comm = ";{0:>9}{1:>10}{2:>10}{3:>5}{4:>18}{5:>18} \n".format("i", "j", "k", "f", "eq", "k")
        itp_ang_line = "{ang[0]:>10d}{ang[1]:>10d}{ang[2]:>10d}{functype:>5d}{ang[3]:>18.4E}{ang[4]:>18.4E} \n"

        itp_dih_G_head = "[ dihedrals ] \n"
        itp_dih_G_comm = ";{0:>9}{1:>10}{2:>10}{3:>10}{4:>5}{5:>18}{6:>18}{7:>18} \n".format("i", "j", "k", "l", "f", "eq", "k", "w")
        itp_dih_G_line = "{dih[0]:>10d}{dih[1]:>10d}{dih[2]:>10d}{dih[3]:>10d}{functype:>5d}{dih[4]:>18.4E}{k:>18.4E}{sig:>18.4E} \n"

        itp_dih_P_head = "[ dihedrals ] \n"
        itp_dih_P_comm = ";{0:>9}{1:>10}{2:>10}{3:>10}{4:>5}{5:>18}{6:>18}{7:>5} \n".format("i", "j", "k", "l", "f", "eq", "k", "n")
        itp_dih_P_line = "{dih[0]:>10d}{dih[1]:>10d}{dih[2]:>10d}{dih[3]:>10d}{functype:>5d}{dih[4]:>18.4E}{k:>18.4E}{n:>5d} \n"


        for j in range(2):
            itp_name = "dna_strand{0}.itp".format(j + 1)
            itp_file = open(itp_name, 'w')
            strnd_atm_list = dna_atm_list[j]
            strnd_bnd_list = dna_bnd_list[j]
            strnd_ang_list = dna_ang_list[j]
            strnd_dih_P_list = dna_dih_P_list[j]
            strnd_dih_G_list = dna_dih_G_list[j]
            # write molecule type information
            itp_strand_name = "dna_strand_{0}".format(j + 1)
            itp_file.write(itp_mol_head)
            itp_file.write(itp_mol_comm)
            itp_file.write(itp_mol_line.format(itp_strand_name, MOL_NR_EXCL))
            itp_file.write("\n")
            # write atoms information
            itp_file.write(itp_atm_head)
            itp_file.write(itp_atm_comm)
            for i, a in enumerate(strnd_atm_list):
                itp_file.write(itp_atm_line.format(atm=a, cgnr=CG_ATOM_FUNC_NR))
            itp_file.write("\n")
            # write bond information
            itp_file.write(itp_bnd_head)
            itp_file.write(itp_bnd_com1)
            for i, b in enumerate(strnd_bnd_list):
                itp_file.write(itp_bnd_line.format(bond=b, functype=CG_BOND_FUNC4_NR, k=bond_k_2*2))
            # itp_file.write(itp_bnd_com2)
            # for i, b in enumerate(strnd_bnd_list):
                # itp_file.write(itp_bnd_line.format(bond=b, functype=CG_BOND_FUNC4_NR, k=bond_k_4))
            itp_file.write("\n")
            # write angle information
            itp_file.write(itp_ang_head)
            itp_file.write(itp_ang_comm)
            for i, a in enumerate(strnd_ang_list):
                itp_file.write(itp_ang_line.format(ang=a, functype=CG_ANG_FUNC_NR))
            itp_file.write("\n")
            # write Gaussian dihedral information
            itp_file.write(itp_dih_G_head)
            itp_file.write(itp_dih_G_comm)
            for i, d in enumerate(strnd_dih_G_list):
                itp_file.write(itp_dih_G_line.format(dih=d, functype=CG_DIH_GAUSS_FUNC_NR, k=dih_gauss_k, sig=dih_gauss_sigma))
            itp_file.write("\n")
            # write Periodic dihedral information
            itp_file.write(itp_dih_P_head)
            itp_file.write(itp_dih_P_comm)
            for i, d in enumerate(strnd_dih_P_list):
                itp_file.write(itp_dih_P_line.format(dih=d, functype=CG_DIH_PERIODIC_FUNC_NR, k=dih_periodic_k, n=CG_DIH_PERIODIC_FUNC_SP))
            itp_file.write("\n")

            itp_file.close()

    # output_itp()

    # ================
    # Output .gro file
    # ================
    def output_gro():
        """Output .gro file
        """
        # HEAD: time in the unit of ps
        GRO_HEAD_STR  = "{system_info}, t= {time0:>16.3f} \n"
        # ATOM NUM: free format int
        GRO_ATOM_NUM  = "{atom_num:>12d} \n"
        # XYZ: in the unit of nm!!!
        # GRO_ATOM_LINE = "{res_num:>5d}{res_name:>5}{atm_name:>5}{atm_num:>5d} {x:>8.4f} {y:>8.4f} {z:>8.4f} {vx:>8.4f} {vy:>8.4f} {vz:>8.4f} \n"
        GRO_ATOM_LINE = "{a[0]:>5d}{a[1]:>5}{a[3]:>5}{a[2]:>5d} {x[0]:>8.4f} {x[1]:>8.4f} {x[2]:>8.4f} {v[0]:>8.4f} {v[1]:>8.4f} {v[2]:>8.4f} \n"
        GRO_BOX_LINE  = "{box_v1x:>15.4f}{box_v2y:>15.4f}{box_v3z:>15.4f} \n\n"

        gro_file_name = protein_name + ".gro"
        gro_file = open(gro_file_name, 'w')
        gro_file.write(GRO_HEAD_STR.format(system_info="Protein modeled by AICG2+ ", time0=0))
        gro_file.write(GRO_ATOM_NUM.format(atom_num = cg_pro_num))

        for i in range(cg_pro_num):
            gro_file.write(GRO_ATOM_LINE.format(a = top_cg_pro_atoms[i],
                                                x = cg_pro_coors[i] / 10,
                                                v = [0.0, 0.0, 0.0]))
        gro_file.write(GRO_BOX_LINE.format(box_v1x =0.0, box_v2y =0.0, box_v3z =0.0))
        gro_file.close()
    output_gro()


if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Generate 3SPN.2C .itp and .gro files from DNA PDB.')
        parser.add_argument('pdb', type=str, help="PDB file name.")
        parser.add_argument('-s', '--scale', help="Scale AICG local interactions.", action="store_true")
        return parser.parse_args()
    args = parse_arguments()
    main(args.pdb, args.scale)

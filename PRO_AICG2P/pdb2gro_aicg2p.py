#!/usr/bin/env python3

import numpy as np
import MDAnalysis
from tqdm import tqdm

def main(PDB_name, scale_scheme):
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
    CAL2JOU                = 4.184
    # bond force constant
    BOND_K                 = 110.40 * CAL2JOU * 100 * 2  # kcal * mol^-1 * nm^-2
    # sigma for Gaussian angle
    AICG2P_ANG_GAUSS_SIGMA = 0.15 * 0.1  # nm
    # sigma for Gaussian dihedral
    AICG2P_DIH_GAUSS_SIGMA = 0.15        # Rad ??
    # atomistic contact cutoff
    GO_ATOMIC_CUTOFF       = 6.5
    # AICG2+ pairwise interaction cutoff
    AICG2P_ATOMIC_CUTOFF   = 5.0
    # AICG2+ hydrogen bond cutoff
    HYDROGEN_BOND_CUTOFF   = 3.2
    # AICG2+ salt bridge cutoff
    SALT_BRIDGE_CUTOFF     = 3.5
    # AICG2+ energy cutoffs
    AICG2P_ENE_UPPER_LIM   = -0.5
    AICG2P_ENE_LOWER_LIM   = -5.0
    # average and general AICG2+ energy values
    AICG2P_13_AVE          = 1.72
    AICG2P_14_AVE          = 1.23
    AICG2P_CONTACT_AVE     = 0.55
    AICG2P_13_GEN          = 1.11
    AICG2P_14_GEN          = 0.87
    AICG2P_CONTACT_GEN     = 0.32

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

    aicg2p_pairwise_energy = [0 for i in range(17)]
    aicg2p_pairwise_energy[ITYPE_BB_HB] = - 1.4247   # B-B hydrogen bonds
    aicg2p_pairwise_energy[ITYPE_BB_DA] = - 0.4921   # B-B donor-accetor contacts
    aicg2p_pairwise_energy[ITYPE_BB_CX] = - 0.2404   # B-B carbon-X contacts
    aicg2p_pairwise_energy[ITYPE_BB_xx] = - 0.1035   # B-B other
    aicg2p_pairwise_energy[ITYPE_SS_HB] = - 5.7267   # S-S hydrogen bonds
    aicg2p_pairwise_energy[ITYPE_SS_SB] = -12.4878   # S-S salty bridge
    aicg2p_pairwise_energy[ITYPE_SS_DA] = - 0.0308   # S-S donor-accetor contacts
    aicg2p_pairwise_energy[ITYPE_SS_CX] = - 0.1113   # S-S carbon-X contacts
    aicg2p_pairwise_energy[ITYPE_SS_QX] = - 0.2168   # S-S charge-X contacts
    aicg2p_pairwise_energy[ITYPE_SS_xx] =   0.2306   # S-S other
    aicg2p_pairwise_energy[ITYPE_SB_HB] = - 3.4819   # S-B hydrogen bonds
    aicg2p_pairwise_energy[ITYPE_SB_DA] = - 0.1809   # S-B donor-accetor contacts
    aicg2p_pairwise_energy[ITYPE_SB_CX] = - 0.1209   # S-B carbon-X contacts
    aicg2p_pairwise_energy[ITYPE_SB_QX] = - 0.2984   # S-B charge-X contacts
    aicg2p_pairwise_energy[ITYPE_SB_xx] = - 0.0487   # S-B other
    aicg2p_pairwise_energy[ITYPE_LR_CT] = - 0.0395   # long range contacts
    aicg2p_pairwise_energy[ITYPE_offst] = - 0.1051   # offset


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
    CG_CONTACT_FUNC_TYPE = 2


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
        # return (dih - np.pi) / np.pi * 180.0
        return dih / np.pi * 180.0
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
        if atom_name.startswith('O') or atom_name.startswith('S'):
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
        if is_hb_acceptor(atom_name_1) and is_hb_donor(atom_name_2, res_name_2):
            return True
        if is_hb_acceptor(atom_name_2) and is_hb_donor(atom_name_1, res_name_1):
            return True
        return False
    def is_sb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2):
        if is_cation(atom_name_1, res_name_1) and is_anion(atom_name_2, res_name_2):
            return True
        if is_cation(atom_name_2, res_name_2) and is_anion(atom_name_1, res_name_1):
            return True
        return False
    def is_nonsb_charge_pair(atom_name_1, res_name_1, atom_name_2, res_name_2):
        if is_cation(atom_name_1, res_name_1) or is_anion(atom_name_1, res_name_1) or \
           is_cation(atom_name_2, res_name_2) or is_anion(atom_name_2, res_name_2):
            return True
        return False
    def is_go_contact(resid1, resid2):
        """
        Keyword Arguments:
        resid1 -- residue 1
        resid2 -- residue 2
        Return:
        min_dist -- minimum distance between heavy atoms in two residues
        """
        for atom1 in resid1.atoms:
            atom_name_1     = atom1.name
            if atom_name_1.startswith('H'):
                continue
            coor_1          = atom1.position
            for atom2 in resid2.atoms:
                atom_name_2 = atom2.name
                if atom_name_2.startswith('H'):
                    continue
                coor_2      = atom2.position
                dist_12     = compute_distance(coor_1, coor_2)
                if dist_12 < GO_ATOMIC_CUTOFF:
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
        contact_count              = [0 for i in range(17)]
        contact_count[ITYPE_offst] = 1
        res_name_1                 = resid1.resname
        res_name_2                 = resid2.resname
        num_short_range_contact    = 0
        for atom1 in resid1.atoms:
            atom_name_1 = atom1.name
            if atom_name_1.startswith('H'):
                continue
            coor_1      = atom1.position
            for atom2 in resid2.atoms:
                atom_name_2     = atom2.name
                if atom_name_2.startswith('H'):
                    continue
                coor_2          = atom2.position
                dist_12         = compute_distance(coor_1, coor_2)

                is_hb           = is_hb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2)
                is_sb           = is_sb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2)
                is_nonsb_charge = is_nonsb_charge_pair(atom_name_1, res_name_1, atom_name_2, res_name_2)
                is_1_backbone   = is_backbone(atom_name_1)
                is_2_backbone   = is_backbone(atom_name_2)
                if dist_12 < GO_ATOMIC_CUTOFF:
                    contact_count[ITYPE_LR_CT] += 1
                if dist_12 < AICG2P_ATOMIC_CUTOFF:
                    num_short_range_contact    += 1
                    if is_1_backbone and is_2_backbone:
                        if is_hb:
                            if dist_12 < HYDROGEN_BOND_CUTOFF:
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
                                if dist_12 < SALT_BRIDGE_CUTOFF:
                                    contact_count[ITYPE_SS_SB] += 1
                                else:
                                    contact_count[ITYPE_SS_QX] += 1
                            elif dist_12 < HYDROGEN_BOND_CUTOFF:
                                contact_count[ITYPE_SS_HB] += 1
                            elif is_nonsb_charge:
                                contact_count[ITYPE_SS_QX] += 1
                            else:
                                contact_count[ITYPE_SS_DA] += 1
                        elif is_nonsb_charge:
                            contact_count[ITYPE_SS_QX] += 1
                        elif atom_name_1.startswith('C') or atom_name_2.startswith('C'):
                            contact_count[ITYPE_SS_CX] += 1
                        else:
                            contact_count[ITYPE_SS_xx] += 1
                    elif ( is_1_backbone and ( not is_2_backbone ) ) or \
                         ( is_2_backbone and ( not is_1_backbone ) ):
                        if is_hb:
                            if dist_12 < HYDROGEN_BOND_CUTOFF:
                                contact_count[ITYPE_SB_HB] += 1
                            elif is_nonsb_charge:
                                contact_count[ITYPE_SB_QX] += 1
                            else:
                                contact_count[ITYPE_SB_DA] += 1
                        elif is_nonsb_charge:
                            contact_count[ITYPE_SB_QX] += 1
                        elif atom_name_1.startswith('C') or atom_name_2.startswith('C'):
                            contact_count[ITYPE_SB_CX] += 1
                        else:
                            contact_count[ITYPE_SB_xx] += 1
        # control the number of long-range contacts
        if GO_ATOMIC_CUTOFF > AICG2P_ATOMIC_CUTOFF:
            contact_count[ITYPE_LR_CT] -= num_short_range_contact
        else:
            contact_count[ITYPE_LR_CT] = 0

        # control the number of salty bridge
        if contact_count[ITYPE_SS_SB] >= 2:
            contact_count[ITYPE_SS_QX] += contact_count[ITYPE_SS_SB] - 1
            contact_count[ITYPE_SS_SB] = 1
        return contact_count[:]
            

    ###########################################################################
    #  ___ _   _ ____  _   _ _____ 
    # |_ _| \ | |  _ \| | | |_   _|
    #  | ||  \| | |_) | | | | | |  
    #  | || |\  |  __/| |_| | | |  
    # |___|_| \_|_|    \___/  |_|  
    # 
    ###########################################################################
    print("> Step 1: open PDB file.")
    u = MDAnalysis.Universe(PDB_name)
    pro_atom_group = u.select_atoms("protein")

    # Number of CG particles
    cg_pro_num     = len( pro_atom_group.residues )

    # ===============
    # Core structures
    # ===============
    cg_pro_coors           = np.empty([cg_pro_num, 3])
    top_cg_pro_atoms       = []
    top_cg_pro_bonds       = []
    top_cg_pro_angles      = []
    top_cg_pro_dihedrals   = []
    top_cg_pro_13          = []
    top_cg_pro_14          = []
    top_cg_pro_contact     = []
    param_cg_pro_e_13      = []
    param_cg_pro_e_14      = []
    param_cg_pro_e_contact = []

    # ====================================
    # Get coordinates and basic properties
    # ====================================
    calpha_list  = pro_atom_group.select_atoms("name CA")

    # check the number
    if len(calpha_list) != cg_pro_num:
        print(" ERROR! Number of C_alpha is not equal to number of residues!")
        exit()

    print("> Step 2: find out alpha carbons.")
    for i, ca in enumerate(tqdm( calpha_list )):
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
    print("> Step 3: calculate bonds.")
    for i in tqdm( range(cg_pro_num - 1) ):
        cai      = calpha_list[i]
        caj      = calpha_list[i + 1]
        segi     = cai.segid
        segj     = caj.segid
        if segi != segj:
            continue
        coori    = cai.position
        coorj    = caj.position
        dist_ij  = compute_distance(coori, coorj)
        top_cg_pro_bonds.append((i + 1, dist_ij))

    # ======
    # angles
    # ======
    print("> Step 4: calculate angles.")
    e_ground_local = 0.0
    e_ground_13    = 0.0
    num_angle      = 0
    for i in tqdm( range(cg_pro_num - 2) ):
        cai  = calpha_list[i]
        cak  = calpha_list[i + 2]
        segi = cai.segid
        segk = cak.segid
        if segk != segi:
            continue
        coori   = cai.position
        coork   = cak.position
        dist_ik = compute_distance(coori, coork)
        top_cg_pro_angles.append(i + 1)
        top_cg_pro_13.append((i + 1, dist_ik))

        # count AICG2+ atomic contact
        ri = pro_atom_group.residues[i]
        rk = pro_atom_group.residues[i + 2]
        contact_counts = count_atomic_contact(ri, rk)

        # calculate AICG2+ pairwise energy
        e_local = 0
        for n, w in enumerate( contact_counts ):
            e_local += w * aicg2p_pairwise_energy[n]
        if e_local > AICG2P_ENE_UPPER_LIM:
            e_local = AICG2P_ENE_UPPER_LIM
        if e_local < AICG2P_ENE_LOWER_LIM:
            e_local = AICG2P_ENE_LOWER_LIM
        e_ground_local += e_local
        e_ground_13    += e_local
        num_angle      += 1
        param_cg_pro_e_13.append(e_local)

    # =========
    # dihedrals
    # =========
    print("> Step 5: calculate dihedrals.")
    e_ground_14 = 0.0
    num_dih = 0
    for i in tqdm( range(cg_pro_num - 3) ):
        cai   = calpha_list[i]
        caj   = calpha_list[i + 1]
        cak   = calpha_list[i + 2]
        cal   = calpha_list[i + 3]
        segi  = cai.segid
        segl  = cal.segid
        if segl != segi:
            continue
        coori = cai.position
        coorj = caj.position
        coork = cak.position
        coorl = cal.position
        dihed = compute_dihedral(coori, coorj, coork, coorl)
        top_cg_pro_dihedrals.append(i + 1)
        top_cg_pro_14.append((i + 1, dihed))

        # count AICG2+ atomic contact
        ri = pro_atom_group.residues[i]
        rl = pro_atom_group.residues[i + 3]
        contact_counts = count_atomic_contact(ri, rl)

        # calculate AICG2+ pairwise energy
        e_local = 0
        for n, w in enumerate( contact_counts ):
            e_local += w * aicg2p_pairwise_energy[n]
        if e_local > AICG2P_ENE_UPPER_LIM:
            e_local = AICG2P_ENE_UPPER_LIM
        if e_local < AICG2P_ENE_LOWER_LIM:
            e_local = AICG2P_ENE_LOWER_LIM
        e_ground_local += e_local
        e_ground_14 += e_local
        num_dih += 1
        param_cg_pro_e_14.append(e_local)

    # =========
    # Normalize
    # =========
    e_ground_local /= (num_angle + num_dih)
    e_ground_13    /= num_angle
    e_ground_14    /= num_dih
    
    if scale_scheme == 0:
        for i in range(len(param_cg_pro_e_13)):
            param_cg_pro_e_13[i] *= AICG2P_13_AVE / e_ground_13
        for i in range(len(param_cg_pro_e_14)):
            param_cg_pro_e_14[i] *= AICG2P_14_AVE / e_ground_14
    elif scale_scheme == 1:
        for i in range(len(param_cg_pro_e_13)):
            param_cg_pro_e_13[i] *= -AICG2P_13_GEN
        for i in range(len(param_cg_pro_e_14)):
            param_cg_pro_e_14[i] *= -AICG2P_14_GEN

    # ======================================================
    # Native contacts and aicg2+ pairwise energy coefficient
    # ======================================================
    print("> Step 6: calculate native contacts.")
    e_ground_contact = 0.0
    num_contact = 0
    for i in tqdm( range(cg_pro_num - 4) ):
        cai = calpha_list[i]
        coor_cai = cai.position
        residi = pro_atom_group.residues[i]
        for j in range(i + 4, cg_pro_num):
            caj = calpha_list[j]
            coor_caj = caj.position
            residj = pro_atom_group.residues[j]
            if is_go_contact(residi, residj):
                native_dist = compute_distance(coor_cai, coor_caj)
                num_contact += 1
                top_cg_pro_contact.append((i + 1, j + 1, native_dist))

                # count AICG2+ atomic contact
                contact_counts = count_atomic_contact(residi, residj)

                # calculate AICG2+ pairwise energy
                e_local = 0
                for n, w in enumerate( contact_counts ):
                    e_local += w * aicg2p_pairwise_energy[n]
                if e_local > AICG2P_ENE_UPPER_LIM:
                    e_local = AICG2P_ENE_UPPER_LIM
                if e_local < AICG2P_ENE_LOWER_LIM:
                    e_local = AICG2P_ENE_LOWER_LIM
                e_ground_contact += e_local
                num_contact += 1
                param_cg_pro_e_contact.append(e_local)

    # normalize
    e_ground_contact /= num_contact

    if scale_scheme == 0:
        for i in range(len(param_cg_pro_e_contact)):
            param_cg_pro_e_contact[i] *= AICG2P_CONTACT_AVE / e_ground_contact
    elif scale_scheme == 1:
        for i in range(len(param_cg_pro_e_contact)):
            param_cg_pro_e_contact[i] *= -AICG2P_CONTACT_GEN


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
        itp_mol_head = "[ moleculetype ]\n"
        itp_mol_comm = ";{0:15} {1:>6}\n".format("name", "nrexcl")
        itp_mol_line = "{0:16} {1:>6d}\n"

        itp_atm_head = "[ atoms ]\n"
        itp_atm_comm = ";{0:>9}{1:>5}{2:>10}{3:>5}{4:>5}{5:>5} {6:>8} {7:>8}\n".format("nr", "type", "resnr", "res", "atom", "cg", "charge", "mass")
        itp_atm_line = "{a[2]:>10d}{a[1]:>5}{a[0]:>10d}{a[1]:>5}{a[3]:>5}{cgnr:>5d} {a[4]:>8.3f} {a[5]:>8.3f}\n"
        # itp_atm_line = "{a[2]:>6d}{a[1]:>8}{a[0]:>8d}{a[1]:>8}{a[3]:>8}{cgnr:>8d}{a[4]:>8.3f}{a[5]:>8.3f}\n"

        itp_bnd_head = "[ bonds ]\n"
        itp_bnd_comm = ";{0:>9}{1:>10}{2:>5}{3:>18}{4:>18}\n".format("i", "j", "f", "eq", "k2")
        itp_bnd_line = "{bi:>10d}{bj:>10d}{functype:>5d}{eq:>18.4E}{k:>18.4E}\n"
        # itp_bnd_line = "{bi:>5d}{bj:>5d}{functype:>5d}{eq:>15.4E}{k:>15.4E}\n"

        itp_13_head = "[ angles ] ; AICG2+ 1-3 interaction\n"
        itp_13_comm = ";{0:>9}{1:>10}{2:>10}{3:>5}{4:>15}{5:>15}{6:>15}\n".format("i", "j", "k", "f", "eq", "k", "w")
        itp_13_line = "{ai:>10d}{aj:>10d}{ak:>10d}{functype:>5d}{eq:>15.4E}{k:>15.4E}{w:>15.4E}\n"
        # itp_13_line = "{ai:>5d}{aj:>5d}{ak:>5d}{functype:>5d}{eq:>15.4E}{k:>15.4E}{w:>15.4E}\n"

        itp_ang_head = "[ angles ] ; AICG2+ flexible local interaction\n"
        itp_ang_comm = ";{0:>9}{1:>10}{2:>10}{3:>5}\n".format("i", "j", "k", "f")
        itp_ang_line = "{ai:>10d}{aj:>10d}{ak:>10d}{functype:>5d}\n"
        # itp_ang_line = "{ai:>5d}{aj:>5d}{ak:>5d}{functype:>5d}\n"

        itp_dih_G_head = "[ dihedrals ] ; AICG2+ Gaussian dihedrals\n"
        itp_dih_G_comm = ";{0:>9}{1:>10}{2:>10}{3:>10}{4:>5}{5:>15}{6:>15}{7:>15}\n".format("i", "j", "k", "l", "f", "eq", "k", "w")
        itp_dih_G_line = "{di:>10d}{dj:>10d}{dk:>10d}{dl:>10d}{functype:>5d}{eq:>15.4E}{k:>15.4E}{sig:>15.4E}\n"
        # itp_dih_G_line = "{di:>5d}{dj:>5d}{dk:>5d}{dl:>5d}{functype:>5d}{eq:>15.4E}{k:>15.4E}{sig:>15.4E}\n"

        itp_dih_F_head = "[ dihedrals ] ; AICG2+ flexible local interation\n"
        itp_dih_F_comm = ";{0:>9}{1:>10}{2:>10}{3:>10}{4:>5}\n".format("i", "j", "k", "l", "f")
        itp_dih_F_line = "{di:>10d}{dj:>10d}{dk:>10d}{dl:>10d}{functype:>5d}\n"
        # itp_dih_F_line = "{di:>5d}{dj:>5d}{dk:>5d}{dl:>5d}{functype:>5d}\n"

        itp_contact_head = "[ pairs ] ; Go-type native contact\n"
        itp_contact_comm = ";{0:>9}{1:>10}{2:>10}{3:>15}{4:>15}\n".format("i", "j", "f", "eq", "k")
        itp_contact_line = "{di:>10d}{dj:>10d}{functype:>10d}{eq:>15.4E}{k:>15.4E}\n"
        # itp_contact_line = "{di:>5d}{dj:>5d}{functype:>5d}{eq:>15.4E}{k:>15.4E}\n"

        itp_exc_head = "[ exclusions ] ; Genesis exclusion list\n"
        itp_exc_comm = ";{0:>9}{1:>10}\n".format("i", "j")
        itp_exc_line = "{di:>10d}{dj:>10d}\n"
        # itp_exc_line = "{di:>5d}{dj:>5d}\n"


        itp_name = protein_name + ".itp"
        itp_file = open(itp_name, 'w')

        # write molecule type information
        itp_system_name = "Pro_{0}".format(protein_name)
        itp_file.write(itp_mol_head)
        itp_file.write(itp_mol_comm)
        itp_file.write(itp_mol_line.format(itp_system_name, MOL_NR_EXCL))
        itp_file.write("\n")

        # write atoms information
        itp_file.write(itp_atm_head)
        itp_file.write(itp_atm_comm)
        for i, atom in enumerate(top_cg_pro_atoms):
            itp_file.write(itp_atm_line.format(a=atom, cgnr=CG_ATOM_FUNC_NR))
        itp_file.write("\n")

        # write bond information
        itp_file.write(itp_bnd_head)
        itp_file.write(itp_bnd_comm)
        for i, b in enumerate(top_cg_pro_bonds):
            itp_file.write(itp_bnd_line.format(bi       = b[0],
                                               bj       = b[0] + 1,
                                               functype = CG_BOND_FUNC_TYPE,
                                               eq       = b[1] * 0.1,
                                               k        = BOND_K))
        itp_file.write("\n")

        # write 13 interaction information
        itp_file.write(itp_13_head)
        itp_file.write(itp_13_comm)
        for i, a in enumerate(top_cg_pro_13):
            itp_file.write(itp_13_line.format(ai       = a[0],
                                               aj       = a[0] + 1,
                                               ak       = a[0] + 2,
                                               functype = CG_ANG_G_FUNC_TYPE,
                                               eq       = a[1] * 0.1,
                                               k        = param_cg_pro_e_13[i] * CAL2JOU,
                                               w        = AICG2P_ANG_GAUSS_SIGMA))
        itp_file.write("\n")

        # write angle interaction information
        itp_file.write(itp_ang_head)
        itp_file.write(itp_ang_comm)
        for i, a in enumerate(top_cg_pro_angles):
            itp_file.write(itp_ang_line.format(ai       = a,
                                               aj       = a + 1,
                                               ak       = a + 2,
                                               functype = CG_ANG_F_FUNC_TYPE))
        itp_file.write("\n")

        # write Gaussian dihedral information
        itp_file.write(itp_dih_G_head)
        itp_file.write(itp_dih_G_comm)
        for i, d in enumerate(top_cg_pro_14):
            itp_file.write(itp_dih_G_line.format(di       = d[0],
                                                 dj       = d[0] + 1, 
                                                 dk       = d[0] + 2, 
                                                 dl       = d[0] + 3, 
                                                 functype = CG_DIH_G_FUNC_TYPE,
                                                 eq       = d[1],
                                                 k        = param_cg_pro_e_14[i] * CAL2JOU,
                                                 sig      = AICG2P_DIH_GAUSS_SIGMA))
        itp_file.write("\n")

        # write local flexible dihedral information
        itp_file.write(itp_dih_F_head)
        itp_file.write(itp_dih_F_comm)
        for i, d in enumerate(top_cg_pro_dihedrals):
            itp_file.write(itp_dih_F_line.format(di       = d,
                                                 dj       = d + 1, 
                                                 dk       = d + 2, 
                                                 dl       = d + 3, 
                                                 functype = CG_DIH_F_FUNC_TYPE))
        itp_file.write("\n")

        # write Go-type native contacts
        itp_file.write(itp_contact_head)
        itp_file.write(itp_contact_comm)
        for i, c in enumerate(top_cg_pro_contact):
            itp_file.write(itp_contact_line.format(di       = c[0],
                                                   dj       = c[1], 
                                                   functype = CG_CONTACT_FUNC_TYPE,
                                                   eq       = c[2] * 0.1,
                                                   k        = param_cg_pro_e_contact[i] * CAL2JOU))
        itp_file.write("\n")

        # write Genesis local-exclusion list
        itp_file.write(itp_exc_head)
        itp_file.write(itp_exc_comm)
        for i, c in enumerate(top_cg_pro_contact):
            itp_file.write(itp_exc_line.format(di       = c[0],
                                               dj       = c[1]))
        itp_file.write("\n")

        itp_file.close()

    print("> Step 7: output topology information to itp.")
    output_itp()

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

    print("> Step 8: output coordinate information to gro.")
    output_gro()


if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Generate 3SPN.2C .itp and .gro files from DNA PDB.')
        parser.add_argument('pdb', type=str, help="PDB file name.")
        parser.add_argument('-s', '--scale', type=int, choices=[0, 1], default=0,
                            help="Scale local interactions: 0) average; 1) general")
        return parser.parse_args()
    args = parse_arguments()
    print("> Welcome!")
    print("> This tool helps you prepare CG protein files for MD simulations in Genesis.")
    print("> ------ ")
    main(args.pdb, args.scale)

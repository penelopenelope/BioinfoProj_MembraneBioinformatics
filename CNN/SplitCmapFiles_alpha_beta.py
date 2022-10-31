import shutil
import os
import argparse
import random

#Check if folder exists, if not create folder
def check_folder_exists(folder_name):
  if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input selected cmap threshold.')
    parser.add_argument('cmap_threshold', help='an integer for the cmap generation', type=int)
    args = parser.parse_args()

    # copy alpha helix cmaps to new folder
    ah_target_folder = './alpha_helix/'
    check_folder_exists(ah_target_folder)
    alpha_helix_file = open('./AlphaHelix.txt', 'r')
    alpha_helix_entries = alpha_helix_file.readlines()
    for ah_entry in alpha_helix_entries:
        file_path = '../FeatureEngineering/SUM_contact_aa_maps_' + str(args.cmap_threshold) + '/' + ah_entry[:-1] + '_SUM_CMap'
        #print(file_path)
        #print(os.path.exists(file_path))
        if os.path.exists(file_path):
            ah_original_path = os.fsdecode(file_path)
            shutil.copy(ah_original_path, ah_target_folder)

    # shuffle alpha helix cmaps to train and test sets & delete the original folders
    ah_files = os.listdir(ah_target_folder)
    ah_files.sort()
    random.seed(2022)
    random.shuffle(ah_files)

    ah_split = int(0.8 * len(ah_files))
    ah_train_set = ah_files[:ah_split]
    ah_test_set = ah_files[ah_split:]
    check_folder_exists('./train' + str(args.cmap_threshold) + '/alpha_helix/')
    for ah_train_file in ah_train_set:
        ori_train_abspath = os.path.abspath(ah_target_folder + ah_train_file)
        target_train_abspath = './train' + str(args.cmap_threshold) + '/alpha_helix/' + ah_train_file
        shutil.move(ori_train_abspath, target_train_abspath)
    check_folder_exists('./test' + str(args.cmap_threshold) + '/alpha_helix/')
    for ah_test_file in ah_test_set:
        ori_test_abspath = os.path.abspath(ah_target_folder + ah_test_file)
        target_test_abspath = './test' + str(args.cmap_threshold) + '/alpha_helix/' + ah_test_file
        shutil.move(ori_test_abspath, target_test_abspath)

    os.rmdir(ah_target_folder)

    # copy beta strand cmaps to new folder 
    bs_target_folder = './beta_strand/'
    check_folder_exists(bs_target_folder)
    beta_strand_file = open('./BetaStrand.txt', 'r')
    beta_strand_entries = beta_strand_file.readlines()
    for bs_entry in beta_strand_entries:
        file_path = '../FeatureEngineering/SUM_contact_aa_maps_' + str(args.cmap_threshold) + '/' + bs_entry[:-1] + '_SUM_CMap'
        #print(file_path)
        #print(os.path.exists(file_path))
        if os.path.exists(file_path):
            bs_original_path = os.fsdecode(file_path)
            shutil.copy(bs_original_path, bs_target_folder)

    # shuffle beta sheet cmaps to train and test sets & delete the original folders
    bs_files = os.listdir(bs_target_folder)
    bs_files.sort()
    random.seed(2022)
    random.shuffle(bs_files)

    bs_split = int(0.8 * len(bs_files))
    bs_train_set = bs_files[:bs_split]
    bs_test_set = bs_files[bs_split:]
    check_folder_exists('./train' + str(args.cmap_threshold) + '/beta_strand/')
    for bs_train_file in bs_train_set:
        ori_train_abspath = os.path.abspath(bs_target_folder + bs_train_file)
        target_train_abspath = './train' + str(args.cmap_threshold) + '/beta_strand/' + bs_train_file
        shutil.move(ori_train_abspath, target_train_abspath)
    check_folder_exists('./test' + str(args.cmap_threshold) + '/beta_strand/')
    for bs_test_file in bs_test_set:
        ori_test_abspath = os.path.abspath(bs_target_folder + bs_test_file)
        target_test_abspath = './test' + str(args.cmap_threshold) + '/beta_strand/' + bs_test_file
        shutil.move(ori_test_abspath, target_test_abspath)

    os.rmdir(bs_target_folder)

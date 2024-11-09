import json

def load_mappings(wnids_file_path, idxs_file_path, words_file_path):
    """
    Load mappings from WNID to index, index to words, and WNID to words.
    
    Args:
        wnids_file_path (str): Path to the 'wnids.txt' file.   (n02124075)
        idxs_file_path (str): Path to the 'idxs.txt' file.     (tench, Tinca tinca)
        words_file_path (str): Path to the 'words.txt' file.   (n00001740	entity)
    
    Returns:
        dict: A dictionary containing:
            - 'wnids_to_idxs': Mapping from WNIDs to numerical indexs.  (n02124075 -> 285)
            - 'idxs_to_words': Mapping from indexs to words.            (285 -> Egyptian cat)
            - 'wnids_to_words': Mapping from WNIDs to words.            (n02124075 -> Egyptian cat)
    """

    words_to_idxs = {}
    with open(idxs_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        # Create a mapping: words in idxs.txt -> idxs
        words_to_idxs = {word: idx for idx, word in enumerate(lines)}

        # Load WNIDs from wnids.txt
    with open(wnids_file_path, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]

    # Find the words of (the WNIDs in wdin.txt) in words.txt
    wnid_name_mapping = {}
    with open(words_file_path, 'r') as f:
        for line in f:
            wnid, name = line.strip().split('\t', 1)  # Split by tab
            wnid_name_mapping[wnid] = name

    # Create a mapping: WNIDs in wnids.txt -> words
    wnids_to_words = {wnid: wnid_name_mapping.get(wnid, "Unknown") for wnid in wnids}

    # Create a mapping: WNIDs in wnids.txt -> idxs
    wnids_to_idxs = {}
    for wnid in wnids:
        word = wnids_to_words.get(wnid, None)
        if word is not None and word in words_to_idxs:
            wnids_to_idxs[wnid] = words_to_idxs[word]

    return {
        'words_to_idxs': words_to_idxs,
        'wnids_to_words': wnids_to_words,
        'wnids_to_idxs': wnids_to_idxs
    }

def map_annotations(val_annotations_path, mapping_path, output_path):
    """
    Map the encoding beginning with n in val_annotations.txt 
    to the same index as in tiny-imagenet-mapping.txt.
    Parameters:
    val_annotations_path (str): the path for val_annotations.txt
    mapping_path (str): the path for tiny-imagenet-mapping.txt 
    output_path (str):  the output path for mapping result
    """
    n_code_to_index = {}

    with open(mapping_path, 'r') as mapping_file:
        for index, line in enumerate(mapping_file):
            parts = line.strip().split('\t')
            if parts[0].startswith('n'):  # Make sure it starts with n
                n_code_to_index[parts[0]] = index  # index starts at 0


    with open(val_annotations_path, 'r') as val_file, open(output_path, 'w') as output_file:
        for line in val_file:
            parts = line.strip().split('\t')
            if parts[1] in n_code_to_index:  # Check that the second column is in the mapping dictionary
                new_index = n_code_to_index[parts[1]]
                output_file.write(f"{parts[0]}\t{new_index}\t{parts[2]}\t{parts[3]}\t{parts[4]}\t{parts[5]}\n")

    print(f"The Mapping is complete, the result is saved to {output_path}")



if __name__ == "__main__":

    # wnids_file_path = '/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/wnids.txt'
    # idxs_file_path = '../dataset/tiny-imagenet-200/idxs.txt'
    # words_file_path = '../dataset/tiny-imagenet-200/words.txt'
    # output_file_path = '../dataset/tiny-imagenet-200/tiny-imagenet-mapping.txt'
    # Load the mappings
    # mappings = load_mappings(wnids_file_path, idxs_file_path, words_file_path)

    tiny_imagenet_mapping_path = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/tiny-imagenet-200/tiny-imagenet-mapping.txt'
    val_annotations_path = '/scratch/zhao.lin1/dataset/tiny-imagenet-200/val/val_annotations.txt'
    output_file_path = '/home/zhao.lin1/DD-DDIM-Inversion/ds_inf/tiny-imagenet-200/val_mapped_annotations.txt'

    map_annotations(val_annotations_path, tiny_imagenet_mapping_path, output_file_path)

# ------------------------------------------------------------------------------------------
    # TEST CORRECTNESS
# ------------------------------------------------------------------------------------------
    
    # # Access the mappings
    # words_to_idxs = mappings['words_to_idxs']
    # wnids_to_words = mappings['wnids_to_words']
    # wnids_to_idxs = mappings['wnids_to_idxs']

    # # Save wnids_to_idxs to tiny-imagenet-mapping.txt
    # with open(output_file_path, 'w') as f:
    #     for wnid, index in wnids_to_idxs.items():
    #         f.write(f"{wnid}\t{index}\n")  # Write WNID and index separated by a tab

    # # Get the index of a specific WNID
    # target_wnid = 'n02124075'
    # if target_wnid in wnids_to_idxs:
    #     target_index = wnids_to_idxs[target_wnid]
    #     print(f"The index of WNID '{target_wnid}' is: {target_index}")
    # else:
    #     print(f"WNID '{target_wnid}' not found in the mapping.")


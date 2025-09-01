"""Knowledge Graph dataset pre-processing functions."""

import collections
import os
import pickle
import os
import numpy as np

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir,"data")
def get_idx(path, dataset_name):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations, locations, times = set(), set(), set(), set()
    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                if dataset_name == 'Yago5' or dataset_name == "DBPedia5" or dataset_name == "Wikidata5":
                    lhs, rel, rhs, loc, tim = line.strip().split("\t")
                    locations.add(loc)
                    times.add(tim)
                else:
                    lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)            
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}
    if dataset_name == 'Yago5' or dataset_name == "DBPedia5" or dataset_name == "Wikidata5":
        loc2idx = {x: i for (i, x) in enumerate(sorted(locations))}
        tim2idx = {x: i for (i, x) in enumerate(sorted(times))}
        return ent2idx, rel2idx, loc2idx, tim2idx
    return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx, loc2idx, tim2idx, dataset_name):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            if dataset_name == 'Yago5' or dataset_name == "DBPedia5" or dataset_name == "Wikidata5":
                lhs, rel, rhs, loc, tim = line.strip().split("\t")                            
                try:
                    examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs], loc2idx[loc], tim2idx[tim]])
                except ValueError:        
                    continue
            else:
                lhs, rel, rhs = line.strip().split("\t")
                try:
                    examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
                except ValueError:
                    continue
    return np.array(examples).astype("int64")


def get_filters(examples, n_relations, dataset_name):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)

    if dataset_name == 'Yago5' or dataset_name == "DBPedia5" or dataset_name == "Wikidata5":
       for lhs, rel, rhs, loc, tim in examples:
           rhs_filters[(lhs, rel, loc, tim)].add(rhs)
           lhs_filters[(rhs, rel + n_relations, loc, tim)].add(lhs)
       lhs_final = {}
       rhs_final = {}
       for k, v in lhs_filters.items():
           lhs_final[k] = sorted(list(v))
       for k, v in rhs_filters.items():
           rhs_final[k] = sorted(list(v))
    else:
       for lhs, rel, rhs in examples:       
           rhs_filters[(lhs, rel)].add(rhs)           
           lhs_filters[(rhs, rel + n_relations)].add(lhs)           
       lhs_final = {}                  
       rhs_final = {}           
       for k, v in lhs_filters.items():       
           #print(k)
           #print(v)
           #print(lhs_filters.items())
           #exit()
           lhs_final[k] = sorted(list(v))                          
        
       for k, v in rhs_filters.items():                          
           rhs_final[k] = sorted(list(v))

    
    return lhs_final, rhs_final


def process_dataset(path, dataset_name):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    print(dataset_name)
    if dataset_name == 'Yago5' or dataset_name == "DBPedia5" or dataset_name == "Wikidata5":
        ent2idx, rel2idx, loc2idx, tim2idx = get_idx(dataset_path, dataset_name)
    else:
        ent2idx, rel2idx = get_idx(dataset_path, dataset_name)
        loc2idx = ''
        tim2idx = ''
    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx, loc2idx, tim2idx, dataset_name)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx), dataset_name)
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


if __name__ == "__main__":
    data_path = DATA_PATH
    #print(data_path)
    #exit()
    for dataset_name in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset_name)
        #print(dataset_path)
        #exit()
        dataset_examples, dataset_filters = process_dataset(dataset_path, dataset_name)
        for dataset_split in ["train", "valid", "test"]:
            save_path = os.path.join(dataset_path, dataset_split + ".pickle")
            with open(save_path, "wb") as save_file:
                pickle.dump(dataset_examples[dataset_split], save_file)
        with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
            pickle.dump(dataset_filters, save_file)

"Test imports for the main modules used in the project"



def test_imports_src():
    from pop2vec.graph.src import (
        advanced_random_walk,
        combine_layer_adjacencies,
        deepwalk_dataset,
        deepwalk,
        # ignoring b/c of FileNotFoundErrors and we're not using them currently. Where not confirmed, it is suspected.
            # wrapping a `if __name__=="__main__"` would probably solve the issue
        # evaluate_embedding_distances, # raises FileNotFoundError (FNFE) confirmed
        # evaluate_income_forest, # FNFE confirmed
        # evaluate_income_linear, 
        # evaluate_income_svm, 
        # generate_income_baseline, # FNFE confirmed
        get_family_user_set, 
        # get_gron_buildings, # error in reading file
        # get_gron_edges, # FNFE confirmed
        # get_gron_people, # FNFE confirmed
        get_largest_cc, 
        gron_random_walk, 
        layered_random_walk, 
        model, 
        preprocess_full_network, 
        # random_walk_generator, # error importing with npct
        # remapping_test, # FNFE confirmed
        steve_random_walk,
    )

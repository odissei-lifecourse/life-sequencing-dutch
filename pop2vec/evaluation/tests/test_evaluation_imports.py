

def test_imports():
    from pop2vec.evaluation import (
        combine_gron_layer_adjacencies,
        combine_layer_adjacencies,
        # convert_data_to_sqlite, # need __main__ wrap
        convert_embeddings_to_hdf5,
        convert_pickle_embeddings,
        extract_embedding_subset,
        # generate_income_baseline, # need __main__ wrap
        generate_life_course_report,
        # get_death_dates, # probably need __main__ wrap
        get_hops_from_ground_truth,
        # get_job_changes, # need __main__ wrap
        # isolate_income_subset, # need __main__ wrap 
        # isolate_marriage_subset # need __main__ wrap
        nearest_neighbor,
        prepare_full_adjacency,
        # prepare_marriage_data, # need __main__ wrap 
        # prepare_naive_baseline need __main__ wrap
        report_tests,
        report_utils,
        write_embedding_metadata
    )



import pickle 
import argparse 

network_emb_root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/"
llm_emb_root_old = "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/"
llm_emb_root = "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/embeddings/"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Embedding Metadata")

    parser.add_argument(
        "--collection_name",
        default="standard_embedding_set",
        type=str,
        help="Savename for this collection of embeddings"
    )

    args = parser.parse_args()

    save_url = "/gpfs/ostor/ossc9424/hoemdir/Life_Course_Evaluation/embedding_meta/" + args.collection_name + ".pkl"

    if args.collection_name == "network_embedding_set":
        
        embedding_sets = [


            # {"name": "Full network 2010",
            #  "year": "2010",
            #  "type": "NET",
            #  "root": llm_emb_root_old,
            #  "mapping": "mappings/family_2010.pkl",
            #  "url": "embeddings/lr_steve_full_network_2010_30.emb",
            #  "truth": "full"},

            {"name": "Full network 2020",
             "year": "2020",
             "type": "NET",
             "root": network_emb_root,
             "mapping": "mappings/family_2020.pkl",
             "url": "embeddings/lr_steve_full_network_2020.emb",
             "truth": "full"}
        ]
        
    # omitted gron
    elif args.collection_name == "llm_embedding_set":

        embedding_sets = [

            # {"name": "LLM Mean 2016 d64",
            #  "year": "2016",
            #  "type": "LLM",
            #  "root": llm_emb_root,
            #  "url": "mean_embedding_2017_64.h5",
            #  "truth": "full"
            #  },

            # {"name": "LLM CLS 2017 small",
            #  "year": "2016",
            #  "type": "LLM",
            #  "root": llm_emb_root,
            # # "url": 2017_small.h5,
            #  "url": "2017_small_v0.0.1.h5",
            #  "emb_type": "cls_emb",
            #  "truth": "full"
            #  },

            {"name": "LLM CLS 2017 medium",
             "year": "2016",
             "type": "LLM",
             "root": llm_emb_root,
             "url": "2017_medium.h5",
             #"url": "2017_medium_v0.0.1.h5",
             "emb_type": "cls_emb",
             "truth": "full"
             },


            {"name": "LLM CLS 2017 medium2x",
             "year": "2016",
             "type": "LLM",
             "root": llm_emb_root,
             "url": "2017_medium2x.h5",
             #"url": "2017_medium2x_v0.0.1.h5",
             "emb_type": "cls_emb",
             "truth": "full"
             },

            {"name": "LLM CLS 2017 large",
             "year": "2016",
             "type": "LLM",
             "root": llm_emb_root,
             "url": "2017_large.h5",
             #"url": "2017_large_v0.0.1.h5",
             "emb_type": "cls_emb",
             "truth": "full"
             }

        ]

    with open(save_url, "wb") as pkl_file:
        pickle.dump(embedding_sets, pkl_file)


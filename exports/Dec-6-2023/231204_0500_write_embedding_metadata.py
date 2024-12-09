import pickle
import argparse

network_emb_root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/"
llm_emb_root = "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Embedding Metadata')

    parser.add_argument(
        "--collection_name",
        default='standard_embedding_set',
        type=str,
        help='Savename for this collection of embeddings'
    )

    args = parser.parse_args()

    save_url = '/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/embedding_meta/' + args.collection_name + '.pkl'
    
######################################################################################################################################################################################
    
    if args.collection_name == 'network_embedding_set':
    
        embedding_sets = [
        
        #{'name': 'Full Network 2010',
        # 'year': '2010',
        # 'type': 'NET',
        # 'root': network_emb_root,
        # 'mapping': 'mappings/family_2010.pkl',
        # 'url': 'embeddings/lr_steve_full_network_2010_30.emb',
        # 'truth': 'full'
        # }

        {'name': 'Full Network 2020',
         'year': '2020',
         'type': 'NET',
         'root': network_emb_root,
         'mapping': 'mappings/family_2020.pkl',
         'url': 'embeddings/lr_steve_full_network_2020.emb',
         'truth': 'full'
         }

        #{'name': 'Family 2010',
        # 'year': '2010',
        # 'type': 'NET',
        # 'root': network_emb_root,
        # 'mapping': 'mappings/family_2010.pkl',
        # 'url': 'embeddings/family_2010_40.emb',
        # 'truth': 'full'
        #}
        ]
        
###########################################################################################################################################################################################

    elif args.collection_name == 'llm_embedding_set':
    
        embedding_sets = [
            
            {'name': 'LLM Mean 2016',
             'year': '2016',
             'type': 'LLM',
             'root': llm_emb_root,
             'url': 'mean_embedding_2016.json',
             'truth': 'full'
             },
             
            {'name': 'LLM CLS 2016',
             'year': '2016',
             'type': 'LLM',
             'root': llm_emb_root,
             'url': 'cls_embedding_2016.json',
             'truth': 'full'
             }
        ]
        
#####################################################################################################################################################################################################
        
    elif args.collection_name == 'gron_embedding_set':
    
        years = ['2010', '2012', '2014']
        #years = ['2016', '2018', '2020']
        
        embedding_sets = []
        
        for year in years:
            gron_dict = {'name': 'Gr ' + year, 'year': year, 'type': 'NET', 'root': network_emb_root, 'mapping': 'mappings/gron_' + year + '_mappings.pkl', 'url': 'embeddings/gron_full_network_' + year + '.emb', 'truth': 'gron'}
            embedding_sets.append(gron_dict)

###############################################################################################################################################################################3

    elif args.collection_name == 'gron_layers':
    
        layers = ['family', 'household', 'classmate', 'neighbor', 'colleague']

        embedding_sets = []

        for layer in layers:
            gron_dict = {'name': 'Gr ' + layer[:3], 'year': '2016', 'type': 'NET', 'root': network_emb_root, 'mapping': 'mappings/gron_2016_mappings.pkl', 'url': 'embeddings/gron_' + layer + '_priority_2016.emb', 'truth': 'gron'}
            embedding_sets.append(gron_dict)

    with open(save_url, 'wb') as pkl_file:
        pickle.dump(embedding_sets, pkl_file)
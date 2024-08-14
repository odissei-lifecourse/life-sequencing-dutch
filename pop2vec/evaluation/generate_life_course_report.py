import argparse
import numpy as np
import random
from fpdf import FPDF
from sentence_transformers import util
from torch import Tensor
import time
from datetime import date
import report_utils
import pickle
import copy
import logging
from prediction_models.models import model_dict 

embedding_map = { # keys are tasks, values are names of embeddings
    "income": ["income_eval"],
    "marriage": ["marriage_eval"],
    "marriage_rank": ["marriage_rank"]
}

BASELINE_TASK = "income" # determines which embeddings are used for summary stats

if __name__ == '__main__':

    # Report Sections
    # 0 = Embedding Similarity Comparisons
    # 1 = Hop Distribution Images
    # 2 = Binary Distribution Images
    # 3 = Prediction - Income at age 30
    # 4 = Prediction - Job Changes
    # 5 = Prediction - Marriage
    # 6 = Prediction - Partnership
    # 7 = Prediction - Highest Education Level
    # 8 = Prediction - Death

    income_baseline_year = 2016
    report_parts = [3, 5.1]
    regen_images = True
    is_eval = True 
    years = list(range(2011, 2022))

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "--collection_name",
        default='standard_embedding_set',
        type=str,
        help='Savename for this collection of embeddings'
    )

    parser.add_argument(
        "--savename",
        default='results/NL_test_1_',
        type=str,
    )

    parser.add_argument(
        "--sample",
        default=-1,
        type=int,
        help="If positive, use embeddings of the first `sample` records in the hdf5 file."
    )

    parser.add_argument(
        "--embedding_size",
        default=-1,
        type=int,
        help="If positive, restrict embedding dimension to this size."
    )
    parser.add_argument(
        "--pred_model",
        type=str,
        default="linear_regression",
        help="Model type to use for prediction"
    )

    args = parser.parse_args()

    
    single_model = model_dict[args.pred_model]["single"]
    pair_model = model_dict[args.pred_model]["pair"]

    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG
    )

    # Load embedding set written by write_embedding_metadata.py
    load_url = 'embedding_meta/' + args.collection_name + '.pkl'
    with open(load_url, 'rb') as pkl_file:
        embedding_sets = list(pickle.load(pkl_file))

    baseline_dict = report_utils.precompute_global(
        'background', years, 
        income_baseline_year=income_baseline_year,
        is_eval=is_eval)
    income_by_year = report_utils.precompute_global('income', years, is_eval=is_eval)
    marriages_by_year = report_utils.precompute_global('marriage', years, is_eval=is_eval)



    logging.info("Beginning report generation for embedding set: %s", args.collection_name)

    full_start = time.time()

    distribution_savenames = []
    binary_savenames = []

    # Results for the predictions tasks, indexed by embedding name
    income_results = {}
    income_test_counts_by_year = {}

    marriage_results = {}
    marriage_test_counts_by_year = {}

    partnership_results = {}
    partnership_test_counts_by_year = {}

    marriage_rank_results = {}
    marriage_rank_test_counts_by_year = {}

    partnership_rank_results = {}
    partnership_rank_test_counts_by_year = {}

    death_results = {}

    # Used for the first comparison table in section 0.
    entries = []
    table_rows = []

    # Used for the summary table in section 0.
    summary_dict = {}

    for i, emb in enumerate(embedding_sets):

        section_start = time.time()

        embeddings_per_task = {}
        for task, query_keys in embedding_map.items():
            logging.debug("loading embeddings for %s" % task)
            embedding_dict = report_utils.precompute_local(
                emb,
                nested_query_keys=query_keys,
                only_embedding=True,
                sample_size=args.sample,
                embedding_size=args.embedding_size
            )
            embeddings_per_task[task] = embedding_dict

        # embedding_dict = report_utils.precompute_local(emb, only_embedding=True)
        distance_matrix = {}
        current_embedding_dict = embeddings_per_task[BASELINE_TASK]

        root = emb['root']
        url = emb['url']
        year = emb['year']
        truth = emb['truth']
        name = emb['name']

        summary_dict[name] = {}
        summary_dict[name]['type'] = emb['type']
        summary_dict[name]['year'] = emb['year']

        # Get the relevant prediction years from the embedding metadata
        target_year = int(emb['year']) + 1
        relevant_years = years[years.index(target_year):]

        summary_dict[name]['n_samples'] = str(len(current_embedding_dict))
        random_person = random.choice(list(current_embedding_dict.keys()))
        summary_dict[name]['dimensions'] = str(len(current_embedding_dict[random_person]))

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Loaded embeddings: %s over %s minutes", name, np.round(delta / 60.0, 2))

        savename = ""
        for chunk in name.split(" "):
            savename += chunk + "_"
        savename = savename[:-1]

        # 0. Compare Embeddings and print results to first section
        if 0 in report_parts:

            section_start = time.time()

            for j in range(i + 1, len(embedding_sets)):
                emb_2 = embedding_sets[j]
                embedding_dict_2 = report_utils.precompute_local(
                    emb_2, only_embedding=True, sample_size=args.sample, embedding_size=args.embedding_size)
                name_2 = emb_2['name']

                results_10 = report_utils.embedding_rank_comparison(embedding_dict, embedding_dict_2, top_k=10,
                                                                    methods=['intersection', 'spearman'])
                results_100 = report_utils.embedding_rank_comparison(embedding_dict, embedding_dict_2, top_k=100,
                                                                     methods=['intersection', 'spearman'])

                table_row = [name, name_2,
                             str(np.round(results_10['intersection'], 3)),
                             str(np.round(results_100['intersection'], 3)),
                             str(np.round(results_10['spearman'], 3)),
                             str(np.round(results_100['spearman'], 3))]
                table_rows.append(table_row)

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Computed Embedding Comparisons for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 1. Plot hop distance distributions
        #####################################################################################################
        if 1 in report_parts:

            section_start = time.time()

            plot_savename = '/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/results/' + savename + "_hop_distributions.png"
            distribution_savenames.append(plot_savename)

            if regen_images:
                subtitle = name + " Embeddings"
                plot_title = truth.capitalize() + " Network Distance Distributions: \n" + subtitle
                report_utils.plot_embedding_distances(embedding_dict, hops_network, distance_matrix, 1, plot_title,
                                                      plot_savename, show=True)

                section_end = time.time()
                delta = section_end - section_start
                logging.info("Plotted Hop Distances for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 2. Plot Real vs. Fake distance distributions
        #####################################################################################################
        if 2 in report_parts:

            section_start = time.time()

            plot_savename = '/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/results/' + savename + "_real_vs_random.png"
            binary_savenames.append(plot_savename)

            if regen_images:
                subtitle = name + " Embeddings"
                plot_title = truth.capitalize() + " Network Real vs Random Distances: \n" + subtitle

                report_utils.plot_distance_vs_ground_truth(embedding_dict, ground_truth_dict, distance_matrix, 1,
                                                           plot_title, plot_savename, show=True)

                section_end = time.time()
                delta = section_end - section_start
                logging.info("Plotted Real vs. Random Distances for: %s over %s minutes", name,
                             np.round(delta / 60.0, 2))

        # 3 - Generate Income Prediction Results
        #######################################################################################################################
        if 3 in report_parts:

            section_start = time.time()
            embedding_dict = embeddings_per_task["income"]

            # Predict using just embeddings
            #result_dict, test_counts_by_year = report_utils.linear_variable_prediction(
            #    embedding_dict, income_by_year, relevant_years, dtype='single')
            result_dict, test_counts_by_year = report_utils.one_model_per_year_variable_prediction(
                single_model, embedding_dict, income_by_year, relevant_years, dtype='single')

            # Predict using embeddings + baseline
            #result_with_baseline, test_counts_with_baseline, only_baseline = report_utils.linear_variable_prediction(
            #    embedding_dict, income_by_year, relevant_years, dtype='single', baseline=baseline_dict)
            result_with_baseline, test_counts_with_baseline, only_baseline = report_utils.one_model_per_year_variable_prediction(
                single_model, embedding_dict, income_by_year, relevant_years, dtype='single', baseline=baseline_dict)

            if result_with_baseline is not None:

                income_results["Baseline"] = only_baseline

                income_results[name] = result_dict
                income_test_counts_by_year[name] = test_counts_by_year

                income_results[name + "\n+ Base"] = result_with_baseline
                income_test_counts_by_year[name + "\n+ Base"] = test_counts_with_baseline

            else:
                income_results[name] = result_dict
                income_test_counts_by_year[name] = test_counts_by_year

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Income Results for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 4 - Generate Job Change Prediction Results
        #######################################################################################################################
        if 4 in report_parts:
            section_start = time.time()

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Job Change Results for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 5.1 - Generate Marriage Pair Prediction Results
        ##################################################################################################################################################################################################
        if 5.1 in report_parts:

            section_start = time.time()
            embedding_dict = embeddings_per_task["marriage"]

            # Predict using just embeddings
            result_dict, test_counts_by_year = report_utils.linear_variable_prediction(
               embedding_dict, marriages_by_year, relevant_years, dtype='pair')
            # result_dict, test_counts_by_year = report_utils.one_model_per_year_variable_prediction(
            #     pair_model, embedding_dict, marriages_by_year, relevant_years, dtype='pair')

            # Predict using embeddings + baseline
            result_with_baseline, test_counts_with_baseline, only_baseline = report_utils.linear_variable_prediction(
               embedding_dict, marriages_by_year, relevant_years, dtype='pair', baseline=baseline_dict)
            # result_with_baseline, test_counts_with_baseline, only_baseline = report_utils.one_model_per_year_variable_prediction(
            #     pair_model, embedding_dict, marriages_by_year, relevant_years, dtype='pair', baseline=baseline_dict)

            if result_with_baseline is not None:

                marriage_results['Baseline'] = only_baseline

                marriage_results[name] = result_dict
                marriage_test_counts_by_year[name] = test_counts_by_year

                marriage_results[name + '\n+ Base'] = result_with_baseline
                marriage_test_counts_by_year[name + '\n+ Base'] = test_counts_with_baseline

            else:

                marriage_results[name] = result_dict
                marriage_test_counts_by_year[name] = test_counts_by_year

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Binary Marriage Results for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 5.2 - Generate Marriage Partner Rank Prediction Results
        ##################################################################################################################################################################################################
        if 5.2 in report_parts:
            section_start = time.time()
            embedding_dict = embeddings_per_task["marriage_rank"]

            marriage_ranks_by_year, test_counts_by_year = report_utils.get_marriage_rank_by_year(
                embedding_dict, distance_matrix, dtype='marriage')

            marriage_rank_results[name] = marriage_ranks_by_year
            marriage_rank_test_counts_by_year[name] = test_counts_by_year

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Marriage Rank Results for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 6.1 - Generate Partnership Pair Predictions
        ##################################################################################################################################################################################################
        if 6.1 in report_parts:

            section_start = time.time()

            #result_dict, test_counts_by_year = report_utils.linear_variable_prediction(
            #    embedding_dict, partnerships_by_year,years, dtype='pair')
            result_dict, test_counts_by_year = report_utils.one_model_per_year_variable_prediction(
                pair_model, embedding_dict, partnerships_by_year, relevant_years, dtype='pair')

            #result_with_baseline, test_counts_with_baseline, only_baseline = report_utils.linear_variable_prediction(
            #    embedding_dict, partnerships_by_year, years, dtype='pair', baseline=baseline_dict)
            result_with_baseline, test_counts_with_baseline, only_baseline = report_utils.one_model_per_year_variable_prediction(
                pair_model, embedding_dict, partnerships_by_year, relevant_years, dtype='pair', baseline=baseline_dict)


            if result_with_baseline is not None:

                partnership_results['Baseline'] = only_baseline

                partnership_results[name] = result_dict
                partnership_test_counts_by_year[name] = test_counts_by_year

                partnership_results[name + '\n+ Base'] = result_with_baseline
                partnership_test_counts_by_year[name + '\n+ Base'] = test_counts_with_baseline

            else:
                partnership_results[name] = result_dict
                partnership_test_counts_by_year[name] = test_counts_by_year

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Binary Partnership Results for: %s over %s minutes", name,
                         np.round(delta / 60.0, 2))

        # 6.2 - Generate Partnership Partner Rank Predictions
        ##################################################################################################################################################################################################
        if 6.2 in report_parts:
            section_start = time.time()

            partnership_ranks_by_year, test_counts_by_year = report_utils.get_marriage_rank_by_year(embedding_dict,
                                                                                                    distance_matrix,
                                                                                                    dtype='partnership')
            partnership_rank_results[name] = partnership_ranks_by_year
            partnership_rank_test_counts_by_year[name] = test_counts_by_year

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Partnership Rank Results for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 7 - Generate Highest Education Level Results
        ##########################################################################################################################################################################################
        if 7 in report_parts:
            section_start = time.time()

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Divorce Results for: %s over %s minutes", name, np.round(delta / 60.0, 2))

        # 8 - Generate Death Predictions
        ######################################################################################################################################################################################
        if 8 in report_parts:
            section_start = time.time()

            result_dict = report_utils.yearly_probability_prediction(embedding_dict, deaths_by_year,
                                                                     list(deaths_by_year.keys()))
            death_results[name] = result_dict

            section_end = time.time()
            delta = section_end - section_start
            logging.info("Generated Death Results for: %s over %s minutes", name, np.round(delta / 60.0, 2))

    ###########################################################################################################################################################################################
    # 4. Generate the report
    # Prep pdf hyperparemeters
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    date_str = str(date.today())

    x_offset = 10
    y_offset = 10

    pdf.set_font('Arial', '', 10)

    plot_height = 100
    header_height = 20

    # A4 dimensions in mm
    max_width = 210
    max_height = 297

    ##############################################################################################################################################################

    section_start = time.time()

    # Print the information to the report
    pdf.add_page()
    y_offset = 10

    pdf.set_font_size(18)
    pdf.set_font('', style='B')
    pdf.cell(w=0, h=header_height, txt="0. Embedding Descriptions/Comparisons")
    pdf.set_font_size(8)
    pdf.set_font('', style='')
    y_offset += header_height
    pdf.ln(header_height)

    # 1. Reformat data into rows
    data = []

    metrics = ['type', 'year', 'n_samples', 'dimensions']

    for i, metric in enumerate(metrics):

        row = [metric]

        for j, emb_type in enumerate(summary_dict):
            value = summary_dict[emb_type][metric]
            row.append(value)

        data.append(row)

    # 2. Print the table
    line_height = pdf.font_size * 2.5
    epw = max_width - 20
    col_width = epw / (len(summary_dict) + 1)

    header_row = [" "] + list(summary_dict.keys())

    # Print header row
    pdf.set_font('', style='BI')
    for datum in header_row:
        pdf.cell(w=col_width, h=line_height, txt=datum, border=1, align='C')
    pdf.ln(line_height)

    # Print other rows
    for i, row in enumerate(data):
        # print(row, flush=True)

        for j, datum in enumerate(row):
            if j == 0:
                pdf.set_font('', style='BI')
            else:
                pdf.set_font('', style='')
            pdf.cell(w=col_width, h=line_height, txt=str(datum), border=1, align='C')

        pdf.ln(line_height)

    section_end = time.time()
    delta = section_end - section_start
    logging.info("Wrote summary in %s seconds", delta)

    # Only print the table if we have 2 or more embeddings to compare
    if len(embedding_sets) > 1 and 0 in report_parts:

        section_start = time.time()
        pdf.add_page()

        header_row = ['Emb 1', 'Emb 2', 'Top 10 Intersection', 'Top 100 Intersection', 'Top 10 Spearman',
                      'Top 100 Spearman']
        col_width = epw / len(header_row)

        # Print header row
        pdf.set_font('', style='BI')
        for datum in header_row:
            pdf.cell(w=col_width, h=line_height, txt=datum, border=1, align='C')
        pdf.ln(line_height)

        # Print other rows
        for i, row in enumerate(table_rows):

            # print(row, flush=True)
            pdf.set_font('', style='')
            for j, datum in enumerate(row):
                pdf.cell(w=col_width, h=line_height, txt=datum, border=1, align='C')
                pass

            pdf.ln(line_height)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 0 in %s seconds", np.round(delta, 2))

    #########################################################################################################################################

    if 1 in report_parts:

        section_start = time.time()
        # Paste plots
        pdf.add_page()
        y_offset = 10

        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="1. Distance Distributions by Hop")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height

        for plot_savename in distribution_savenames:

            if y_offset + plot_height > max_height:
                y_offset = 10
                pdf.add_page()

            pdf.image(plot_savename, x=x_offset, y=y_offset, h=plot_height)
            y_offset += plot_height

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 1 in %s seconds", np.round(delta, 2))

    #########################################################################################################################################

    if 2 in report_parts:

        section_start = time.time()

        # Add new page in between plot sections
        pdf.add_page()
        y_offset = 10

        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="2. Distance Distributions: Real vs Fake Connections")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height

        for plot_savename in binary_savenames:

            if y_offset + plot_height > max_height:
                y_offset = 10
                pdf.add_page()

            pdf.image(plot_savename, x=x_offset, y=y_offset, h=plot_height)
            y_offset += plot_height

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 2 in %s seconds", np.round(delta, 2))

    ########################################################################################################################################

    if 3 in report_parts:
        section_start = time.time()

        pdf.add_page()
        y_offset = 10

        # Generate and add tables
        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="3. Prediction - Income at Age 30 (R^2)")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height
        pdf.ln(header_height)

        pdf = report_utils.print_output_table(pdf, relevant_years, income_results, highlight=True)
        pdf.cell(w=0, h=header_height, txt="Test Counts")
        pdf.ln(header_height)
        pdf = report_utils.print_output_table(pdf, relevant_years, income_test_counts_by_year, highlight=False)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 3 in %s seconds", np.round(delta, 2))

    ###########################################################################################################################################################################

    if 4 in report_parts:
        section_start = time.time()

        pdf.add_page()
        y_offset = 10

        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="4. Prediction - Job Changes (R^2)")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height
        pdf.ln(header_height)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 4 in %s seconds", np.round(delta, 2))

    ##########################################################################################################################################

    if 5.1 in report_parts:
        section_start = time.time()

        pdf.add_page()
        y_offset = 10

        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="5. Prediction - Marriage (Mean Accuracy)")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height
        pdf.ln(header_height)

        # Subheader
        pdf.set_font_size(14)
        pdf.cell(w=0, h=header_height, txt='5.1 Binary Prediction (Real vs. Fake Pairs)')
        pdf.set_font_size(8)
        y_offset += header_height
        pdf.ln(header_height)

        pdf = report_utils.print_output_table(pdf, relevant_years, marriage_results, highlight=True)
        pdf.cell(w=0, h=header_height, txt="Test Counts")
        pdf.ln(header_height)
        pdf = report_utils.print_output_table(pdf, relevant_years, marriage_test_counts_by_year, highlight=False)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 5.1 in %s seconds", np.round(delta, 2))

        #################################################################################################################

    if 5.2 in report_parts:
        section_start = time.time()

        pdf.add_page()
        y_offset = 10

        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="5. Prediction - Marriage Partner Rank")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height
        pdf.ln(header_height)

        # Subheader
        pdf.set_font_size(14)
        pdf.cell(w=0, h=header_height, txt='5.2 Actual Partner Rank vs. Random Pairings')
        pdf.set_font_size(8)
        y_offset += header_height
        pdf.ln(header_height)

        pdf = report_utils.print_output_table(pdf, relevant_years, marriage_rank_results, highlight=True, reverse=True)
        pdf.cell(w=0, h=header_height, txt="Test Counts")
        pdf.ln(header_height)
        pdf = report_utils.print_output_table(pdf, relevant_years, marriage_rank_test_counts_by_year, highlight=False)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 5.2 in %s seconds", np.round(delta, 2))

    ######################################################################################################################

    if 6.1 in report_parts:
        section_start = time.time()

        pdf.add_page()
        y_offset = 10

        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="6. Prediction - Partnerships (Mean Accuracy)")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height
        pdf.ln(header_height)

        # Subheader
        pdf.set_font_size(14)
        pdf.cell(w=0, h=header_height, txt='6.1 Binary Prediction (Real vs. Fake Pairs)')
        pdf.set_font_size(8)
        y_offset += header_height
        pdf.ln(header_height)

        pdf = report_utils.print_output_table(pdf, relevant_years, partnership_results, highlight=True)
        pdf.cell(w=0, h=header_height, txt="Test Counts")
        pdf.ln(header_height)
        pdf = report_utils.print_output_table(pdf, relevant_years, partnership_test_counts_by_year, highlight=False)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 6.1 in %s seconds", np.round(delta, 2))

    #################################################################################################################

    if 6.2 in report_parts:
        section_start = time.time()

        pdf.add_page()
        y_offset = 10

        pdf.set_font_size(18)
        pdf.set_font('', style='B')
        pdf.cell(w=0, h=header_height, txt="6. Prediction - Partnership Partner Rank")
        pdf.set_font_size(8)
        pdf.set_font('', style='')
        y_offset += header_height
        pdf.ln(header_height)

        # Subheader
        pdf.set_font_size(14)
        pdf.cell(w=0, h=header_height, txt='6.2 Actual Partner Rank vs. Random Pairings')
        pdf.set_font_size(8)
        y_offset += header_height
        pdf.ln(header_height)

        pdf = report_utils.print_output_table(pdf, relevant_years, partnership_rank_results, highlight=True, reverse=True)
        pdf.cell(w=0, h=header_height, txt="Test Counts")
        pdf.ln(header_height)
        pdf = report_utils.print_output_table(pdf, relevant_years, partnership_rank_test_counts_by_year, highlight=False)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 6.2 in %s seconds", np.round(delta, 2))

    ######################################################################################################################

    if 7 in report_parts:
        section_start = time.time()

        pdf.add_page()
        pdf.set_font_size(18)
        pdf.cell(w=0, h=header_height, txt="7. Prediction - Highest Education Level (R^2)")
        pdf.set_font_size(8)
        y_offset += header_height
        pdf.ln(header_height)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 7 in %s seconds", np.round(delta, 2))

    ######################################################################################################################

    if 8 in report_parts:
        section_start = time.time()

        pdf.add_page()
        pdf.set_font_size(18)
        pdf.cell(w=0, h=header_height, txt="8. Prediction - Death (R^2)")
        pdf.set_font_size(8)
        y_offset += header_height
        pdf.ln(header_height)

        pdf = report_utils.print_output_table(pdf, relevant_years, death_results, highlight=True)

        section_end = time.time()
        delta = section_end - section_start
        logging.info("Wrote section 8 in %s seconds", np.round(delta, 2))

    ######################################################################################################################

    pdf.output(args.savename + date_str + '.pdf', 'F')

    full_end = time.time()
    delta = full_end - full_start
    logging.info("Generated report over: %s hours", np.round(delta / 60. / 60., 2))
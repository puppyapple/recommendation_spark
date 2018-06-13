# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import reduce
from itertools import product
from Code import data_generator, data_calculator, comp_property
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

path_comp_tags_all = "../Data/Output/recommendation/comp_tags_all.pkl"
path_ctag_ctag = "../Data/Output/recommendation/ctag_ctag.pkl"
path_ctag_nctag = "../Data/Output/recommendation/ctag_nctag.pkl"
path_nctag_nctag = "../Data/Output/recommendation/nctag_nctag.pkl"
path_concept_tree_property = "../Data/Output/recommendation/concept_tree_property.pkl"
path_ctag_position = "../Data/Output/recommendation/ctag_position.pkl"
path_comp_id_name_dict = "../Data/Output/recommendation/comp_id_name_dict.pkl"

def all_inputs_generator(new_result="company_tag_info_latest", old_result="company_tag"):
    comp_ctag_table, comp_nctag_table, comp_ctag_table_all_infos = data_generator.comp_tag(new_result=new_result, old_result=old_result, db=db)
    print("---Raw data preparation finished---")
    ctag_comps_aggregated, nctag_comps_aggregated, comp_total_num = data_generator.data_aggregator(comp_ctag_table, comp_nctag_table, recalculate=True)
    print("---Data aggregation finished---")
    nctag_idf = data_calculator.nctag_idf(nctag_comps_aggregated, comp_total_num)
    print("---Fake idf of nctag calculation finished---")
    data_calculator.ctag_relation(ctag_comps_aggregated)
    print("---Ctag-ctag relation calculation finished---")
    data_calculator.ctag_nctag_relation(ctag_comps_aggregated, nctag_comps_aggregated, nctag_idf)
    print("---Ctag-nctag relation calculation finished---")
    data_calculator.nctag_nctag(nctag_comps_aggregated, nctag_idf)
    print("---Ntag-nctag relation calculation finished---")
    comp_property.concept_tree_property(comp_ctag_table_all_infos)
    print("---Ctag tree position information preparation finished---")
    return 0


def data_loader():
    comp_tags_all = pickle.load(open(path_comp_tags_all, "rb"))
    ctag_ctag = pickle.load(open(path_ctag_ctag, "rb"))
    ctag_nctag = pickle.load(open(path_ctag_nctag, "rb"))
    nctag_nctag = pickle.load(open(path_nctag_nctag, "rb"))
    concept_tree_property = pickle.load(open(path_concept_tree_property, "rb"))
    ctag_position = pickle.load(open(path_ctag_position, "rb"))
    comp_id_name_dict = pickle.load(open(path_comp_id_name_dict, "rb"))
    
    comp_tags_all_df = pd.DataFrame(list(comp_tags_all.items()))
    # comp_tags_all_df.columns = ["comp_id", "tags_infos_dict"]   
    concept_tree_property_df = pd.DataFrame(list(concept_tree_property.items()))
    # concept_tree_property_df.columns = ["comp_id", "concept_tree_property"]
    comp_infos = pd.concat([comp_tags_all_df, concept_tree_property_df]).groupby(0).agg(lambda x: reduce(lambda a, b: {**a, **b} ,x)).reset_index()
    comp_infos.columns = ["comp_id", "comp_property_dict"]
    print("Data loaded!")
    return (comp_infos, ctag_ctag, ctag_nctag, nctag_nctag, ctag_position, comp_id_name_dict)


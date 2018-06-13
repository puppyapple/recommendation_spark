#%%
import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
import configparser
from py2neo import Graph
from functools import reduce
from itertools import product
from Code import data_generator, data_calculator, comp_property
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pyspark import SparkContext 
from pyspark.sql import SQLContext 

#%%
path_comp_tags_all = "../Data/Output/recommendation/comp_tags_all.pkl"
path_ctag_ctag = "../Data/Output/recommendation/ctag_ctag.pkl"
path_ctag_nctag = "../Data/Output/recommendation/ctag_nctag.pkl"
path_nctag_nctag = "../Data/Output/recommendation/nctag_nctag.pkl"
path_concept_tree_property = "../Data/Output/recommendation/concept_tree_property.pkl"
path_ctag_position = "../Data/Output/recommendation/ctag_position.pkl"
path_comp_id_name_dict = "../Data/Output/recommendation/comp_id_name_dict.pkl"

config = configparser.ConfigParser()
config.read("../Data/Input/database_config/database.conf")
host = config['NEO4J']['host']
user_name = config['NEO4J']['username']
pass_word = config['NEO4J']['password']
graph = Graph(
    host,
    username=user_name,
    password=pass_word
)

def final_count(l1, l2):
    if len(l1.union(l2)) == 0:
        return 0
    else:
        return len(l1.intersection(l2))/len(l1.union(l2))

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


def cal_tag_cartesian(tag_set1, tag_set2, value_dict, tag_link_filter):
    if tag_set1 == 0 or tag_set2 == 0:
        return 0
    else:
        pair_list = list(product(tag_set1, tag_set2))
        value_list = [value_dict.get(t[0] + "-" + t[1], 0) for t in pair_list]
        value_sum = sum([v for v in value_list if v >= tag_link_filter])
        # print(value_sum)
        return value_sum
    
def cal_tags_link(comp_info1, comp_info2, tag_link_filters):
    ctags1 = comp_info1.get("ctags", set())
    ctags2 = comp_info2.get("ctags", set())
    nctags1 = comp_info1.get("nctags", set())
    nctags2 = comp_info2.get("nctags", set())
    num_ctags1 = len(ctags1)
    num_ctags2 = len(ctags2)
    num_nctags1 = len(nctags1)
    num_nctags2 = len(nctags2)
    num_all1 = num_ctags1 + num_nctags1
    num_all2 = num_ctags2 + num_nctags2
    
    coef1 = 1/np.sqrt(1 + (num_ctags1 - num_ctags2)**2)
    coef2 = 1/np.sqrt(1 + (num_nctags1 - num_nctags2)**2)
    coef3 = final_count(ctags1, ctags2)
    coef4 = final_count(nctags1, nctags2)
    
    v1 = coef3 * cal_tag_cartesian(ctags1, ctags2, ctag_ctag, tag_link_filters[0])
    v2 = cal_tag_cartesian(nctags1, ctags2, ctag_nctag, tag_link_filters[1]) + cal_tag_cartesian(nctags2, ctags1, ctag_nctag, tag_link_filters[1])
    v3 = coef4 * cal_tag_cartesian(nctags1, nctags2, nctag_nctag, tag_link_filters[2])
    return (v1, v2, v3)

def cal_tags_link_with_target(target_comp_info, ctag_list, nctag_list, tag_link_filters):
    ctags1 = target_comp_info.get("ctags", [])
    nctags1 = target_comp_info.get("nctags", [])

    num_ctags1 = len(ctags1)
    num_ctags2 = len(ctag_list)
    num_nctags1 = len(nctags1)
    num_nctags2 = len(nctag_list)
    num_all1 = num_ctags1 + num_nctags1
    num_all2 = num_ctags2 + num_nctags2
    
    coef1 = 1/np.sqrt(1 + (num_ctags1 - num_ctags2)**2)
    coef2 = 1/np.sqrt(1 + (num_nctags1 - num_nctags2)**2)
    coef3 = final_count(ctags1, ctag_list)
    coef4 = final_count(nctags1, nctag_list)
    
    v1 = coef3 * cal_tag_cartesian(ctags1, ctag_list, ctag_ctag, tag_link_filters[0])
    v2 = cal_tag_cartesian(nctags1, ctag_list, ctag_nctag, tag_link_filters[1]) + cal_tag_cartesian(nctag_list, ctags1, ctag_nctag, tag_link_filters[1])
    v3 = coef4 * cal_tag_cartesian(nctags1, nctag_list, nctag_nctag, tag_link_filters[2])
    return (v1, v2, v3)

def cal_company_dis(target_comp_info, part,  weights, tag_link_filters):
    # print("start")
    three_value_list = list(part.comp_property_dict.apply(lambda x: cal_tags_link(target_comp_info, x, tag_link_filters)))
    part["three_values"] = three_value_list
    # print("end")
    return part

def concept_tree_relation(comp_info1, comp_info2):
    top_tag1 = comp_info1.get("top_ctag", set())
    top_tag2 = comp_info2.get("top_ctag", set())
    bottom_tag1 = comp_info1.get("bottom_ctag", set())
    bottom_tag2 = comp_info2.get("bottom_ctag", set())
    is_same_tree = len(top_tag1.intersection(top_tag2)) > 0
    bottom_tag_relation = np.array([ctag_position.get(t[0] + "-" + t[1], -1) for t in list(product(bottom_tag1, bottom_tag2))])
    is_same_link = sum(bottom_tag_relation >= 0) > 0
    return (is_same_tree, is_same_link)
    
def branch_stock_relation(comp_id, graph):
    stock_rel_statement = "match p=(c:company{id:'%s'})-[:ABSOLUTE_HOLDING|:UNKNOWN|:WHOLLY_OWNED|:JOINT_STOCK|:RELATIVE_HOLDING*1..2]-(c2:company) \
        return c2.id as comp_id,TRUE as has_stock_relation" % (comp_id)
    stock_rel_comps = pd.DataFrame(graph.run(stock_rel_statement).data(), columns=["comp_id", "has_stock_relation"])
    branch_rel_statement = "match p=(c:company{id:'%s'})-[:BRANCH*1..2]-(c2:company) return c2.id as comp_id,TRUE as has_branch_relation" % (comp_id)
    branch_rel_comps = pd.DataFrame(graph.run(branch_rel_statement).data(), columns=["comp_id", "has_branch_relation"])
    return (stock_rel_comps, branch_rel_comps)


def recommendation(comp_name, graph=graph, weights=(0.5, 0.4, 0.1), response_num=None, tag_link_filters=(0.0, 0.3, 0.3)):
    return 0

#%%
comp_infos, ctag_ctag, ctag_nctag, nctag_nctag, ctag_position, comp_id_name_dict = data_loader()

#%%
sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
comp_tags_all = pickle.load(open("../Data/Output/recommendation/comp_tags_all.pkl", "rb"))
comp_tags_all_df_dict = pd.DataFrame.from_dict(comp_tags_all, orient="index")
comp_tags_all_df_dict.fillna(1.0, inplace=True)
comp_tags_all_df_list = comp_tags_all_df_dict.applymap(lambda x: list(x) if x != 1.0 else []).reset_index().rename(index=str, columns={"index": "comp_id"})
comp_tags_all_spark_df = sqlContext.createDataFrame(comp_tags_all_df_list)
comp_tags_all_spark_df.cache()
comp_tags_all_spark_df.show(1)

#%%
target_comp = comp_tags_all.get("17286238067736781588")
comp_tags_all_spark_df.repartition(1000)

#%%
result_test_rdd = comp_tags_all_spark_df.rdd.map(lambda x: x[0]).zip(comp_tags_all_spark_df.rdd.map(lambda x: cal_tags_link_with_target(target_comp, x[1], x[2], (0, 0, 0))))
result_test_df = result_test_rdd.toDF(["comp_id", "result"])
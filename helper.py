import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment

def dcg_at_k(ranked_list, k):
    if k < 1:
        return 0.0

    ranked_list = ranked_list[:k]
    return sum([(2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ranked_list)])

def idcg_at_k(ground_truth, k):
    ideal_sorted = sorted(ground_truth, reverse=True)
    return dcg_at_k(ideal_sorted, k)

def ndcg_at_k(ranked_list, ground_truth, k):
    dcg = dcg_at_k(ranked_list, k)
    idcg = idcg_at_k(ground_truth, k)
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_list, ground_truth, k):
    if k < 1:
        return 0.0

    ranked_list = ranked_list[:k]
    relevant_items = [item for item in ranked_list if item in ground_truth]
    return len(relevant_items) / k

def cal_rels(tables, i):
    rels = []
    for table in tables:
        #print (table, tables[i])
        rels.append(cal_rel(table, tables[i]))
    return rels
    
def cal_rel(table1, table2):
    #print (type(table1), type(table2))
    rel_mat = np.zeros((table1.shape[1], table2.shape[1]))
    for i in range(table1.shape[1]):
        for j in range(table2.shape[1]):
            col1 = table1[:,i]
            col2 = table2[:,j]
            rel_mat[i][j] = 1 / (1+fastdtw(col1, col2)[0])
    row_ind, col_ind = linear_sum_assignment(cost_matrix=rel_mat, maximize=True)
    
    return rel_mat[row_ind, col_ind].sum()

def rand_select(charts, tables, neg):
    batch_size = len(charts)
    
    for i in range(batch_size):
        for j in range(neg):
            neg_index = np.random.randint(0, batch_size)
            while neg_index == j:
                neg_index = np.random.randint(0, batch_size)
            charts.append(charts[i].clone())
            tables.append(tables[neg_index].clone())
    
    return charts, tables

def hard_select(charts, tables, neg):
    batch_size = len(charts)
    
    for i in range(batch_size):
        rels = cal_rels(tables, i)
        idxs = np.argsort(rels)[::-1]
        idxs = np.delete(idxs, np.where(idxs == i))
        for j in range(neg):
            charts.append(charts[i].clone())
            tables.append(tables[idxs[j]].clone())
    
    return charts, tables
    
def semihard_select(charts, tables, neg):
    batch_size = len(charts)
    
    for i in range(batch_size):
        rels = cal_rels(tables, i)
        idxs = np.argsort(rels)
        idxs = np.delete(idxs, np.where(idxs == i))
        start = (batch_size-neg)//2
        end = (batch_size-neg)//2+neg
        for j in idxs[start:end]:
            charts.append(charts[i].clone())
            tables.append(tables[j].clone())
    
    return charts, tables

def easy_select(charts, tables, neg):
    batch_size = len(charts)
    
    for i in range(batch_size):
        rels = cal_rels(tables, i)
        idxs = np.argsort(rels)
        idxs = np.delete(idxs, np.where(idxs == i))
        for j in range(neg):
            charts.append(charts[i].clone())
            tables.append(tables[idxs[j]].clone())
    
    return charts, tables

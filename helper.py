import numpy as np

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



def data_type_judgment(row):
    file_path = row['table_data']
    table = pd.read_csv(file_path, sep = ',')
    columns = table.columns
    pair = eval(row['pairs'])
    x = pair[0][0]
    #selected_rows = random.sample(len(table),10)
    #sample_data = table.iloc[selected_rows]
    
    data_types = table.dtypes
    indicator = [0] * len(data_types)
    
    for i, col in enumerate(columns):
        if str(data_types[i]) == 'object':
            indicator[i] = 1
    
    if sum(indicator) == 0:
        return 1, x
    elif sum(indicator) > 1:
        return 2, x
    else:
        if indicator.index(1) != x:
            return 2, x
        else:
            idx = random.sample(list(range(len(table))), 1)
            #print (idx,x)
            sample_row = table.iloc[idx[0]]
            #print (sample_row[columns[x]])
            try:
                parser.parse(sample_row[columns[x]])
                return 0, x
            except:
                return 2, x
        
def cal_rel(table1, table2, x1, x2):
    cols_2 = table2.columns.to_list()
    #print (x1,x2)
    #print (cols_1, cols_2)
    cols_1.pop(x1)
    cols_2.pop(x2)
    rel_mat = np.zeros((len(cols_1),len(cols_2)))
    try:
        for i, col_1 in enumerate(cols_1):
            for j, col_2 in enumerate(cols_2):
                val_1 = table1[col_1].values
                val_2 = table2[col_2].values
                #print (val_1, val_2)
                rel_mat[i][j] = -1 / (1+fastdtw(val_1, val_2)[0])
        col_idxs_1, col_idxs_2 = linear_sum_assignment(rel_mat)
        max_rel = -rel_mat[col_idxs_1, col_idxs_2].sum() / len(cols_1)
    except:
        return 0
    
    #print (max_rel)
    return max_rel

def get_visualization_information(row):
    layout_path = row.layout
    chart_path = row.chart_data
    
    with open(layout_path, 'r') as layout_file:
        layout = json.load(layout_file)
    
    with open(chart_path, 'r') as chart_file:
        chart = json.load(chart_file)
        
    return chart, layout

def create_visualization(row, path):
    fig = plt.figure(figsize=(8,6),dpi=300)
    table = pd.read_csv(row.table_data, sep=',')
    cols = table.columns
    pairs = eval(row.pairs)
    
    for x, y in pairs:
        x = np.arange(len(table))
        y = table[cols[y]]
    
        plt.plot(x, y)

    plt.show()
    fig.savefig(path,dpi=fig.dpi,bbox_inches='tight')
    plt.clf()

def semihard(tables, charts, margin=0.1):

    batch_size = len(tables)
    
    dist_matrix = np.zeros((batch_size,batch_size))
    distances = torch.cal_rel(table, vis, p=2)

    # Initialize anchor, positive, and semi-hard negative indices
    anchor_indices = []
    positive_indices = []
    semi_hard_negative_indices = []

    for i in range(batch_size):
        # Find the positive pair (same sample)
        same_sample_indices = torch.nonzero(i == torch.arange(batch_size), as_tuple=False)
        negative_pairs = torch.where(distances[i] - distances[i, i] <= margin)[0]
        
        # Remove the same sample index from the negative pairs
        negative_pairs = negative_pairs[negative_pairs != i]

        if len(same_sample_indices) > 1 and len(negative_pairs) > 0:
            # Select the closest positive pair
            closest_positive_idx = same_sample_indices[torch.argmin(distances[i][same_sample_indices])]
            
            # Find the semi-hard negative sample
            diff = distances[i, closest_positive_idx] - distances[i, negative_pairs]
            semi_hard_neg_idx = negative_pairs[torch.argmax(diff)]
            
            anchor_indices.append(i)
            positive_indices.append(closest_positive_idx)
            semi_hard_negative_indices.append(semi_hard_neg_idx)

    anchor_indices = torch.tensor(anchor_indices)
    positive_indices = torch.tensor(positive_indices)
    semi_hard_negative_indices = torch.tensor(semi_hard_negative_indices)

    return anchor_indices, positive_indices, semi_hard_negative_indices

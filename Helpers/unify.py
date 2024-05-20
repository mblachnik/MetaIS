from pandas import DataFrame


def unify_weight_set(weight_set : DataFrame, path):
    filtered_df = weight_set[weight_set['weight'] == 1]
    filtered_df = filtered_df[['id']]

    filtered_df.to_csv(path_or_buf=path, index=False)
    return filtered_df
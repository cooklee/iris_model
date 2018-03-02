import pandas as pd
from sklearn.datasets import load_iris


def prepare_data_from_sklearn():
    """
    :return: data frame containing information about iris
    """
    loaded_iris = load_iris()
    df_iris = pd.concat(
                [pd.DataFrame(loaded_iris.get('data'), columns=loaded_iris.get('feature_names')),
                 pd.DataFrame(loaded_iris.get('target'), columns=['iris_type'])], axis=1)
    df_iris = df_iris.assign(iris_type=lambda fr: fr['iris_type'].replace({k: v for k, v in enumerate(loaded_iris.get('target_names'))}))
    return df_iris


def write_to_csv(data_frame, path, encoding='utf-8', index=False):
    """
    :param data_frame: data frame to write to csv
    :param path: path to file, if non existing it will be created
    :param encoding: encoding of a file
    :param index: bool if index should be created
    :return None
    """
    data_frame.to_csv(path, encoding=encoding, index=index)


if __name__ == "__main__":
    iris = prepare_data_from_sklearn()
    write_to_csv(iris, 'E:/projekty/project-01/data/raw/iris.csv')


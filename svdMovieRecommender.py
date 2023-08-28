'''
@article{10.1145/2827872,
author = {Harper, F. Maxwell and Konstan, Joseph A.},
title = {The MovieLens Datasets: History and Context},
year = {2015},
issue_date = {January 2016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {5},
number = {4},
issn = {2160-6455},
url = {https://doi.org/10.1145/2827872},
doi = {10.1145/2827872},
journal = {ACM Trans. Interact. Intell. Syst.},
month = dec,
articleno = {19},
numpages = {19},
keywords = {Datasets, recommendations, ratings, MovieLens}
}
'''

import pandas
import os
from sklearn.preprocessing import LabelEncoder
from lenskit.algorithms import funksvd
from lenskit import crossfold, util
from lenskit.batch import predict
from lenskit.metrics.predict import rmse

os.environ["MKL_THREADING_LAYER"] = "TBB"

ratings_dataframe = pandas.read_csv("ratings.csv", sep=",")

ratings_dataframe["userId"] = LabelEncoder().fit_transform(ratings_dataframe["userId"])
ratings_dataframe["movieId"] = LabelEncoder().fit_transform(ratings_dataframe["movieId"])

dataframe_renamed = ratings_dataframe.rename(columns={
    "userId": "user",
    "movieId": "item",
    "rating": "rating"
})

NUMBER_OF_FEATURES = 10
kFolds = 10
nSamples = 1

svd_model = funksvd.FunkSVD(NUMBER_OF_FEATURES)


def calculate_error(model, train_subset, test_subset):
    svd_model_copy = util.clone(model)
    svd_model_copy.fit(train_subset)

    predictions = predict(svd_model_copy, test_subset)

    return predictions


def train_model(dataframe, model):
    train_data = []
    test_data = []
    subset_errors = []

    partitioned_users = crossfold.partition_users(dataframe, kFolds, crossfold.SampleN(nSamples))

    fold_index = 1

    for train_subset, test_subset in partitioned_users:
        train_data.append(train_subset)
        test_data.append(test_subset)

        predictions = calculate_error(model, train_subset, test_subset)

        error = rmse(predictions["prediction"], predictions["rating"])

        subset_errors.append(error)

        fold_index += 1

        print(fold_index)

    return subset_errors


error = train_model(dataframe_renamed, svd_model)

print(error)

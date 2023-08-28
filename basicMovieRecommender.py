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

#ratings_features = ["user_id", "movie_id", "rating", "timestamp"]

ratings_dataframe = pandas.read_csv("ratings.csv")       #, names=ratings_features)
movies_dataframe = pandas.read_csv("movies.csv")

ratings_dataframe = ratings_dataframe.drop(ratings_dataframe.index[0])

ratings_dataframe = ratings_dataframe.astype("float")

movie_titles_dataframe = movies_dataframe[["movieId", "title"]]

movie_titles_dataframe["movieId"] = movie_titles_dataframe["movieId"].astype(str).astype(float)

merged_dataframe = pandas.merge(ratings_dataframe, movie_titles_dataframe, on = "movieId")

crosstab = merged_dataframe.pivot_table(values = "rating", index = "userId", columns = "title", fill_value = 0)

X = crosstab.T

from sklearn.decomposition import TruncatedSVD

NUMBER_OF_COMPONENTS = 12

SVD = TruncatedSVD(n_components = NUMBER_OF_COMPONENTS, random_state = 1)

matrix = SVD.fit_transform(X)

import numpy

correlation_matrix = numpy.corrcoef(matrix)

movie_titles = crosstab.columns

movies_list = list(movie_titles)

test_movie_index = movies_list.index("Batman: Year One (2011)")

example_correlation = correlation_matrix[test_movie_index]

print(list(movie_titles[(example_correlation < 1.0) & (example_correlation > 0.9)])) #print top 10 percent of movies matching the example movie



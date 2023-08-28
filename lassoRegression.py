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

import numpy
import pandas
import matplotlib
import seaborn
from matplotlib import pyplot
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

ratings_dataframe = pandas.read_csv("u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"],
                                    encoding="ISO-8859-1")

movies_dataframe = pandas.read_csv("u.item", sep="|",
                                   names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
                                          "unknown",
                                          "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                                          "Documentary",
                                          "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                                          "Sci-Fi", "Thriller", "War", "Western"],
                                   encoding="ISO-8859-1")

users_dataframe = pandas.read_csv("u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"],
                                  encoding="ISO-8859-1")

# print(users_dataframe)

NUMBER_OF_GROUPS = 5

users_dataframe["age_group"] = pandas.qcut(users_dataframe["age"], q=NUMBER_OF_GROUPS, precision=0)

merged_dataframe = pandas.merge(
    pandas.merge(ratings_dataframe, users_dataframe[["user_id",  # first merge ratings and users
                                                     "age_group",  # with these columns
                                                     "gender",
                                                     "occupation"]],
                 on="user_id",  # use user id as primary key
                 how="left"),  # left join
    movies_dataframe,  # take the prior dateframe and merge with movies
    on="movie_id",
    how="left")

merged_dataframe.drop(["movie_id", "title", "timestamp", "release_date", "video_release_date", "unknown", "IMDb_URL"],
                      axis=1,
                      inplace=True)

# print(merged_dataframe)

merged_dataframe["age_group"] = pandas.Categorical(merged_dataframe["age_group"])
merged_dataframe["gender"] = pandas.Categorical(merged_dataframe["gender"])
merged_dataframe["occupation"] = pandas.Categorical(merged_dataframe["occupation"])

age_group_dummies = pandas.get_dummies(merged_dataframe["age_group"])
gender_dummies = pandas.get_dummies(merged_dataframe["gender"])
occupation_dummies = pandas.get_dummies(merged_dataframe["occupation"])

merged_dataframe = pandas.concat([merged_dataframe,
                                  age_group_dummies,
                                  gender_dummies,
                                  occupation_dummies],
                                 axis=1)

merged_dataframe.drop(["age_group",  # we don't need this after doing encoding
                       "gender",
                       "occupation"],
                      axis=1,
                      inplace=True)

lasso_model = Lasso()

alpha = []

ALPHA_LENGTH = 7
NUMBER_OF_FOLDS = 5

for i in range(ALPHA_LENGTH):
    alpha.extend(numpy.arange(10 ** (i - 5), 10 ** (i - 4), 10 ** (i - 5) * 2))

params = {"alpha": alpha}

lasso_cross_validation = GridSearchCV(estimator=lasso_model,
                                      param_grid=params,
                                      scoring="neg_mean_absolute_error",
                                      cv=NUMBER_OF_FOLDS,
                                      return_train_score=True,
                                      verbose=1)

X = merged_dataframe.drop("rating",
                          axis=1)
X.columns = X.columns.astype(str)

y = merged_dataframe.rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_alpha_value_lasso = 1e-05

best_alpha_lasso_model = Lasso(alpha=best_alpha_value_lasso)

best_alpha_lasso_model.fit(X_train, y_train)

lasso_error = mean_squared_error(y_test,
                                 best_alpha_lasso_model.predict(X_test))

print(lasso_error)  # 1.190668

lasso_results_dataframe = pandas.DataFrame({
    "Features": X_train.columns,
    "Coefficient": best_alpha_lasso_model.coef_
})

figure, axes = pyplot.subplots(figsize=[7, 15])

seaborn.barplot(x="Coefficient", y="Features", ax=axes, data=lasso_results_dataframe)

lasso_results_dataframe.sort_values(by="Coefficient",
                                    inplace=True,
                                    ascending=False)

lasso_results_dataframe.reset_index(inplace=True, drop=True)  # reseting the index

lasso_results_dataframe = lasso_results_dataframe.iloc[:10]

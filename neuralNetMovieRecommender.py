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
from keras.layers import Input, Reshape, Dot
from keras.layers import Embedding
from keras.models import Model
from keras import optimizers, backend
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ratings_dataframe = pandas.read_csv("ratings.csv", sep=",")

ratings_dataframe["userId"] = LabelEncoder().fit_transform(ratings_dataframe["userId"])
ratings_dataframe["movieId"] = LabelEncoder().fit_transform(ratings_dataframe["movieId"])

dataframe_renamed = ratings_dataframe.rename(columns={
    "userId": "user",
    "movieId": "item",
    "rating": "rating"
})

LEARNING_RATE = 0.1
NUMBER_OF_FEATURES = 10
kFolds = 10
nSamples = 1

class CustomVectorizer:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, input):
        input = Embedding(self.input_dim,
                          self.output_dim)(input)
        return input


def custom_error(actual_y, predicted_y):
    return backend.sqrt(backend.mean(backend.square(predicted_y - actual_y), axis=-1))


def neural_network(number_of_inputs_users, number_of_inputs_movies, latent_space_dimension=20):
    input_layer_users = Input(shape=(1,))

    input_layer_movies = Input(shape=(1,))

    vectorized_inputs_users = CustomVectorizer(number_of_inputs_users,
                                               latent_space_dimension)(input_layer_users)

    vectorized_inputs_movies = CustomVectorizer(number_of_inputs_movies,
                                                latent_space_dimension)(input_layer_movies)

    latent_feature_vectors_users = Reshape((latent_space_dimension,))(vectorized_inputs_users)
    latent_feature_vectors_movies = Reshape((latent_space_dimension,))(vectorized_inputs_movies)

    output = Dot(axes=1)([latent_feature_vectors_users, latent_feature_vectors_movies])

    model = Model(inputs=[input_layer_users, input_layer_movies], outputs=output)

    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer,
                  loss="mean_squared_error",
                  metrics=[custom_error])

    return model


def run_model(model):

    number_of_users = ratings_dataframe["userId"].nunique()
    number_of_movies = ratings_dataframe["movieId"].nunique()

    model_object = model(number_of_users, number_of_movies)

    data = ratings_dataframe[["userId", "movieId"]].values

    target = ratings_dataframe["rating"].values  # what we want the model to predict

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.1)

    X_train = [X_train[:, 0], X_train[:, 1]]  # flattening the data
    X_test = [X_test[:, 0], X_test[:, 1]]

    NUMBER_OF_ITERATIONS = 20  #

    model_object.fit(x=X_train,
                     y=y_train,
                     epochs=NUMBER_OF_ITERATIONS,
                     verbose=1,
                     validation_split=.1)

    error = model_object.evaluate(x=X_test,
                                  y=y_test)

    return error


model_error = run_model(neural_network)

print(model_error)

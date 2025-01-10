import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv



def load_csv_data(csv_data):

    with open(csv_data) as r:
        data_reader = csv.reader(r)
        next(data_reader)
        data_reader = list(data_reader)




def create_sparse_matrix(list_data):
    """
    Create the data structure (a list of list)
    Returns
    -------
    data_by_usr_idx, data_by_itm_idx, map_idx_to_itm, map_idx_to_usr, map_itm_to_idx,map_usr_to_idx
    """
    
    map_usr_to_idx = {}
    map_idx_to_usr = []

    map_itm_to_idx = {}
    map_idx_to_itm = []

    data_by_usr_idx = []
    data_by_itm_idx = []

    usr_idx = 0
    itm_idx = 0

    for i in list_data:
        row0 = int(i[0])
        row1 = int(i[1])
        row2 = float(i[2])

        if row0 not in map_usr_to_idx:
            map_usr_to_idx[row0] = usr_idx
            map_idx_to_usr.append(row0)
            data_by_usr_idx.append([])
            usr_idx += 1

        if row1 not in map_itm_to_idx:
            map_itm_to_idx[row1] = itm_idx
            map_idx_to_itm.append(row1)
            data_by_itm_idx.append([])
            itm_idx += 1

        usr_idx1 = map_usr_to_idx[row0]
        itm_idx1 = map_itm_to_idx[row1]

        data_by_usr_idx[usr_idx1].append((itm_idx1, row2))
        data_by_itm_idx[itm_idx1].append((usr_idx1, row2))

    return data_by_usr_idx, data_by_itm_idx, map_idx_to_itm, map_idx_to_usr, map_itm_to_idx,map_usr_to_idx




def split_train_and_test(data, split_value=0.1):
    """
    Split the data  into training and test set
    Parameters
    ----------
    data : te data to be split
    split_value : the size of the test size

    Returns
    -------
    training data, test data
    """

    data_train = []
    data_test = []

    for i in range(len(data)):
        data_train.append([])
        data_test.append([])

    for j in data[i]:
        if np.random.rand() < split_value:
            data_test[i].append(j)
        else:
            data_train[i].append(j)

    return data_train, data_test




def compute_loss_rmse_bias_only(
    data_by_usr,
    data_by_itm,
    user_biases,
    item_biases,
    lambd = 0.01,
    gamma = 0.01,
    epoch = 50,
    ):

    all_loss = []
    all_rmse = []

    for i in range(epoch):
    # Update user biases
        for m in range(len(data_by_usr)):
            bias = 0
            item_counter = 0
            for (n, r) in data_by_usr[m]:
                bias += lambd * (r - item_biases[n])
                item_counter += 1

            bias = bias / (lambd * item_counter + gamma)
            user_biases[m] = bias

        # Update item biases
        for n in range(len(data_by_itm)):
            bias = 0
            user_counter = 0
            for (m, r) in data_by_itm[n]:
                bias += lambd * (r - user_biases[m])
                user_counter += 1

            bias = bias / (lambd * user_counter + gamma)
            item_biases[n] = bias

        # Compute loss after updating biases
        loss_one = 0
        rmse_one = 0
        counter = 0
        for m in range(len(data_by_usr)):
            for (n, r) in data_by_usr[m]:
                error = r - user_biases[m] - item_biases[n]
                loss_one += ( lambd / 2) * (error ** 2) + (gamma / 2) * (user_biases[m] ** 2 + item_biases[n] ** 2)
                rmse_one += error**2
                counter += 1

        rmse_one = np.sqrt(rmse_one / counter)

        print(f"Epoch {i}: loss = {loss_one}, rmse = {rmse_one}")

        # Store loss for this iteration
        all_loss.append(loss_one)
        all_rmse.append(rmse_one)

    return all_loss, all_rmse






def update_biases_and_vectors(
    data_by_usr,
    data_by_itm,
    usr_bias,
    itm_bias,
    usr_vect,
    itm_vect,
    gamma,
    lambd,
    K
    ):

    for m, user_ratings in enumerate(data_by_usr):
        bias_sum = np.sum([r - usr_vect[m] @ itm_vect[n] - itm_bias[n] for n, r in user_ratings])
        usr_bias[m] = bias_sum / (len(user_ratings) + gamma)

        sum1 = np.zeros((K, K))
        sum2 = np.zeros(K)
        for n, r in user_ratings:
            sum1 += np.outer(itm_vect[n], itm_vect[n])
            sum2 += itm_vect[n] * (r - usr_bias[m] - itm_bias[n])

        usr_vect[m] = np.linalg.solve(lambd * sum1 + np.identity(K), lambd * sum2)


    for n, item_ratings in enumerate(data_by_itm):
        bias_sum = np.sum([r - usr_vect[m] @ itm_vect[n] - usr_bias[m] for m, r in item_ratings])
        itm_bias[n] = bias_sum / (len(item_ratings) + gamma)

        sum1 = np.zeros((K, K))
        sum2 = np.zeros(K)
        for m, r in item_ratings:
            sum1 += np.outer(usr_vect[m], usr_vect[m])
            sum2 += usr_vect[m] * (r - usr_bias[m] - itm_bias[n])

        itm_vect[n] = np.linalg.solve(lambd * sum1 + np.identity(K), lambd * sum2)
    

    return usr_bias, itm_bias, usr_vect, itm_vect




def compute_log_likelihood(
    data_by_usr,
    usr_bias,
    itm_bias,
    usr_vect,
    itm_vect,
    gamma,
    lambd,
    tau
    ):

    for m, user_ratings in enumerate(data_by_usr):
        # Residual
        for n, r in user_ratings:
            error = r - (usr_vect[m] @ itm_vect[n] + usr_bias[m] + itm_bias[n])
            first_term += (error ** 2) 
    
    # Regularization
    loss_usr_vect = (tau / 2) * np.sum(np.linalg.norm(usr_vect, axis=1)**2)
    loss_itm_vect = (tau / 2) * np.sum(np.linalg.norm(itm_vect, axis=1)**2)

    loss = (lambd/2) * first_term + (gamma/2) * (np.sum(usr_bias ** 2) + np.sum(itm_bias ** 2)) + loss_usr_vect + loss_itm_vect

    return loss




def compute_rmse(
    data_by_usr,
    usr_bias,
    itm_bias,
    usr_vect,
    itm_vect,
    ):

    for m, user_ratings in enumerate(data_by_usr):
        for n, r in user_ratings:
            error = r - (usr_vect[m] @ itm_vect[n] + usr_bias[m] + itm_bias[n])
            rmse += error ** 2 

    return np.sqrt(np.mean(rmse))




def compute_loss_and_rmse_with_embeddings(
    data_by_usr,
    data_by_itm,
    user_biases,
    item_biases,
    lambd=0.1,
    gamma=0.2,
    tau=0.5,
    K=4,
    epoch=20,
):
    # Initialisation
    user_vector = np.random.normal(0, 1/np.sqrt(K), (len(data_by_usr), K))
    item_vector = np.random.normal(0, 1/np.sqrt(K), (len(data_by_itm), K))

    all_loss = np.zeros(epoch)
    all_rmse = np.zeros(epoch)

    for i in range(epoch):
        for m, user_ratings in enumerate(data_by_usr):
            # Update user biases
            bias_sum = np.sum([r - user_vector[m] @ item_vector[n] - item_biases[n] for n, r in user_ratings])
            user_biases[m] = bias_sum / (len(user_ratings) + gamma)

            # Update user vectors
            sum1 = np.zeros((K, K))
            sum2 = np.zeros(K)
            for n, r in user_ratings:
                item_vec = item_vector[n]
                sum1 += np.outer(item_vec, item_vec)
                sum2 += item_vec * (r - user_biases[m] - item_biases[n])

            user_vector[m] = np.linalg.solve(lambd * sum1 + tau * np.identity(K), lambd * sum2)

        
        for n, item_ratings in enumerate(data_by_itm):
            # Update item biases
            bias_sum = np.sum([r - user_vector[m] @ item_vector[n] - user_biases[m] for m, r in item_ratings])
            item_biases[n] = bias_sum / (len(item_ratings) + gamma)

            # Update item vectors
            sum1 = np.zeros((K, K))
            sum2 = np.zeros(K)
            for m, r in item_ratings:
                user_vec = user_vector[m]
                sum1 += np.outer(user_vec, user_vec)
                sum2 += user_vec * (r - user_biases[m] - item_biases[n])

            item_vector[n] = np.linalg.solve(lambd * sum1 + tau * np.identity(K), lambd * sum2)


        # Compute loss and RMSE
        loss_first_term = 0
        rmse_one = 0
        count = 0

        # Error in the loss
        for m, user_ratings in enumerate(data_by_usr):
            for n, r in user_ratings:
                error = r - (user_vector[m] @ item_vector[n] + user_biases[m] + item_biases[n])
                loss_first_term += (error ** 2) 
                rmse_one += error ** 2
                count += 1

        # Regularization terms for users and items
        loss_user_vector = (tau / 2) * np.sum(np.linalg.norm(user_vector, axis=1)**2)
        loss_item_vector = (tau / 2) * np.sum(np.linalg.norm(item_vector, axis=1)**2)

        # Total loss and RMSE calculation
        loss = (lambd/2) * loss_first_term + (gamma/2) * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2)) + loss_user_vector + loss_item_vector
        rmse = np.sqrt(rmse_one / count)

        # Store the results for each epoch
        all_loss[i] = loss
        all_rmse[i] = rmse

        print(f"Epoch {i+1}: loss = {loss:.6f}, rmse = {rmse:.6f}")

    return all_loss, all_rmse




# Show all possible movie given the title or a part of the title
def list_new_user_possible_movie(possible_title:str):
    return movies_df[movies_df['title'].str.contains(possible_title, case=False)]




# Get the corresponding index of the movie given the Id
def get_movie_index_new_user(index_value):
    return map_movie_to_index[index_value]




# Make predictions / recommendation
def predict_similar_movies(
    index_value_movie,
    rating_movie_new_user,
    number_pred,
    usr_bias = user_bi,
    itm_bias = item_bi,
    itm_vect = item_vec,
    epochs = 5,
    lambd = 0.01,
    gamma = 0.0001,
    tau = 0.9,
    K = 10,
    ):

    movie_index_new_user = get_movie_index_new_user(index_value_movie)
    new_user = [[(movie_index_new_user, rating_movie_new_user)]]
    usr_vect = np.random.normal(0, 1/np.sqrt(K), (len(new_user), K))

    # train new user vector
    for i in range(epochs):
        for m in range(len(new_user)):
            # update user vector
            sum1 = np.zeros((K, K))
            sum2 = np.zeros(K)
            for (n, r) in new_user[m]:
                sum1 += np.outer(itm_vect[n,:], itm_vect[n,:])
                sum2 += itm_vect[n,:] * (r - usr_bias[m] - itm_bias[n])
            u_first_term = lambd * sum1 + tau * np.identity(K)
            usr_vect[m,:] = np.dot(np.linalg.inv(u_first_term), lambd * sum2)

    
    score_for_item = np.zeros(len(data_by_movie_index))
    
    for n in range(len(data_by_movie_index)):
        score_for_item[n] = np.inner(usr_vect, itm_vect[n]).item() + 0.05 * itm_bias[n]

    # select top k score
    top_movies_with_scores = sorted(enumerate(score_for_item), key=lambda x: x[1], reverse=True)[:number_pred]
    top_movies = [i for i, _ in top_movies_with_scores]
    
    # get to k movies
    predict = []
    for i in range(len(top_movies)):
        movie_index = top_movies[i]
        movie_id = map_index_to_movie[movie_index]
        predict.append(movie_id)

    # get the rows corresponding to the movie recommendation / prediction
    new_movies_df = movies_df.loc[movies_df['movieId'].isin(predict)]
    new_movies_df = new_movies_df.drop(new_movies_df[new_movies_df['movieId'] == index_value_movie].index)

    
    return new_movies_df






"""
Aayush Joshipura, Mayukha Bhamidipati, Marley Ferguson, Abigail Sodergren
DS3500
HW04
Mar 27, 2023
"""

##Importing libraries
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

def euclidean_algo(df, track_col, artist_col, album_col, genre_col):
    """Creating a function to produce a euclidean distance algorithm and find the similarity score for
    each pair of songs in the dataframe. Requests a dataframe and column numbers to supply key info.
    Returns a dataframe with scores and info for each song."""

    ##Initializing list
    similarity_scores = []

    ##Nested for loops to create arrays and compute distance throughout dataframe

    for index in range(len(df)):
        ##Create a vector by indexing numerical variables and creating numpy arrays
        vector = df.iloc[index][4:-1].to_numpy()

        for index2 in range(len(df)):
            # if song2['track_id'] != song['track_id']:
            vector2 = df.iloc[index2][4:-1].to_numpy()
            song_similarity_scores = []

            euclidean_distance = np.linalg.norm(vector - vector2)
            similarity_score = 1 / (1 + euclidean_distance)

            ##Appending information for song1
            song_similarity_scores.append(df.iloc[index][track_col])  ##Track Name
            song_similarity_scores.append(df.iloc[index][artist_col])  ##Artist Name
            song_similarity_scores.append(df.iloc[index][album_col])  ##Album Name
            song_similarity_scores.append(df.iloc[index][genre_col])  ##Track Genre

            ##appending information for song2
            song_similarity_scores.append(df.iloc[index2][track_col])  ##Track Name
            song_similarity_scores.append(df.iloc[index2][artist_col])  ##Artist Name
            song_similarity_scores.append(df.iloc[index2][album_col])  ##Album Name
            song_similarity_scores.append(df.iloc[index2][genre_col])  ##Track Genre

            ##appending similarity scores between song1 and song2
            song_similarity_scores.append(similarity_score)

            similarity_scores.append(song_similarity_scores)
            new_df = pd.DataFrame(similarity_scores)

    return new_df

def main():

    ##Loading data and dropping extra columns
    songs = pd.read_csv("spotify-sampled.csv")
    songs = songs.drop(columns=['explicit'])

    ##Running function and renaming data columns
    all_df = euclidean_algo(songs, 3, 2, 1, 18)
    all_df.columns = ['song1', 'artist1', 'album1', 'genre1', 'song2', 'artist2', 'album2', 'genre2', 'score']

    ##Filtering by similarity threshold and including Regina Spektor songs
    regina_scores = all_df[all_df['artist1'] == 'Regina Spektor']
    filter_scores = all_df[all_df['score'] > 0.00001]

    ##Combining dataframes to create final dataframe for csv
    dfs = [regina_scores, filter_scores]
    total_scores = pd.concat(dfs)

    # Writing DataFrame to a CSV file for Neo4j Export
    total_scores.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()

##Nodes are the two songs, scores represent relationship between the two nodes
##Write query in Neo4j to find top 5 songs for Regina Spektor (query must work for all artists)
import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open("C:/Users/zsiro/Desktop/Python scripts/spotify_model/trained_model.sav",'rb'))

def genre_prediction(input_data):
    input_as_numpy = np.asarray(input_data)
    input_as_numpy_reshaped = input_as_numpy.reshape(1,-1)
    pred = loaded_model.predict(input_as_numpy_reshaped)
    return pred


def main():
    st.set_page_config(page_title="Spotify Genre Prediction Webapp", page_icon='🎧')
    st.title("Genre Prediction Webapp 🎧")
    

    #input data
    #Bpm,Energy,Danceability,Loudness,Liveness,Valence,Length,Acousticness,Speechiness,Popularity

    Bpm = st.number_input("Song's Beats per minute")
    Energy = st.number_input("Energy value of the song")
    Danceability = st.number_input("Danceability value")
    Loudness = st.number_input("Loudness value")
    Liveness = st.number_input("The song's liveness")
    Valence = st.number_input("Valence value")
    Length = st.number_input("The song's length")
    Acousticness = st.number_input("Acousticness value")
    Speechiness = st.number_input("Speechiness value")
    Popularity = st.number_input("The song's popularity")

    genre = ""

    if st.button("Evaluate Genre"):
        genre = genre_prediction([Bpm,Energy,Danceability,Loudness,Liveness,Valence,Length,Acousticness,Speechiness,Popularity])


    st.success(genre)

if __name__ == '__main__':
    main()
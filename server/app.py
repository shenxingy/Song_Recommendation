from flask import Flask, request, jsonify
import pandas as pd 
import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity

version = os.environ.get('VERSION', 'unknown')

app = Flask(__name__)
@app.route('/')
def index():
    return 'Server Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

# here is the key part
@app.route('/api/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        data = request.get_json(force = True)
        new_playlist = pd.DataFrame(data['songs'], columns=['track_name'])
        
        # the following code generate the recommendation from the embeddings extracted by matrix factorization
        track_name_embedding = pd.read_pickle('/data/model/track_name_embedding.pkl')
        candidate_track_name = track_name_embedding.index
        
        # it is possible that the track_name in the new playlist is not in the embedding, exclude them
        valid_track_name = []
        for i in new_playlist['track_name']:
            if i in track_name_embedding.index:
                valid_track_name.append(i)
                try:
                    candidate_track_name = candidate_track_name.drop(i)
                except:    
                    pass
        candidate_track_name_embedding = track_name_embedding.loc[candidate_track_name]
        print(f'valid track_name takes up {len(valid_track_name) / len(new_playlist) * 100:.2f}%')
        
        # Calculate the mean of the new playlist embedding for the seen track_name
        valid_new_playlist = new_playlist[new_playlist.track_name.isin(valid_track_name)]
        new_playlist_embedding = track_name_embedding.loc[valid_new_playlist.track_name].mean()

        
        # calculate the cosine similarity
        similarity = cosine_similarity(candidate_track_name_embedding, new_playlist_embedding.values.reshape(1, -1))
        similarity = similarity.reshape(-1)

        # extract the top 10 most similar track_name
        top_10_idx = similarity.argsort()[-10:][::-1]
        top_10_track_name = track_name_embedding.iloc[top_10_idx].index
        
        songs = list(top_10_track_name)
        last_updated_timestamp = os.path.getmtime('/data/model/track_name_embedding.pkl')
        model_date = datetime.datetime.fromtimestamp(last_updated_timestamp)
        
        return_data = {
            'songs': songs,
            'version': version,
            'last_updated': model_date.strftime('%Y-%m-%d')
        }
        return jsonify(return_data)

if __name__ == '__main__':
    app.run(debug=True)
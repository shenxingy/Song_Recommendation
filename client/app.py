from flask import Flask, request, render_template
import requests

url = 'http://localhost:52006/api/recommend'
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        data = request.form['music_input']
        print("recieved data: ", data)
        # reconstruct the data into a list of songs
        data = data.split(', ')
        # convert the data to a json and send request to my server app
        json_data = {'songs': data}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=json_data)
        
        if response.status_code == 200:
            # Parse JSON response
            recommendation_list = response.json().get('songs', [])
            print("Recommendation list:", recommendation_list)
        else:
            print("Failed to get recommendations.")
        print(data)
        
        # do something with the data
        return render_template('recommendation.html', recommendation_list=recommendation_list)
    
if __name__ == '__main__':
    app.run(debug=True)
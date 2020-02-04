import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify


url = "https://raw.githubusercontent.com/med-cabinet-5/data-science/master/data/build_data.csv"
df = pd.read_csv(url)
tfidf = TfidfVectorizer(min_df=0.025, max_df=.98, ngram_range=(1,3))

def create_app():
    """Creates and configures Flask app instance"""
    app = Flask(__name__)
    
    def pred_list(x):
        """
        x = string to predict from (description)
        1. Predict the nearest neighbors to the inputted description
        2. Predict what type of cannabis the user is looking for with probability

        """
        # Read in data
        df = pd.read_csv("https://raw.githubusercontent.com/med-cabinet-5/data-science/master/data/canna.csv")
        # Fill NaN with empty strings
        df = df.fillna("")

        # Instantiate vectorizer object
        tfidf = TfidfVectorizer(stop_words="english", min_df=0.025, max_df=.98, ngram_range=(1,3))

        # Create a vocabulary and get word counts per document
        dtm = tfidf.fit_transform(df['alltext'])

        # Get feature names to use as dataframe column headers
        dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

        # Fit on TF-IDF Vectors
        nn = NearestNeighbors(n_neighbors=5, algorithm="kd_tree", radius=0.5)
        nn.fit(dtm)

        # Turn Review into a list, transform, and predict
        review = [x]
        new = tfidf.transform(review)
        pred = nn.kneighbors(new.todense())[1][0]


        #create empty list
        pred_dict = []
        for x in pred:
            # add new dictionary to pred_dict containing predictions
            preds_list ={"strain":df["Strain"][x],
                         "type": df["Type_raw"][x],
                         "description": df["Description_raw"][x],
                         "flavor": df["Flavor_raw"][x],
                         "effects": df["Effects_raw"][x],
                         "ailments": df["Ailment_raw"][x]}
            pred_dict.append(preds_list)

        # Load data for model 2
        model = pickle.load(open("../stretch.sav", "rb"))
        #Pull result out
        pred_2 = model.predict(review)[0]

        #Grab max predict proba                   
        predict_proba = model.predict_proba(review)[0].max() * 100

        # Mapper to change result into string
        mapper = ({5: "Hybrid",
               4: "Indica",
               3: "Sativa",
               2: "Hybrid, Indica",
               1: "Sativa, Hybrid"})

        # Apply mapper to newly made Series
        strain_type = pd.Series(pred_2).map(mapper)[0]

        # Create new dictionary element
        new_dict = {"proba":f"There is a {round(predict_proba, 2)}% that your looking for a {strain_type}"}

        # Add new dicitonary to list of dictionaries
        pred_dict.append(new_dict)

        return pred_dict
    
    

    @app.route('/json', methods=['GET'])
    def root():
        req_data = request.get_json()
        our_string = req_data['USER_INPUT_STRING']
        """Until we are on the same page with the front end"""
        output = pred_list(our_string)
        return jsonify(output)

    return app

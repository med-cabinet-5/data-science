import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from flask import Flask, request


url = "https://raw.githubusercontent.com/med-cabinet-5/data-science/master/build_data.csv"
df = pd.read_csv(url)
tfidf = TfidfVectorizer(min_df=0.025, max_df=.98, ngram_range=(1,3))

def create_app():
    """Creates and configures Flask app instance"""
    app = Flask(__name__)
    
    
    def pred(x, df):
        """Make prediction and return nested dictionary

           x = string
        """
        # Load mode file and perform prediction
        model = pickle.load(open("mvp.sav", "rb"))
        x = [x]
        trans = tfidf.transform(x)
        pred = model.kneighbors(trans.todense())[1][0]

        # create empty dictionary
        pred_dict = {}


        # add new dictionary to pred_dict containing predictions
        preds_dict = {(1 + len(pred_dict)): {"strain": df["Strain"][x],
                                             "type": df["Type"][x],
                                             "description": df["Description"][x],
                                             "flavor": df["Flavor"][x],
                                             "effects": df["Effects"][x],
                                             "ailments": df["Ailment"][x]}}
        pred_dict.update(preds_dict)

        return pred, pred_dict
    

    @app.route('/json', methods=['POST'])
    def root():
        req_data = request.get_json()
        our_string = req_data['USER_INPUT_STRING']
        """Until we are on the same page with the front end"""
        # return pred(our_string, df)
        return {'description':'API active'}

    return app

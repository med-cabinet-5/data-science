# imports
from collections import Counter
import pickle
import pandas as pd
import xgboost
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify

url = "https://raw.githubusercontent.com/med-cabinet-5/data-science/master/data/canna.csv"
# Read in data
df = pd.read_csv(url)
# Fill NaN with empty strings
df = df.fillna("")


def lister(x):
    """Function to return top seen words from a desired column"""
    # make new df from preds
    df_preds = df.loc[pred]
    # make empty list
    word_ls = []

    # loop over items in desired column and append into a list and title it
    for x in df_preds[x]:
        x = x.split(" ")
        for x in x:
            word_ls.append(x.strip(",").title())

    # Count the number of times each element appears
    count = Counter(word_ls)

    # Create new empty list
    word_ls = []

    # Loop over first 3 most common elements and join together in a string
    for x in range(3):
        word_ls.append(count.most_common(3)[x][0])
    result = ", ".join(word_ls)

    return result


def starter(x):
    # Instantiate vectorizer object
    tfidf = TfidfVectorizer(stop_words="english", min_df=0.025, max_df=.98, ngram_range=(1, 3))

    # Create a vocabulary and get word counts per document
    dtm = tfidf.fit_transform(df['alltext'])

    # Get feature names to use as dataframe column headers
    dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

    # Fit on TF-IDF Vectors and return 30 neighbors
    nn = NearestNeighbors(n_neighbors=30, algorithm="kd_tree", radius=0.5)
    nn.fit(dtm)

    # Turn Review into a list, transform, and predict
    review = [x]
    new = tfidf.transform(review)

    global pred
    pred = nn.kneighbors(new.todense())[1][0]

    return


def pred_list(x):
    """
    x = string to predict from (description)
    1. Predict the nearest neighbors to the inputted description
    2. Predict what type of cannabis the user is looking for with probability

    """
    starter(x)

    # create empty list
    pred_dict = []

    # only loop through 5 closest neighbors
    for x in pred[:5]:
        # add new dictionary to pred_dict containing predictions
        preds_list = {"strain": df["Strain"][x],
                      "type": df["Type_raw"][x],
                      "description": df["Description_raw"][x],
                      "flavor": df["Flavor_raw"][x],
                      "effects": df["Effects_raw"][x],
                      "ailments": df["Ailment_raw"][x]}
        pred_dict.append(preds_list)

    return pred_dict


def pred_list2(x):
    starter(x)

    # Create initial dictionary with tops from relevant columns
    test_dict = {"top_effects": lister("Effects_raw"),
                 "top_flavors": lister("Flavor_raw"),
                 "top_ailments": lister("Ailment_raw")
                 }

    model = pickle.load(open("stretch.sav", "rb"))
    # Pull result out
    pred_2 = model.predict(review)[0]

    # Grab max predict proba
    predict_proba = model.predict_proba(review)[0].max() * 100

    # Mapper to change result into string
    mapper = ({5: "Hybrid",
               4: "Indica",
               3: "Sativa",
               2: "Hybrid, Indica",
               1: "Sativa, Hybrid"})

    # Apply mapper to newly made Series
    strain_type = pd.Series(pred_2).map(mapper)[0]

    # Add new entry
    test_dict["proba"] = f"There is a {round(predict_proba, 2)}% that your looking for a {strain_type}"

    return test_dict

def create_app():
    """Creates and configures Flask app instance"""
    app = Flask(__name__)
    
    @app.route('/stats', methods=['GET'])
    def root():
        req_data = request.get_json()
        our_string = req_data["USER_INPUT_STRING"]
        output = pred_list2(our_string)
        return output
    
    @app.route('/json', methods=['GET'])
    def root2():
        req_data = request.get_json()
        our_string = req_data['USER_INPUT_STRING']
        output = pred_list(our_string)
        return jsonify(output)

    @app.route("/")
    def root3():
        return """Med Cabinet 5 By  David Vollendroff, Jan Jaap de Jong, Nicole Williams, & Mikio Harman"""

    return app


import pickle
from flask import Flask

url = "https://raw.githubusercontent.com/kushyapp/cannabis-dataset/master/Dataset/Strains/strains-kushy_api.2017-11-14.csv"
url2 = "https://raw.githubusercontent.com/med-cabinet-5/data-science/master/cannabis.csv"
df = pd.read_csv(url)
df2 = pd.read_csv(url2)

def create_app():
    """Creates and configures Flask app instance"""
    app = Flask(__name__)

    def cleaner(df1, df2):
        """Clean Dataframes and concat together & keep as much text information as possible"""

        # Fill NaN with empyty strings to make concat easier
        df1 = df1.fillna("")

        # Concat all df1 text columns into a single column containing row corpus
        df1['alltext'] = df1['description'].str.cat(df1["type"], sep=" ")
        df1['alltext'] = df1['alltext'].str.cat(df1["effects"], sep=" ")
        df1['alltext'] = df1["alltext"].str.cat(df1["ailment"], sep=" ")
        df1['alltext'] = df1["alltext"].str.cat(df1["flavor"], sep=" ")
        df1['alltext'] = df1["alltext"].str.cat(df1["location"], sep=" ")
        df1['alltext'] = df1["alltext"].str.cat(df1["terpenes"], sep=" ")

        # Rename columns to match DF to concat too
        df1 = df1[["name", "description", "alltext", "type", "effects", "ailment", "flavor"]]
        df1 = df1.rename(columns={"name": "Strain",
                                  "type": "Type",
                                  "effects": "Effects",
                                  "flavor": "Flavor",
                                  "description": "Description"
                                  })

        # Fill NaN with empyty strings to make concat easier
        df2 = df2.fillna("")

        # Add 'ailment' column to df2 to make concat easier
        df2["ailment"] = ""

        # Concat all df2 text columns into a single column containing row corpus
        df2['alltext'] = df2['Effects'].str.cat(df2["Flavor"], sep=" ")
        df2['alltext'] = df2['alltext'].str.cat(df2["Description"], sep=" ")
        df2['alltext'] = df2["alltext"].str.cat(df2["Type"], sep=" ")

        # Concat df1 & df2
        df_cat = pd.concat([df1, df2], sort=False)

        # Create column that shows the length of alltext to identify low word count rows
        df_cat["len_length"] = df_cat["alltext"].apply(lambda x: len(x))

        # Filter for rows only with more than 100 chars to filter our undescriptive rows out of the df
        condition = df_cat['len_length'] > 100
        df_cat = df_cat[condition]

        # Create "lemmas" column to clean and groupby on cleaned strain names
        df_cat['lemmas'] = df_cat['Strain'].apply(get_lemmas)

        # Combine lemmas lists to create cleaned strain names with hyphens removed and text lowered
        df_cat["strain_clean"] = df_cat["lemmas"].apply(lambda x: " ".join(x))
        df_cat["strain_clean"] = df_cat["strain_clean"].apply(lambda x: x.replace("-", " "))
        df_cat["strain_clean"] = df_cat["strain_clean"].apply(lambda x: x.lower())

        # Groupby "strain_clean" and agg using join then reset index
        df_cat = df_cat.groupby("strain_clean").agg(" ".join).reset_index()

        # Only keep needed columns
        keep_cols = ["strain_clean", "Type", "Effects", "ailment", "Flavor", "Description", "alltext"]
        df_cat = df_cat[keep_cols]

        # Rename columns to keep similar name structure
        df_cat = df_cat.rename(columns={"strain_clean": "Strain",
                                        "ailment": "Ailment"})

        # Add in single strain name
        df_cat["Strain"][0] = "one to one"

        # Title the strain names
        df_cat["Strain"] = df_cat["Strain"].apply(lambda x: x.title())

        # Remove duplicates and make text presentable
        ls_dupe = ["Effects", "Flavor", "Type", "Ailment", "alltext"]
        for x in ls_dupe:
            df_cat[x] = df_cat[x].apply(get_lemmas)
            df_cat[x] = df_cat[x].map(lambda x: list(set(map(str.lower, x))))
            df_cat[x] = df_cat[x].str.join(", ")
            df_cat[x] = df_cat[x].apply(lambda x: x.title())

        return df_cat

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

    clean_df = cleaner(df, df2)

    @app.route('/pred/<string>', methods=['GET'])
    def root():
        return pred(string, clean_df)

    return app

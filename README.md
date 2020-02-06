<div align="center">
  <img src="https://github.com/med-cabinet-5/data-science/blob/master/img/Med%20(1).png"><br>
</div>

# Med Cabinet 5 
App for new cannabis consumers (especially those trying to get off of pharmaceuticals) who want to use cannabis as a means to battle medical conditions and ailments. Help patients find the right strains, dosing, intake method and intake schedule! 
Use user data along with strain data to build ML models to guide recommendations.

- What problem does your app solve?  
    - Patients unable to find proper cannabis strain, dosage, intake method, and intake schedule.
    - Turning away from pharmaceuticals / finding healthy alternatives and providing customized care for each customer
    - Pitch: App for new cannabis consumers (especially those trying to get off of pharmaceuticals) who want to use cannabis as a means to battle medical conditions and ailments. Help patients find the right strains, dosing, intake method and intake schedule! Use user data along with strain data to build ML models to guide recommendations.

# Team:
- UI: [Stephen Gary](https://github.com/stgary)
- React1:  [Kennith Howe](https://github.com/Draxxus702) and [Danika Thomson](https://github.com/DanikaT)
- React2: [KP Parrish](https://github.com/KParrish193)
- Backend: [Lexie Jiang](https://github.com/jiangeyre)
- Data Engineer:  [Mikio Harman](https://github.com/mpHarm88), [Jan Jaap de Jong](https://github.com/Okocha76), [David Vollendroff](https://github.com/DavidVollendroff) and [Nicole Williams](https://github.com/nwilliams030)
- Project Lead: [Vinni Hoke](https://github.com/vinnihoke)

# Data
- [Kaggle Data](https://www.kaggle.com/kingburrito666/cannabis-strains)
- [Kushy App Data](https://github.com/kushyapp/cannabis-dataset/tree/master/Dataset/Strains)

# **How to Use**

Find our Flask API [here](https://github.com/med-cabinet-5/data-science/tree/master/med-cabinet-predictions)

[Homepage](https://med-cabinet-5.herokuapp.com/): This is the landing page for the pred-airbnb web app.

#### Endpoint 1: https://med-cabinet-5.herokuapp.com/json

#### **input**: 

```
{
  "USER_INPUT_STRING": "i want to feel happy and uplifted"
}
```

#### **Expected test output**:
```
[
    {
        "ailments": "",
        "description": "Arabian Gold is a heavy sativa strain of mysterious origins. It may leave most consumers feeling mentally checked out, so you could feel foggy-headed and, for example, lose your train of thought when conversing with friends. ",
        "effects": "Giggly, Euphoric, Creative, Tingly, Sleepy",
        "flavor": "Tea",
        "strain": "Arabian Gold",
        "type": "Sativa"
    },
    {
        "ailments": "",
        "description": "Skunky Diesel is a nice indica-dominant strain that is a cross of Sensi Skunk with NYC Diesel.  A nice relaxing high that you will feel in your face pretty quickly.  She definitely carries the diesel taste and lovely diesel effects.",
        "effects": "Happy, Relaxed, Energetic, Uplifted, Sleepy",
        "flavor": "Skunk, Earthy, Diesel",
        "strain": "Skunky Diesel",
        "type": "Hybrid"
    },
    {
        "ailments": "",
        "description": "Purple Princess is not the girl from your average fairy tale. Thought to be the daughter of Cinderella 99 and Ice Princess, this hybrid has a habit of creeping up on you. Fruity and skunky, Purple Princess produces a medium-level effect. While it won’t leave you stuck on the couch, you will feel medicated. Purple Princess is characterized by dense, small, purple buds and typically flowers around 5-6 weeks.",
        "effects": "Happy, Euphoric, Relaxed, Arouse, Uplifted",
        "flavor": "Earthy, Pungent, Sweet",
        "strain": "Purple Princess",
        "type": "Hybrid"
    },
    {
        "ailments": "",
        "description": "While normally difficult to ignore, this Walrus is one sneaky gal. Upon first taste, many users may not feel the Walrus’s effects; however, give it a few minutes and this strain will surprise you in ways you never thought possible. Perhaps slightly indica-dominant, Walrus Kush may not be the best medication for getting things done. Fairly well-balanced, it’s a giggly and sociable strain while providing deep, body-relaxing effects at the same time. Great for users who suffer from stress or migraines, Walrus just might be worth a weekend trip to the dispensary.",
        "effects": "Happy, Giggly, Hungry, Creative, Uplifted",
        "flavor": "",
        "strain": "Walrus Kush",
        "type": "Hybrid"
    },
    {
        "ailments": "",
        "description": "A British Columbia native, Killer Queen is the outcome of an imaginative cross between G13 and Cinderella 99. Uplifting and thought-provoking, this hybrid is great for the workaholic who would like some daytime relief. The effects of this strain are felt most heavily in the face, eyes, and forehead, evidencing the sativa aspects of this hybrid. Upon first taste, Killer Queen takes up the fruity characteristics of Cinderella 99. The tropical flavor, however, is quickly followed by an earthy, herbal tone. If you are searching for an energizing strain that allows you to focus, Killer Queen may be just the perfect match.",
        "effects": "Happy, Energetic, Creative, Uplifted, Talkative",
        "flavor": "Tropical, Sweet, Citrus",
        "strain": "Killer Queen",
        "type": "Hybrid"
    }
]
```

#### Endpoint 2: https://med-cabinet-5.herokuapp.com/stats

#### **input**: 
```
{
  "USER_INPUT_STRING": "i want to feel happy and uplifted"
}
```

#### **Expected output**:
```
{
    "proba": "There is a 95.68% that your looking for a Hybrid",
    "top_ailments": ", Depression, Stress",
    "top_effects": "Happy, Uplifted, Relaxed",
    "top_flavors": "Earthy, Sweet, Pungent"
}
```

# Dependencies
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Flask](https://flask.palletsprojects.com/en/1.1.x/api/)
- [Gunicorn](https://gunicorn.org/)
- [Numpy](https://docs.scipy.org/doc/numpy/reference/)

# License
- [MIT License](https://opensource.org/licenses/MIT)

# Word Embedding Visuals (Transfer learning with SpaCy)

### Cannabis Flavor Word Embeddings 
<div align="center">
  <img src="https://github.com/med-cabinet-5/data-science/blob/master/img/flavor.png"><br>
</div>

### Cannabis Effects Word Embeddings 
<div align="center">
  <img src="https://github.com/med-cabinet-5/data-science/blob/master/img/effects.png"><br>
</div>

### Cannabis Ailments Word Embeddings 
<div align="center">
  <img src="https://github.com/med-cabinet-5/data-science/blob/master/img/ailment.png"><br>
</div>




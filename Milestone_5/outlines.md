What we need to have:

1) Up-to-date notebooks
2) README (if anything was changed for the notebooks)
3) 6-page report (as a pdf)
4) 2-minute screencast (we need to prepare slides, the text or at least talking points)
5) zip file with cleaned data


Outline for the report:

Timur's notes:

1) Data we used. How many samples, how many features, sources. 
Train, test, validation set sizes.  

2) Type of classification (multi-class, multi-label), imbalance of the classes.
Loss function. Metrics to monitor.

3) Baseline (dummy) classifiers 
(we could consider some popular classifiers without much tuning).  

4) Traditional methods.

5) Deep learning methods (from-scratch vs pre-trained).

6) Results. Comparison. What we could have done if we had even more time. 

7) Visualizations. We can take some recent movies, find their posters and 
do predictions for all methods we covered.

- distribution of genres
- train/test score plots
- correlation heatmap of genre pairs
- we could try to find out what parts of the image triggered positive 
genre identification (if we have time) 

8) Topics to discuss: overfitting. Six page is not a lot. With visualizations
we need to budget and mention only more important stuff.

9) Important words in the overview/cast/crew analysis.


Alex's notes:

Data collection:

Sources: IMDB, TMDB, random movies, features available
Challenges: throttling, distributed collection

Dataset:

Features: cast / crew, overview / plot,  posters
Features we tried but dismissed
Curse of dimensionality
Imbalance

Problem:

Predict movie genre
Model: multiclass vs multilabel, pros and cons
Classifiers to consider: LR / SVM / Random Forest / CNN
Classifier we considered and dismissed
Multilabel support, multiclass support
How to handle the imbalance (class weights, native resistance)
PCA or not PCA

Implementation:

What we implemented
Model comparision: training / testing time, how much data required
Model performance: results, how to improve


Screencast:

1) Story. Problem. Alex, Keenan, Timur wanted to go to see a movie but had trouble
choosing. Alex wanted to see an GenreA movie, Keenan - GenreB but not GenreC, etc.
The IMDB site was down but they had data. So they decided to use their data science
skills. 

2) We could also organize some experiments where we would guess genres by looking
at random posters and compare the results with the deep learning algorithms.

3) Mention methods, show results, show posters and compare predictions for 
various methods. Visualizations.

4) Here the budget will be even more limited. So only more important stuff.

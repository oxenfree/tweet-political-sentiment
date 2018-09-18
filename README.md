# Political Sentiment and Text Analysis of Candidate Tweets
## Text Mining
This project uses the NLTK word tokenizer and TFIDF transformer, as well as Multinomial Naive Bayes and Support Vector Machine models to predict winning and losing campaigns. 
Text documents (500 tweets) from 10 campaigns for special elections in 2018 were analyzed and compared to four upcoming campaigns to predict the outcome.
The tweets are also analyzed for amount of engagement (favorites and retweets) as well as for sentiment.

## Engagement Analysis
Previous research suggests that the number of likes and retweets each candidate receives can predict their success in the race. In 2018, this hasn't been the case. Most likely due to the outsized amount of national attention these smaller races have garnered.

## Sentiment Analysis
Each Tweet from each candidate was then analyzed for sentiment. 

This analysis shows almost no difference in sentiment between those who have lost and those who have won.

## Classifying Campaign Outcome by Text Analysis

Two machine learning models were created and trained on the “won” and “lost” dataframes, using the cleaned tweet texts. First, pipelines were built for the Multinomial Naïve Bayes and Support Vector Machine (SVM) algorithms. The pipelines used the NTLK Count Vectorizer and the TFIDF Transformer. Parameters for the algorithms such as ngram range, whether to use stop words or not, and “min df” (minimum number of word matches to add to the TFIDF transformer) were then fed into the pipelines using the Grid Search Cross Validation with a cross-validation parameter set to “3”.

The Grid Search Cross Validation then chooses the best set of parameters to use when fitting the model to the training data. The best model for both SVM and MNB, predicted the class as either 1 (won) or 0 (lost) at an overall accuracy of around 80%. Multinomial Naïve Bayes performed slightly better with recall of the 0 (losing) class, while SVM performed slightly better with precision of the 1 (winning) class.

The same pipeline and cross-validation process and training data can be used to fit these models again. Instead of testing against a hold-out subset of training data, the tweet texts for upcoming elections was used. Since the label “won” corresponds with the class “1”, a prediction was made by simply summing the number of tweets with “winning” type texts.

See the jupyter notebook inside for results.

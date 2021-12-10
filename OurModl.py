import pandas as pd
import pickle
# Sklearn processing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load data set
tr_df = pd.read_csv("C:training_data_2_csv_UTF.csv")

bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

tr_df['screen_name_binary'] = tr_df.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
tr_df['name_binary'] = tr_df.name.str.contains(bag_of_words_bot, case=False, na=False)
tr_df['description_binary'] = tr_df.description.str.contains(bag_of_words_bot, case=False, na=False)
tr_df['status_binary'] = tr_df.status.str.contains(bag_of_words_bot, case=False, na=False)
tr_df['listed_count_binary'] = (tr_df.listed_count>20000)==False
tr_df['location_binary'] = (tr_df.location.isnull())

tr_df.drop(['id_str','screen_name','description','url','name','location','status','created_at'], axis=1, inplace=True)
tr_df.drop(['lang'],axis=1, inplace=True)
tr_df.drop(['listed_count'],axis=1, inplace=True)
tr_df.drop(['default_profile_image'],axis=1, inplace=True)
tr_df['has_extended_profile'] = (tr_df.has_extended_profile.isnull())

X = tr_df.drop(['bot'],axis=1)
y = tr_df['bot']

from sklearn.feature_selection import SelectKBest

kbest = SelectKBest(k=10)
X_new = kbest.fit_transform(X,y)

print("----------------------------------------")
print("Feature selection", kbest.get_support())
print("----------------------------------------")
print("Feature scores", kbest.scores_)
print("----------------------------------------")
print("Selected features:", list(X.columns[kbest.get_support()]))
print("----------------------------------------")
print("Removed features:", list(X.columns[~kbest.get_support()]))

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=20)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3,random_state=101)
rf = rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

print("----------------------------------------")
print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))

#predres = concat(y_pred_train,y_pred_test)
#print(predres)
pickle.dump(rf, open('finalmodel.pkl','wb'))
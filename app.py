import numpy as np
import tweepy
from pandas import DataFrame   
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) 
model = pickle.load(open('finalmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')        

@app.route('/predict',methods=['POST'])
def predict():
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
	Screen_name_binary = request.form['t1']
	
	consumer_key="qrbvg9yW3oNQ9QO95SM31FcWr"
	consumer_secret="B88u2Rh1t0watdxyDjn4toE63GaSpDft78B5uKlgOV3h7mbWu0"
	access_token="1363755308309381123-AVyYyk0HC6NdwyA4No06oNK4jUFX8o"
	access_token_secret="tIH7A7bObvCC22voUHg6qfHjx87osBBWEuuoxFle3Wqzx"
	
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)
	screen_name = (Screen_name_binary)
	user = api.get_user(screen_name)
	name = user.name
	description = user.description
	screen_name = user.screen_name
	status = user.status
	location = user.location
	default_profile = user.default_profile
	profile_image = user.profile_image_url_https
	
	bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
	
	list1 = [name, description, screen_name, status, location]
	
	df = DataFrame(list1, columns = ['binary_features'])
	
	df['values'] = df.binary_features.str.contains(bag_of_words_bot, case=False, na=False)
	
	Screen_name_binary= df.values[2][1]
	Name_binary = df.values[0][1]
	Description_binary = df.values[1][1]
	Status_binary = df.values[3][1]
	location_binary = df.values[4][1]
	
	Verified = user.verified
	Followers_count = user.followers_count
	Friends_count = user.friends_count
	Statuses_count = user.statuses_count
	Default_profile_image = user.default_profile_image
	Listed_count = user.listed_count
	id = user.id
	Listed_count_binary = (user.listed_count>20000)==False
	result = model.predict([[id, Followers_count, Verified, default_profile, Screen_name_binary, Name_binary, Description_binary, Status_binary, Listed_count_binary, location_binary]])
	output = result[0]

	if int(output) == 1:
		prediction = 'ACCOUNT IS A BOT.'
	else:
		prediction = 'ACCOUNT IS NOT A BOT.'
	# output = prediction[0]
	return render_template('index.html', prediction_text = prediction, profile_img = profile_image, scr_name = screen_name)

if __name__ == "__main__":
    app.run(debug=True)
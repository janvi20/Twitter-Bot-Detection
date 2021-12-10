import csv
from flask import Flask, request, jsonify, render_template
app = Flask(__name__) 
            
@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/connection',methods=['POST'])
def connection():
	if request.method == 'POST':
		text1 = request.form.get("t1")
		with open('data1.csv', newline='',encoding="utf-8") as csvfile:
			data = csv.DictReader(csvfile)
			for row in data:
				if(row['screen_name'] == text1):
					res =row['predicted_results']
					if(res == 1):
						pred = 'Bot'
					else:
						pred = 'NonBot'
		return render_template('index.html', review_text=pred)
	else:
		return "Unsuccessful"

if __name__ == "__main__":
 app.run(debug=True)
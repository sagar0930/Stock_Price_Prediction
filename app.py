import pandas as pd
from flask import Flask, request, render_template
import tensorflow.keras as keras
from datetime import datetime
import os
os.chdir("C://Users/Admin/Downloads/Data Science/Session_41_DS_Project_Structure/DS1/Stock_Price_Prediction")


# Create Flask app
app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('stock_price_prediction.h5')

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        date_str = request.form['Date']
        Date = datetime.strptime(date_str, '%d/%m/%Y')
        
        # Convert Date to Unix timestamp
        Date = Date.timestamp()
        print('11111111111111111',type(Date))
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low = float(request.form['Low'])
        Volume = float(request.form['Volume'])
        headline_sentiment = float(request.form['headline_sentiment'])
        summary_sentiment = float(request.form['summary_sentiment'])
        
        # Create a DataFrame with the input data
        data = pd.DataFrame({
            'Date': [Date],  # Convert to Unix timestamp
            'Open': [Open],
            'High': [High],
            'Low': [Low],
            'Volume': [Volume],
            'headline_sentiment': [headline_sentiment],
            'summary_sentiment': [summary_sentiment]
        })
               
        # Make prediction using the loaded model
        prediction = model.predict(data)

        return render_template("index.html", prediction_text=prediction)
    
# Run the app if executed directly
if __name__ == "__main__":
    app.run()
    #app.run(host="0.0.0.0", port=80)

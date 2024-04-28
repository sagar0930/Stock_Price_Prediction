import tensorflow.keras as keras
import pandas as pd
#from data_collection import get_news_data, get_stock_price
from data_processing_and_features import news_sentiment, merging_df, features_handling, missing_value_imputation, final_data
from model_building import data_scaling, create_sequences, train_test_split,fit_and_evaluate_model, errors
import os
os.chdir("C://Users/Admin/Downloads/Data Science/Session_41_DS_Project_Structure/DS1/Stock_Price_Prediction")

#df_news = get_news_data()
#df_price = get_stock_price()

df_news = pd.read_csv('news_data.csv')
df_price = pd.read_csv('stock_price.csv')

df_news = news_sentiment(df_news)
merged_df = merging_df(df_news,df_price)
merged_df = features_handling(merged_df)
merged_df = missing_value_imputation(merged_df)
final_df = final_data(merged_df)
final_df.info()
print(final_df)


final_df, dataset, y_scaler = data_scaling(final_df)
x_train, y_train, x_test, y_test = train_test_split(dataset)
model,trainPredict,trainY = fit_and_evaluate_model(x_train, x_test, y_train, y_test, y_scaler)

mae,mape,rmse=errors(trainPredict[:,0], trainY[0])

print('mae:',mae,'mape:',mape,'rmse:',rmse)


model.save('stock_price_prediction.h5')



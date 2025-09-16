import torch
from FetchGoldDataClass import FetchGoldDataClass
from LSTMModelClass import ProfessionalLSTM 
def main():
    print("\n"+ "*" *15 + "FETCHING GLD TICKER" + "*" *15 +"\n")

    fgdc = FetchGoldDataClass()
    df = fgdc.fetch_gld_hourly()    #--- fetch data 
    df = fgdc.calculate_enhanced_indicators(df)     #--- add indicators 
    df = fgdc.preprocess(df) #---   preprocess dataframe
    df = fgdc.scale_data(df)    #--- scale dataset

    X,y = fgdc.create_sequences(df)
    X_train, y_train, X_test, y_test = fgdc.train_test_split_sequences(X, y)

    pl = ProfessionalLSTM()
    model, train_pred, test_pred, y_train, y_test = pl.train_enhanced_model(
    X_train, y_train, X_test, y_test
    )
    torch.save(model.state_dict(), "lstm_model.pth")
    results = model.evaluate_and_plot(train_pred, y_train, test_pred, y_test)

    print("\n"+ "*" *15 + "FETCHING GLD TICKER" + "*" *15 +"\n")
    print(results)
    
    


if __name__ == "__main__":
    main()



"""

    RESULTS:
    ******CALCULATING RESULTS******

📊 Evaluation Metrics:
Train R²: -0.000253
Train MSE: 0.002501
Epoch [100/100] - Loss: 0.001081

******CALCULATING RESULTS******

📊 Evaluation Metrics:
Train R²: -0.000253
Epoch [100/100] - Loss: 0.001081

******CALCULATING RESULTS******

📊 Evaluation Metrics:
Epoch [100/100] - Loss: 0.001081

******CALCULATING RESULTS******
Epoch [100/100] - Loss: 0.001081

Epoch [100/100] - Loss: 0.001081
Epoch [100/100] - Loss: 0.001081

******CALCULATING RESULTS******

📊 Evaluation Metrics:
Train R²: -0.000253
Train MSE: 0.002501
Train MAE: 0.026863
Test R²: -0.000106
Test MSE: 0.000145
Test MAE: 0.006955

"""
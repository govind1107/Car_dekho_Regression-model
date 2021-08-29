from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('car_price_pred_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        featureDict = {}
        age = 2021 - int(request.form['year'])

        km_driven=int(request.form['km_driven'])
        
        mileage=float(request.form['mileage'])
        
        engine=float(request.form['engine'])
        
        max_power=float(request.form['max_power'])

        fuel_type = request.form['fuel_type']
        featureDict['fuel_CNG'] = featureDict['fuel_Diesel'] = featureDict['fuel_LPG'] = featureDict['fuel_Petrol'] = 0
        featureDict[fuel_type] = 1

        owner=request.form['owner']
        featureDict['owner_First'] = featureDict['owner_Second'] = featureDict['owner_Third'] = featureDict['owner_Test'] = featureDict['owner_FourthPlus'] = 0
        featureDict[owner] = 1

        seats=request.form['seats']
        featureDict['seats_4'] = featureDict['seats_5'] = featureDict['seats_6'] = featureDict['seats_7'] = featureDict['seats_8'] = featureDict['seats_9'] = featureDict['seats_10'] = featureDict['seats_14'] = 0
        featureDict[seats] = 1

        seller_type=request.form['seller_type']
        featureDict['seller_type_Dealer'] = featureDict['seller_type_Individual'] = featureDict['seller_type_Trustmark'] = 0
        featureDict[seller_type] = 1

        transmission=request.form['transmission']
        featureDict['transmission_Manual'] = featureDict['transmission_Automatic'] = 0
        featureDict[transmission] = 1

        company=request.form['company']
        featureDict['company_Ambassador'] = featureDict['company_Ashok'] = featureDict['company_Audi'] = featureDict['company_BMW'] = featureDict['company_Chevrolet'] = featureDict['company_Daewoo'] = featureDict['company_Datsun'] = featureDict['company_Fiat'] = featureDict['company_Force'] = featureDict['company_Ford'] = featureDict['company_Honda'] = featureDict['company_Hyundai'] = featureDict['company_Isuzu'] = featureDict['company_Jaguar'] = featureDict['company_Jeep'] = featureDict['company_Kia'] = featureDict['company_Land'] = featureDict['company_Lexus'] = featureDict['company_MG'] = featureDict['company_Mahindra'] = featureDict['company_Maruti'] = featureDict['company_Mercedes_Benz'] = featureDict['company_Mitsubishi'] = featureDict['company_Nissan'] = featureDict['company_Opel'] = featureDict['company_Renault'] = featureDict['company_Skoda'] = featureDict['company_Tata'] = featureDict['company_Toyota'] = featureDict['company_Volkswagen'] = featureDict['company_Volvo'] = 0
        featureDict[company] = 1

        featurelist = [km_driven, mileage, engine, max_power, age, featureDict['fuel_CNG'], featureDict['fuel_Diesel'], featureDict['fuel_LPG'], featureDict['fuel_Petrol'], featureDict['seller_type_Dealer'], featureDict['seller_type_Individual'], featureDict['seller_type_Trustmark'], featureDict['transmission_Automatic'], featureDict['transmission_Manual'], featureDict['owner_First'], featureDict['owner_FourthPlus'], featureDict['owner_Second'], featureDict['owner_Test'], featureDict['owner_Third'], featureDict['seats_10'], featureDict['seats_14'], featureDict['seats_4'], featureDict['seats_5'], featureDict['seats_6'], featureDict['seats_7'], featureDict['seats_8'], featureDict['seats_9'], featureDict['company_Ambassador'], featureDict['company_Ashok'], featureDict['company_Audi'], featureDict['company_BMW'], featureDict['company_Chevrolet'], featureDict['company_Daewoo'], featureDict['company_Datsun'], featureDict['company_Fiat'], featureDict['company_Force'], featureDict['company_Ford'], featureDict['company_Honda'], featureDict['company_Hyundai'], featureDict['company_Isuzu'], featureDict['company_Jaguar'], featureDict['company_Jeep'], featureDict['company_Kia'], featureDict['company_Land'], featureDict['company_Lexus'], featureDict['company_MG'], featureDict['company_Mahindra'], featureDict['company_Maruti'], featureDict['company_Mercedes_Benz'], featureDict['company_Mitsubishi'], featureDict['company_Nissan'], featureDict['company_Opel'], featureDict['company_Renault'], featureDict['company_Skoda'], featureDict['company_Tata'], featureDict['company_Toyota'], featureDict['company_Volkswagen'], featureDict['company_Volvo']]
        print("featureList: ", featurelist)
        featureList = np.array(featurelist)
        featureList = featureList.reshape((1, len(featureList)))
        prediction=model.predict(featureList)
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
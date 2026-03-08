from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():

    prediction=None
    recommendation=None
    alert=None

    temperature=0
    humidity=0
    wind=0
    pollution=0

    if request.method=="POST":

        temperature=float(request.form["temperature"])
        humidity=float(request.form["humidity"])
        wind=float(request.form["wind"])
        pollution=float(request.form["pollution"])

        data=np.array([[temperature,humidity,wind,pollution]])

        result=model.predict(data)[0]

        if result==0:
            prediction="Low Risk"
            recommendation="Environment conditions are stable."

        elif result==1:
            prediction="Medium Risk"
            recommendation="Monitor pollution levels and reduce emissions."

        else:
            prediction="High Risk"
            recommendation="Immediate action needed to control pollution."

        if pollution>200:
            alert="⚠ Pollution Alert! Air quality is dangerous."

    return render_template("index.html",
                           prediction=prediction,
                           recommendation=recommendation,
                           alert=alert,
                           temperature=temperature,
                           humidity=humidity,
                           wind=wind,
                           pollution=pollution)

if __name__=="__main__":
    app.run(debug=True)
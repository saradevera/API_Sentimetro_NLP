from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
# import numeritos as nitos
from functions import *
from os import environ


# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return render_template('hola.html')

@app.route('/', methods=['POST'])
def texto():
    text = pd.Series(request.form['text'])

    # PROCESAMOS EL TEXTO DE LA MISMA MANERA QUE ESTÁ PROCESADO EN EL MODELO
    text = text.apply(signs_tweets)
    text = text.apply(remove_links)
    # text = text.apply(remove_stopwords)
    text = text.apply(spanish_stemmer)
    # text = nitos.clean_emoji(text[0])

    # CARGAMOS EL MODELO
    my_model = pickle.load(open('model/sentiment_model', 'rb')) # SE PUEDE METER EN LA MISMA LINEA

    # SACAMOS LAS PREDICCIONES
    text_pred = pd.Series("'" + text + "'")
    prediction = my_model.predict(text_pred)

    # CONVERTIMOS EL RESULTADO DE LA PREDICCIÓN EN SU ETIQUETA CORRESPONDIENTE

    if prediction[0] == 1:
        resultado = 'Negativo \U0001F641'

    elif prediction[0] == 0:
        resultado = 'Positivo \U0001F600'

    else:
        print('La predicción no se ha hecho correctamente')

    return 'El sentimiento que produce este texto es: ' + str(resultado)


if __name__ == '__main__':
  app.run(debug = True, host = '0.0.0.0', port=environ.get("PORT", 5000))
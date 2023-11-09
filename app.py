from flask import Flask, redirect , url_for, render_template

# Crea un'applicazione Flask
app = Flask(__name__)

# Definisci una route di base
@app.route('/')
def home():
    return render_template("index.html")
    



if __name__ == '__main__':
    app.run(debug = True )

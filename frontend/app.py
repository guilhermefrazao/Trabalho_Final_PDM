from flask import Flask, render_template, jsonify

app = Flask(__name__)

Movies = [
    {"titulo": "O Poderoso Chefão", "genero": "Drama", "nota": 9.2},
    {"titulo": "Matrix", "genero": "Ficção Científica", "nota": 8.7},
    {"titulo": "Toy Story", "genero": "Animação", "nota": 8.3},
    {"titulo": "O Cavaleiro das Trevas", "genero": "Ação", "nota": 9.0},
    {"titulo": "Parasita", "genero": "Suspense", "nota": 8.6},
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/filmes")
def list_movies():
    return jsonify(Movies)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
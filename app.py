from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET']) #slash is homepage
def index():
    return render_template('index.html')

@app.route('/post', methods=['POST']) #make request to backend (call api , etc.)
def post():
    return "recived: {}".format(request.form)

if __name__ == "__main__":
    app.run(debug=True)


#insert LLM and make req
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])


@app.route('/')
def index():
    return render_template('content.html')



# @app.route('/')
# def index():
#     return render_template('table.html', data=zip(x,y))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)



from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        book = request.form['query']
        with open('./resources/abstracts.txt', 'r') as f:
            data = f.read().split('\n\n')
            abstracts = [(i.split('.')[0], '.'.join(i.split('.')[1:])) for i in data]
        data = abstracts[:3]
        return render_template('search.html', data=data)
    return render_template('search.html')

if __name__ == '__main__':
    app.debug = True
    app.run()

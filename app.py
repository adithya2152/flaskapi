from flask import Flask, render_template, request, redirect, url_for, session, Response
from driver_monitor import register, login, generate_frames  # replace with correct filename
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key'
alerts = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register_route():
    if request.method == 'POST':
        name = request.form['username'].strip()
        if not name:
            return "Name is required", 400
        register(name)
        # Pass a success message as a query parameter
        return redirect(url_for('register_route', success=1))
    # Get the success flag from query params
    success = request.args.get('success')
    return render_template('register.html', success=success)

@app.route('/login', methods=['GET', 'POST'])
def login_route():
    if request.method == 'POST':
        user = login()  # OpenCV face match
        if user:
            session['username'] = user
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Face not recognized. Try again.")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return "Unauthorized", 401
    return Response(generate_frames(alerts, session['username']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    return '<br>'.join(alerts[-10:])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

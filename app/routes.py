from app import app, db
from flask import render_template, flash, redirect, request, url_for
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User, Query
from app.forms import RegistrationForm, LoginForm, EditProfileForm, QueryForm
from werkzeug.urls import url_parse
from datetime import datetime
from data_modules.data import result_df, r_squared
from loaded_models import predict

@app.route('/')
@app.route('/index')

def index():
    user = current_user
    return render_template('index.html', title='Home', user=user)

@login_required

@app.route('/login', methods=['GET', 'POST'])

def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')

def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])

def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.before_request

def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required

def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile', form=form)

@app.route('/query', methods=['GET', 'POST'])
@login_required

def query():
    form = QueryForm()
    if form.validate_on_submit():
        query = Query(open=form.open.data, high=form.high.data, low=form.low.data, headline=form.headline.data)
        db.session.add(query)
        db.session.commit()
        flash('Your query has been saved!')
        return redirect(url_for('prediction'))
    return render_template('query.html', title='Make a Prediction', form=form)

@app.route('/prediction', methods=['GET'])
@login_required

def prediction():
    query = Query.query.order_by(Query.id.desc()).first()
    predictions = {}
    predictions['close1'], predictions['close2'] = predict(query.open, query.high, query.low, query.headline, result_df)
    return render_template('prediction.html', title='Made a Prediction!', query=query, predictions=predictions)

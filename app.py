from flask import Flask, render_template, request,flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

app.config["SECRET_KEY"] = "your_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///expense_tracker.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db=SQLAlchemy(app)

# Login manager setup
login_manager = LoginManager(app)
login_manager.login_view = "login"  # redirects unauthorized users to /login

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)
        
    
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False)
    category = db.Column(db.String(80), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=True)

    user = db.relationship('User', backref=db.backref('expenses', lazy=True))


class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(80), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'category', name='_user_category_uc'),)




@app.route('/')
def home():
    return render_template('index.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("pass")

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash("Invalid credentials.", "danger")
            return redirect(url_for("login"))

        login_user(user)
        flash("Logged in successfully.", "success")
        # next_page = request.args.get("next")
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        username = request.form.get("username")
        password = request.form.get("pass")
        conf = request.form.get("conf-pass")

        if not name or not username or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if password != conf:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username is already taken.", "warning")
            return redirect(url_for("register"))

        user = User(name=name, username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash("Registration successful.", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html")

# Logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("home"))



@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_expense():
    if request.method == 'POST':
        date_str = request.form.get('date')
        category = request.form.get('category')
        amount = request.form.get('amount')
        description = request.form.get('description', '').strip()

        if not date_str or not category or not amount:
            flash('All fields are required.', 'danger')
            return redirect(url_for('add_expense'))

        try:
            exp_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            amount_val = float(amount)
        except Exception:
            flash('Invalid input. Check date and amount formats.', 'danger')
            return redirect(url_for('add_expense'))

        expense = Expense(user_id=current_user.id, date=exp_date, category=category, amount=amount_val, description=description)
        db.session.add(expense)
        db.session.commit()

        flash('Expense added successfully.', 'success')
        return redirect(url_for('view_expenses'))

    return render_template('add_expense.html')


@app.route('/view')
@login_required
def view_expenses():
    expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date.desc()).all()
    return render_template('view_expenses.html', expenses=expenses)


@app.route('/dashboard')
@login_required
def dashboard():
    # Gather all expenses for this user
    expenses = Expense.query.filter_by(user_id=current_user.id).all()

    # Daily spending for last 7 days
    today = date.today()
    daily_labels = []
    daily_values = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        daily_labels.append(d.strftime('%Y-%m-%d'))
        total = sum(e.amount for e in expenses if e.date == d)
        daily_values.append(round(total, 2))

    # Monthly expenses (last 6 months, includes current month)
    monthly_labels = []
    monthly_values = []
    months = []
    for offset in range(5, -1, -1):
        m = today.month - offset
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        months.append((y, m))

    for y, m in months:
        monthly_labels.append(f"{y}-{m:02d}")
        total = sum(e.amount for e in expenses if e.date.year == y and e.date.month == m)
        monthly_values.append(round(total, 2))

    # Category breakdown
    cat_map = {}
    for e in expenses:
        cat_map[e.category] = cat_map.get(e.category, 0) + e.amount
    category_labels = list(cat_map.keys())
    category_values = [round(cat_map[k], 2) for k in category_labels]

    
    # --- Prediction Logic ---
    prediction, pred_msg = get_prediction(current_user.id)

    # --- Budget Logic ---
    budgets = Budget.query.filter_by(user_id=current_user.id).all()
    budget_report = []
    
    # Get current month expenses per category
    current_month_expenses = [e for e in expenses if e.date.year == today.year and e.date.month == today.month]
    curr_cat_map = {}
    for e in current_month_expenses:
        curr_cat_map[e.category] = curr_cat_map.get(e.category, 0) + e.amount

    for b in budgets:
        spent = curr_cat_map.get(b.category, 0)
        percent = 0
        if b.amount > 0:
            percent = (spent / b.amount) * 100
        
        status_color = "success"
        if percent > 90:
            status_color = "danger"
        elif percent > 75:
            status_color = "warning"
            
        budget_report.append({
            'category': b.category,
            'limit': b.amount,
            'spent': spent,
            'percent': min(percent, 100),
            'display_percent': min(percent, 100),
            'status_color': status_color,
            'is_over': (spent > b.amount)
        })


    return render_template('dashboard.html',
                           daily_labels=daily_labels,
                           daily_values=daily_values,
                           monthly_labels=monthly_labels,
                           monthly_values=monthly_values,
                           category_labels=category_labels,
                           category_values=category_values,
                           prediction=prediction,
                           pred_msg=pred_msg,
                           budget_report=budget_report)

@app.route('/budget', methods=['GET', 'POST'])
@login_required
def manage_budget():
    if request.method == 'POST':
        category = request.form.get('category')
        amount = request.form.get('amount')
        
        if not category or not amount:
            flash("All fields required", "danger")
            return redirect(url_for('manage_budget'))

        try:
            amount_val = float(amount)
        except:
            flash("Invalid amount", "danger")
            return redirect(url_for('manage_budget'))
            
        # Check if budget exists
        budget = Budget.query.filter_by(user_id=current_user.id, category=category).first()
        if budget:
            budget.amount = amount_val
            flash(f"Updated budget for {category}.", "success")
        else:
            budget = Budget(user_id=current_user.id, category=category, amount=amount_val)
            db.session.add(budget)
            flash(f"Set budget for {category}.", "success")
        
        db.session.commit()
        return redirect(url_for('dashboard'))

    # Show current budgets
    budgets = Budget.query.filter_by(user_id=current_user.id).all()
    return render_template('add_budget.html', budgets=budgets)


def get_prediction(user_id):
    expenses = Expense.query.filter_by(user_id=user_id).all()
    
    if not expenses:
        return 0.0, "Not enough data."

    data = []
    for e in expenses:
        month_id = e.date.year * 12 + e.date.month
        data.append({'month_id': month_id, 'amount': e.amount})
    
    df = pd.DataFrame(data)
    monthly_data = df.groupby('month_id')['amount'].sum().reset_index()
    
    if len(monthly_data) < 2:
        return 0.0, "Need 2+ months of data."

    X = monthly_data[['month_id']]
    y = monthly_data['amount']

    model = LinearRegression()
    model.fit(X, y)

    next_month_id = monthly_data['month_id'].max() + 1
    pred_val = model.predict([[next_month_id]])[0]
    return max(0, round(pred_val, 2)), "Based on spending trend."


@app.route('/edit/<int:exp_id>', methods=['GET', 'POST'])
@login_required
def edit_expense(exp_id):
    expense = Expense.query.get_or_404(exp_id)
    if expense.user_id != current_user.id:
        flash('Unauthorized.', 'danger')
        return redirect(url_for('view_expenses'))

    if request.method == 'POST':
        date_str = request.form.get('date')
        category = request.form.get('category')
        amount = request.form.get('amount')
        description = request.form.get('description', '').strip()

        if not date_str or not category or not amount:
            flash('All fields are required.', 'danger')
            return redirect(url_for('edit_expense', exp_id=exp_id))

        try:
            exp_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            amount_val = float(amount)
        except Exception:
            flash('Invalid input. Check date and amount formats.', 'danger')
            return redirect(url_for('edit_expense', exp_id=exp_id))

        expense.date = exp_date
        expense.category = category
        expense.amount = amount_val
        expense.description = description

        db.session.commit()
        flash('Expense updated successfully.', 'success')
        return redirect(url_for('view_expenses'))

    return render_template('edit_expense.html', expense=expense)


@app.route('/delete/<int:exp_id>', methods=['POST'])
@login_required
def delete_expense(exp_id):
    expense = Expense.query.get_or_404(exp_id)
    if expense.user_id != current_user.id:
        flash('Unauthorized.', 'danger')
        return redirect(url_for('view_expenses'))

    db.session.delete(expense)
    db.session.commit()
    flash('Expense deleted.', 'info')
    return redirect(url_for('view_expenses'))


@app.route('/predict')
@login_required
def predict_expense():
    # Deprecated: Redirecting to dashboard now
    return redirect(url_for('dashboard'))



if __name__ == "__main__":
    # Ensure database tables exist
    with app.app_context():
        db.create_all()

    app.run(debug=True)
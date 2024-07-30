from landslide import create_app, db  
from landslide.models import User  

app = create_app()

# Use the app context to interact with the database
with app.app_context():
    users = User.query.all()
    for user in users:
        print(f"ID: {user.id}, Email: {user.email}, First Name: {user.first_name}")

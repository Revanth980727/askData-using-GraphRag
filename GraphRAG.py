from sqlalchemy import create_engine, MetaData, text
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'
DB_NAME = 'your_database_name'
+ DB_CONNECTION_STRING = f'postgresql://{DB_USER}:{DB_PASSWORD}@localhost/{DB_NAME}'
+ engine = create_engine(DB_CONNECTION_STRING)
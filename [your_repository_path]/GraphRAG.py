from sqlalchemy import create_engine, MetaData, text
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'
DB_NAME = 'your_database_name'
+ DB_HOST = 'your_host'  # Add your database host
+ DB_PORT = 'your_port'  # Add your database port, typically 5432 for PostgreSQL or 3306 for MySQL
- 
+ engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
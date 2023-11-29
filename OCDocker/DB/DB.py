from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import create_database, database_exists
from sqlalchemy.orm import sessionmaker

# Define the Base class
Base = declarative_base()

# Generates the database
def generate(url, Base):
    # Check if the database exists
    if not database_exists(url):
        # Create the database
        create_database(url)

    # Connect to the database
    engine = create_engine(url, echo = True)

    # Create the table in the database
    Base.metadata.create_all(engine)

# Inserts data 
def insert(url, table, payload):
    engine = create_engine(url, echo = True)

    Session = sessionmaker(bind=engine)

    with Session() as session:
        new_data = table(**payload)

        session.add(new_data)

        session.commit()

        session.close()


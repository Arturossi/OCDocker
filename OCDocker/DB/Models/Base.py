from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

class Base(declarative_base()):
    """ Base class for all the tables. """
        
    __abstract__ = True

    def __repr__(self) -> str:
        ''' Return the representation of the object. '''
        
        return f"<{self.__class__.__name__}({self.__dict__})>"

    # Set the id column as the primary key
    id = Column(Integer, primary_key = True)

    # Add created_at and modified_at columns (modified_at is updated automatically)
    created_at = Column(DateTime, server_default = func.now())
    modified_at = Column(DateTime, server_default = None, onupdate = func.now())

    # Add a column for the molecule name
    name = Column(String(2048))

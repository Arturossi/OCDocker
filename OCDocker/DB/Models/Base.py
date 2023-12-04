from sqlalchemy import Column, DateTime, Float, Index, Integer, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declared_attr
from typing import Any, Dict, List, Union

class Base(declarative_base()):
    """ Base class for all the tables. """
    
    # Set the table name
    @declared_attr
    def __tablename__(cls):
        ''' Return the table name. '''
        return cls.__name__.lower()

    ## Class Attributes ##

    # Set the abstract flag
    __abstract__ = True

    # Set the extend existing flag to true (to avoid errors when creating the tables)
    __table_args__ = {
        'extend_existing': True
    }

    # Set the id column as the primary key
    id = Column(Integer, primary_key = True)

    # Add created_at and modified_at columns (modified_at is updated automatically)
    created_at = Column(DateTime, server_default = func.now())
    modified_at = Column(DateTime, server_default = None, onupdate = func.now())

    # Add a column for the molecule name (the size of the name is 760 characters to allow proper indexing) Names are supposed to be unique!
    name = Column(String(760), index=True)


    ## Private Methods ##

    def __repr__(self) -> str:
        ''' Return the representation of the object. 
        
        Returns
        -------
        str
            The representation of the object.
        '''
        
        # Get the data of the object (without the private attributes)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

        # Return the representation
        return f"<{self.__class__.__name__}({data})>"
    

    ## Public Methods ##

    def to_dict(self) -> Dict[str, Any]:
        ''' Return the object as a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            The object as a dictionary.
        '''

        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


    ## Class Methods ##

    @classmethod
    def determine_column_type(cls, descriptor: str) -> Union[Integer, Float]:
        ''' Determine the type of column based on the descriptor. 

        Parameters
        ----------
        descriptor : str
            The descriptor name.

        Returns
        -------
        Integer | Float
            The type of the column.
        ''' 

        # Check if the descriptor is an integer, if not, it is a float
        if descriptor.startswith("fr_") or \
           descriptor.startswith("Num") or \
           descriptor.startswith("count") or \
           descriptor in ["HeavyAtomCount", "NHOHCount", "NOCount", "RingCount", "TotalAALength"]:
            return Integer
        return Float

    @classmethod
    def add_dynamic_columns(cls, collection: List[str]) -> None:
        ''' Dynamically add columns based on descriptor names. 
        
        Parameters
        ----------
        collection : List[str]
            The collection of descriptors.
        '''

        # Iterate over the descriptors
        for descriptor in collection:
            # Determine the type of the column
            column_type = cls.determine_column_type(descriptor)

            # Set the column as an attribute of the class using the descriptor name as the attribute name and setting the type of the column based on the descriptor name
            setattr(cls, descriptor, Column(column_type, server_default = None))

        return None

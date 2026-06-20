#!/usr/bin/env python3

# Description
###############################################################################
"""
Base class for all the tables in the database.

Usage:

from OCDocker.DB.Models.Base import base
"""

# Imports
###############################################################################
from operator import eq, ge, gt, le, lt, ne

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, and_, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase, declared_attr

from typing import Any, Dict, Iterator, List, Optional, Union, cast

import OCDocker.Error as ocerror

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################

# Default session factory fallback. Tests and explicit bootstrap can set this,
# while unbootstrapped imports remain side-effect-free.
session: Any = None


def _resolve_session(session_factory: Any = None) -> Any:
    if session_factory is not None:
        return session_factory
    if session is not None:
        return session

    try:
        from OCDocker.DB.DBMinimal import get_default_session
    except ImportError:  # pragma: no cover
        return None
    return get_default_session()


OPMAP = {
    "==": lambda c, v: c == v,
    "!=": lambda c, v: c != v,
    ">": lambda c, v: c > v,
    ">=": lambda c, v: c >= v,
    "<": lambda c, v: c < v,
    "<=": lambda c, v: c <= v,
    "like": lambda c, v: c.like(v),
    "ilike": lambda c, v: c.ilike(v),
    "in": lambda c, v: c.in_(v if isinstance(v, (list, tuple, set)) else [v]),
}


class Base(DeclarativeBase):
    """Base class for all the tables."""

    @classmethod
    def __repr__(cls) -> str:
        """Return the representation of the object.

        Returns
        -------
        str
            The representation of the object.
        """

        # Get the data of the object (without the private attributes)
        data = {k: v for k, v in cls.__dict__.items() if not k.startswith("_")}

        # Return the representation
        return f"<{cls.__name__}({data})>"

    # Set the table name
    @declared_attr.directive
    def __tablename__(cls):
        """Return the table name."""

        return cls.__name__.lower()

    ## Class Attributes ##

    # Set the abstract flag
    __abstract__ = True

    # Set the extend existing flag to true (to avoid errors when creating the tables)
    __table_args__ = {"extend_existing": True}

    # Set the id column as the primary key
    id = Column(Integer, primary_key=True)

    # Add created_at and modified_at columns (modified_at is updated automatically)
    created_at = Column(DateTime, server_default=func.now())
    modified_at = Column(DateTime, server_default=None, onupdate=func.now())

    # Add a column for the molecule name (size 760 to allow indexing). Uniqueness
    # already enforces an index in most backends, so we avoid setting index=True
    # to prevent duplicate index DDL on SQLite (ix_<table>_name).
    name = Column(String(760), unique=True, nullable=False)

    ## Class Methods ##

    @classmethod
    def add_dynamic_columns(cls, collection: List[str]) -> None:
        """Dynamically add columns based on descriptor names.

        Parameters
        ----------
        collection : List[str]
            The collection of descriptors.
        """

        # Iterate over the descriptors
        for descriptor in collection:
            # Determine the type of the column
            column_type = cls.determine_column_type(descriptor)

            # If the column type is Integer, and the descriptor is a count, set the default value to 0
            if isinstance(column_type, Integer) and (
                descriptor.lower().startswith("count")
                or descriptor.lower().startswith("fr")
                or descriptor.lower().startswith("num")
                or descriptor.lower().endswith("count")
                or descriptor.lower().endswith("num")
            ):
                setattr(cls, descriptor, Column(column_type, server_default="0"))
            else:
                # Set the column as an attribute of the class using the descriptor name as the attribute name and setting the type of the column based on the descriptor name
                setattr(cls, descriptor, Column(column_type, server_default=None))

        return None

    @classmethod
    def delete(cls, idorname: Union[int, str], session: Any = None) -> bool:
        """Delete data from the database.

        Parameters
        ----------
        idorname : Union[int, str]
            The ID or name of the data to be deleted.

        Returns
        -------
        bool
            True if the data was deleted, False otherwise.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return False
            return False

        # Open the session
        with db_session() as s:
            # Perform the search
            data = (
                s.query(cls).filter(cls.id == idorname).first()
                if isinstance(idorname, int)
                else s.query(cls)
                .filter(func.lower(cls.name) == func.lower(idorname))
                .first()
            )

            # Check if the data exists
            if data is None:
                # The data does not exist
                _ = ocerror.Error.data_not_found("The data does not exist.")
                # Return False
                return False

            try:
                # Delete the data
                s.delete(data)

                # Commit the session
                s.commit()
            except SQLAlchemyError as e:
                # Rollback the session
                s.rollback()
                # Print the error
                print(f"Error: {e}")

                # Return False
                return False

        return True

    @classmethod
    def determine_column_type(cls, descriptor: str) -> Union[Integer, Float]:
        """Determine the type of column based on the descriptor.

        Parameters
        ----------
        descriptor : str
            The descriptor name.

        Returns
        -------
        Integer | Float
            The type of the column.
        """

        # Check if the descriptor is an integer-like count; otherwise use float
        if (
            descriptor.startswith("fr_")
            or descriptor.startswith("Num")
            or descriptor.startswith("count")
            or descriptor
            in ["HeavyAtomCount", "NHOHCount", "NOCount", "RingCount", "TotalAALength"]
        ):
            return Integer()

        return Float()

    @classmethod
    def find_iter(
        cls, idorname: Union[int, str], batch_size: int = 1000, session: Any = None
    ) -> Iterator[DeclarativeMeta]:
        """Search data in the database and stream the rows.

        Parameters
        ----------
        idorname : Union[int, str]
            The ID or name of the data to be searched.
        batch_size : int, optional
            Number of rows per fetch batch.

        Returns
        -------
        Iterator[DeclarativeMeta]
            Iterator with matching rows.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return an empty iterator
            return iter(())

        # Ensure a valid positive batch size
        normalized_batch_size = max(1, int(batch_size))

        def _iter_rows() -> Iterator[DeclarativeMeta]:
            # Open the session
            with db_session() as s:
                # Build the query
                if isinstance(idorname, int):
                    query = s.query(cls).filter(cls.id == idorname)
                else:
                    query = s.query(cls).filter(
                        func.lower(cls.name) == func.lower(idorname)
                    )

                # Stream results to avoid materializing large result sets in memory
                for row in query.execution_options(stream_results=True):
                    yield cast(DeclarativeMeta, row)

        return _iter_rows()

    @classmethod
    def find(
        cls, idorname: Union[int, str], session: Any = None
    ) -> List[DeclarativeMeta]:
        """Search data in the database.

        Parameters
        ----------
        idorname : Union[int, str]
            The ID or name of the data to be searched.

        Returns
        -------
        List[DeclarativeMeta]
            The data found.
        """

        return list(cls.find_iter(idorname, session=session))

    @classmethod
    def find_all_iter(
        cls, batch_size: int = 1000, session: Any = None
    ) -> Iterator[DeclarativeMeta]:
        """Search all data in the database and stream the rows.

        Parameters
        ----------
        batch_size : int, optional
            Number of rows per fetch batch.

        Returns
        -------
        Iterator[DeclarativeMeta]
            Iterator with all rows.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return an empty iterator
            return iter(())

        # Ensure a valid positive batch size
        normalized_batch_size = max(1, int(batch_size))

        def _iter_rows() -> Iterator[DeclarativeMeta]:
            # Open the session
            with db_session() as s:
                query = s.query(cls)
                for row in query.execution_options(stream_results=True):
                    yield cast(DeclarativeMeta, row)

        return _iter_rows()

    @classmethod
    def find_all(cls, session: Any = None) -> List[DeclarativeMeta]:
        """Search all data in the database.

        Returns
        -------
        List[DeclarativeMeta]
            The data found.
        """

        return list(cls.find_all_iter(session=session))

    @classmethod
    def find_all_names_iter(
        cls, batch_size: int = 1000, session: Any = None
    ) -> Iterator[str]:
        """Search all names in the database and stream the rows.

        Parameters
        ----------
        batch_size : int, optional
            Number of rows per fetch batch.

        Returns
        -------
        Iterator[str]
            Iterator with all names.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return an empty iterator
            return iter(())

        # Ensure a valid positive batch size
        normalized_batch_size = max(1, int(batch_size))

        def _iter_rows() -> Iterator[str]:
            # Open the session
            with db_session() as s:
                query = s.query(cls.name)
                for row in query.execution_options(stream_results=True).yield_per(
                    normalized_batch_size
                ):
                    yield cast(str, row[0])

        return _iter_rows()

    @classmethod
    def find_all_names(cls, session: Any = None) -> List[str]:
        """Search all names in the database.

        Returns
        -------
        List[str]
            The names found.
        """

        return list(cls.find_all_names_iter(session=session))

    @classmethod
    def find_attribute_iter(
        cls,
        column: str,
        value: Any,
        operator: str = "==",
        batch_size: int = 1000,
        session: Any = None,
    ) -> Iterator[DeclarativeMeta]:
        """Search data in the database based on an attribute and stream rows.

        Parameters
        ----------
        column : str
            The column name.
        value : Any
            The value to be searched.
        operator : str
            The operator to be used.
        batch_size : int, optional
            Number of rows per fetch batch.

        Returns
        -------
        Iterator[DeclarativeMeta]
            Iterator with matching rows.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )
            return iter(())

        # Check if the operator is valid
        if operator not in OPMAP:
            _ = ocerror.Error.malformed_payload(f"Unsupported operator '{operator}'.")
            return iter(())

        # Get the column
        try:
            col = getattr(cls, column)
        except AttributeError:
            _ = ocerror.Error.malformed_payload(f"Unknown column '{column}'.")
            return iter(())

        # Ensure a valid positive batch size
        normalized_batch_size = max(1, int(batch_size))

        def _iter_rows() -> Iterator[DeclarativeMeta]:
            # Open the session
            with db_session() as s:
                query = s.query(cls).filter(OPMAP[operator](col, value))
                for row in query.execution_options(stream_results=True):
                    yield cast(DeclarativeMeta, row)

        return _iter_rows()

    @classmethod
    def find_attribute(
        cls, column: str, value: Any, operator: str = "==", session: Any = None
    ) -> List[DeclarativeMeta]:
        """Search data in the database based on an attribute.

        Parameters
        ----------
        column : str
            The column name.
        value : Any
            The value to be searched.
        operator : str
            The operator to be used.

        Returns
        -------
        List[DeclarativeMeta]
            The data found.
        """

        return list(cls.find_attribute_iter(column, value, operator, session=session))

    @classmethod
    def find_first(
        cls, idorname: Union[int, str], session: Any = None
    ) -> Optional[DeclarativeMeta]:
        """Search data in the database.

        Parameters
        ----------
        idorname : Union[int, str]
            The ID or name of the data to be searched.

        Returns
        -------
        DeclarativeMeta | None
            The first matching row, if any.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return an empty list
            return None

        # Open the session
        with db_session() as s:
            # Perform the search
            data = (
                s.query(cls).filter(cls.id == idorname).first()
                if isinstance(idorname, int)
                else s.query(cls)
                .filter(func.lower(cls.name) == func.lower(idorname))
                .first()
            )

        return cast(Optional[DeclarativeMeta], data)

    @classmethod
    def insert(
        cls, payload: dict, ignorePresence: bool = False, session: Any = None
    ) -> bool:
        """Insert data into the database.

        Parameters
        ----------
        payload : dict
            The data to be inserted.
        ignorePresence : bool
            Whether to ignore the presence of the data in the database.

        Returns
        -------
        bool
            True if the data was inserted, False otherwise.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return False
            return False

        # Check if the payload has the name key
        if "name" not in payload:
            # The payload does not have the name key
            _ = ocerror.Error.malformed_payload(
                "The payload does not have the name key."
            )

            # Return False
            return False

        # Open the session
        with db_session() as s:
            # Check if the data already exists
            if (
                s.query(cls)
                .filter(func.lower(cls.name) == func.lower(payload["name"]))
                .first()
                is not None
            ):
                # If the ignorePresence flag is set to True, return True
                if ignorePresence:
                    return True

                # The data already exists
                _ = ocerror.Error.data_already_exists(
                    f"The data with name '{payload['name']}' already exists."
                )

                # Return False
                return False

            # Create the new data
            new_data = cls(**payload)

            try:
                # Add the new data to the session
                s.add(new_data)
                # Commit the session
                s.commit()
            except SQLAlchemyError as e:
                # Rollback the session
                s.rollback()
                # Print the error
                print(f"Error: {e}")
                # Return False

                return False

        return True

    @classmethod
    def insert_or_update(cls, payload: dict, session: Any = None) -> bool:
        """Insert or update data in the database.

        Parameters
        ----------
        payload : dict
            The data to be inserted or updated.

        Returns
        -------
        bool
            True if the data was inserted or updated, False otherwise.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return False
            return False

        # Open the session
        with db_session() as s:
            # If the payload have an id
            if "id" in payload:
                data = s.query(cls).filter(cls.id == payload["id"]).first()
            # Use name
            else:
                data = (
                    s.query(cls)
                    .filter(func.lower(cls.name) == func.lower(payload["name"]))
                    .first()
                )

            # Check if the data exists
            if data is None:
                # The data does not exist
                # Insert the data
                return cls.insert(payload, session=session)

            try:
                # Update the data
                for key, value in payload.items():
                    setattr(data, key, value)

                # Commit the session
                s.commit()
            except SQLAlchemyError as e:
                # Rollback the session
                s.rollback()

                # Print the error
                print(f"Error: {e}")
                # Return False
                return False

        return True

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Return the object as a dictionary.

        Returns
        -------
        Dict[str, Any]
            The object as a dictionary.
        """

        return {
            column.key: getattr(cls, column.key)
            for column in inspect(cls).attrs
            if not column.key.startswith("_")
        }

    @classmethod
    def update(
        cls, idorname: Union[int, str], payload: dict, session: Any = None
    ) -> bool:
        """Update data in the database.

        Parameters
        ----------
        idorname : Union[int, str]
            The ID or name of the data to be updated.
        payload : dict
            The data to be updated.

        Returns
        -------
        bool
            True if the data was updated, False otherwise.
        """

        # Check if session is defined
        db_session = _resolve_session(session)
        if db_session is None:
            # The session is not defined
            _ = ocerror.Error.session_not_created(
                "The session is not defined. Please create the session first."
            )

            # Return False
            return False

        # Open the session
        with db_session() as s:
            # Perform the search
            data = (
                s.query(cls).filter(cls.id == idorname).first()
                if isinstance(idorname, int)
                else s.query(cls)
                .filter(func.lower(cls.name) == func.lower(idorname))
                .first()
            )

            # Check if the data exists
            if data is None:
                # The data does not exist
                _ = ocerror.Error.data_not_found("The data does not exist.")

                # Return False
                return False

            try:
                # Update the data
                for key, value in payload.items():
                    setattr(data, key, value)

                # Commit the session
                s.commit()
            except SQLAlchemyError as e:
                # Rollback the session
                s.rollback()
                # Print the error

                print(f"Error: {e}")
                # Return False
                return False

        return True


base = Base

# Functions
###############################################################################
## Private ##

## Public ##

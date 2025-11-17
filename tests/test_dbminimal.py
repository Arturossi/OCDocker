import sys
import types
import importlib.util
from pathlib import Path
import pytest

# Dummy classes and functions to stand in for SQLAlchemy
class DummyEngine:
    pass


class DummyURL:
    @classmethod
    def create(cls, drivername, database=None):
        inst = cls()
        inst.drivername = drivername # type: ignore
        inst.database = database # type: ignore
        return inst


def dummy_create_engine(url, echo=False, pool_size=None, max_overflow=None, pool_timeout=None, pool_recycle=None, pool_pre_ping=None, **kwargs):
    return DummyEngine()


def dummy_sessionmaker(bind=None):
    def factory():
        return object()

    return factory


class DummyScopedSession:
    def __init__(self, factory):
        self.factory = factory


@pytest.fixture()
def dbminimal(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "OCDocker"))

    sqlalchemy = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    engine_base = types.ModuleType("sqlalchemy.engine.base")
    engine_base.Engine = DummyEngine # type: ignore
    engine_url = types.ModuleType("sqlalchemy.engine.url")
    engine_url.URL = DummyURL # type: ignore
    engine_url.make_url = lambda url: url if isinstance(url, DummyURL) else DummyURL.create("sqlite", url) # type: ignore
    engine_mod.base = engine_base # type: ignore
    engine_mod.url = engine_url # type: ignore
    sqlalchemy.engine = engine_mod # type: ignore
    sqlalchemy.create_engine = dummy_create_engine # type: ignore

    orm_mod = types.ModuleType("sqlalchemy.orm")
    orm_mod.sessionmaker = dummy_sessionmaker # type: ignore
    orm_mod.scoped_session = DummyScopedSession # type: ignore
    sqlalchemy.orm = orm_mod # type: ignore

    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", engine_mod)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine.base", engine_base)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine.url", engine_url)
    monkeypatch.setitem(sys.modules, "sqlalchemy.orm", orm_mod)

    sqla_utils = types.ModuleType("sqlalchemy_utils")
    
    def dummy_create_database(url):
        dummy_create_database.called = True

    dummy_create_database.called = False
    sqla_utils.create_database = dummy_create_database # type: ignore
    sqla_utils.database_exists = lambda url: True # type: ignore
    monkeypatch.setitem(sys.modules, "sqlalchemy_utils", sqla_utils)

    # Ensure package hierarchy exists for import
    pkg = sys.modules.get("OCDocker", types.ModuleType("OCDocker"))
    pkg.__path__ = [str(project_root / "OCDocker")]
    sys.modules["OCDocker"] = pkg
    db_pkg = sys.modules.get("OCDocker.DB", types.ModuleType("OCDocker.DB"))
    db_pkg.__path__ = [str(project_root / "OCDocker" / "DB")]
    sys.modules["OCDocker.DB"] = db_pkg

    db_file = project_root / "OCDocker" / "DB" / "DBMinimal.py"
    spec = importlib.util.spec_from_file_location("OCDocker.DB.DBMinimal", db_file)
    module = importlib.util.module_from_spec(spec) # type: ignore
    sys.modules["OCDocker.DB.DBMinimal"] = module
    assert spec.loader is not None # type: ignore
    spec.loader.exec_module(module) # type: ignore
    return module


@pytest.mark.order(50)
def test_create_engine_returns_engine(dbminimal):
    dbm = dbminimal
    url = dbm.URL.create("sqlite+pysqlite", database=":memory:")
    engine = dbm.create_engine(url)
    assert isinstance(engine, dbm.Engine)


@pytest.mark.order(51)
def test_create_session_none_returns_none(dbminimal):
    dbm = dbminimal
    assert dbm.create_session(None) is None


@pytest.mark.order(52)
def test_create_session_returns_scoped(dbminimal):
    dbm = dbminimal
    url = dbm.URL.create("sqlite+pysqlite", database=":memory:")
    engine = dbm.create_engine(url)
    session = dbm.create_session(engine)
    assert isinstance(session, dbm.scoped_session)


@pytest.mark.order(53)
def test_create_database_if_not_exists(monkeypatch, dbminimal):
    dbm = dbminimal
    url = dbm.URL.create("sqlite+pysqlite", database=":memory:")
    monkeypatch.setattr(dbm, "database_exists", lambda url: False)
    dbm.create_database.called = False
    dbm.create_database_if_not_exists(url)
    assert dbm.create_database.called is True

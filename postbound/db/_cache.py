"""Provides `ResultCache`, a transparent result-set cache that wraps an arbitrary `Database`.

Prior to the database rework, result caching was built directly into `Database`. It has since been extracted into this
module, so that caching becomes an opt-in decision (wrap the database you want to cache) rather than a built-in default
that every backend has to implement and every user has to reason about.
"""

from __future__ import annotations

import atexit
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ..qal import SqlQuery
from ..util import Version, jsondict
from ._db import (
    Cursor,
    Database,
    DatabaseSchema,
    HintService,
    OptimizerInterface,
    ResultSet,
    StatisticsCatalog,
    simplify_result_set,
)

# TODO: Postgres specific methods, support protocols
# TODO: DuckDB specific methods, support protocols


class _DBCacheJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return {"$datetime": o.isoformat()}
        elif isinstance(o, date):
            return {"$date": o.isoformat()}
        elif isinstance(o, time):
            return {"$time": o.isoformat()}
        elif isinstance(o, timedelta):
            return {"$timedelta": o.total_seconds()}
        return super().default(o)


class _DBCacheJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self._second_hook = kwargs.get("object_hook")
        super().__init__(*args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, obj: Any) -> Any:
        if self._second_hook:
            return self._second_hook(obj)

        if "$datetime" in obj:
            return datetime.fromisoformat(obj["$datetime"])
        elif "$date" in obj:
            return date.fromisoformat(obj["$date"])
        elif "$time" in obj:
            return time.fromisoformat(obj["$time"])
        elif "$timedelta" in obj:
            return timedelta(seconds=obj["$timedelta"])
        return obj


_caches: dict[tuple[Database, Path | None], ResultCache] = {}


class ResultCache(Database):
    """Transparently caches the result sets of `execute_query` calls made against a wrapped `Database`.

    A `ResultCache` behaves like any other `Database`: `schema`, `statistics`, `hinting` and `optimizer` all delegate
    directly to the wrapped database. Only `execute_query` is intercepted -- the query is stringified and used as a
    cache key, so repeated calls with the same query text are served from an in-memory dictionary instead of hitting
    the live database system again.

    If an `offline_cache` path is supplied, the cache also persists across process boundaries: existing entries are
    read from that JSON file when the cache is created and the full (possibly extended) cache content is written back
    to the same file via `atexit` once the process terminates.

    This is most useful for wrapping compute-intensive, read-only queries such as those issued by `PreciseStatistics`,
    since it turns the assumption that the database is immutable during a PostBOUND run into an actual, shared
    performance benefit.

    Parameters
    ----------
    db : Database
        The database whose results should be cached.
    offline_cache : Path, optional
        A JSON file to persist the cache to. If omitted, the cache only lives in memory for the lifetime of this
        `ResultCache` instance.

    See Also
    --------
    ResultCache.create_cache : Preferred way to obtain a `ResultCache`, reusing an existing instance if possible.
    """

    @staticmethod
    def create_cache(db: Database, *, offline_cache: Path | None = None) -> ResultCache:
        """Provides a `ResultCache` for the given database, reusing an already existing cache if one is available.

        Repeated calls with the same `db` and `offline_cache` combination return the exact same `ResultCache`
        instance. This ensures that all callers within a process actually share the same in-memory cache contents,
        rather than each maintaining their own (and potentially racing on the same `offline_cache` file).

        Parameters
        ----------
        db : Database
            The database whose results should be cached.
        offline_cache : Path, optional
            A JSON file to persist the cache to. See `ResultCache` for details.

        Returns
        -------
        ResultCache
            The (possibly newly created) cache instance for the given database and offline file.
        """
        existing_cache = _caches.get((db, offline_cache))
        if existing_cache is not None:
            return existing_cache

        fresh_cache = ResultCache(db, offline_cache=offline_cache)
        _caches[db, offline_cache] = fresh_cache
        return fresh_cache

    def __init__(self, db: Database, *, offline_cache: Path | None = None) -> None:
        super().__init__(db.system_name)
        self.offline_file = offline_cache

        self._db = db
        self._cache: dict[str, ResultSet] = {}

        self._init_offline()

    def schema(self) -> DatabaseSchema:
        return self._db.schema()

    def statistics(self) -> StatisticsCatalog:
        return self._db.statistics()

    def hinting(self) -> HintService:
        return self._db.hinting()

    def optimizer(self) -> OptimizerInterface:
        return self._db.optimizer()

    def execute_query(self, query: SqlQuery | str, *, raw: bool = False) -> Any:
        """Executes the query on the wrapped database, or serves it from the cache if it was executed before.

        The cache key is the string representation of `query`. See `Database.execute_query` for the semantics of
        `raw`.
        """
        stringified_query = str(query) if isinstance(query, SqlQuery) else query

        cached_res = self._cache.get(stringified_query)
        if cached_res is not None:
            return cached_res if raw else simplify_result_set(cached_res)

        result_set = self._db.execute_query(query, raw=True)
        self._cache[stringified_query] = result_set
        return result_set if raw else simplify_result_set(result_set)

    def database_name(self) -> str:
        return self._db.database_name()

    def dbms_version(self) -> Version:
        return self._db.dbms_version()

    def describe(self) -> jsondict:
        return {"interface-type": "result-cache", "database": self._db.describe()}

    def reset_connection(self) -> Any:
        return self._db.reset_connection()

    def cursor(self) -> Cursor:
        return self._db.cursor()

    def close(self) -> None:
        self._db.close()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._db == other._db and self.offline_file == other.offline_file

    def __hash__(self) -> int:
        return hash((self._db, self.offline_file))

    def _dump_cache(self) -> None:
        assert self.offline_file is not None
        with open(self.offline_file, "w") as f:
            json.dump(self._cache, f, cls=_DBCacheJsonEncoder)

    def _init_offline(self) -> None:
        if self.offline_file is None:
            return

        # A small implementation detail: we only register the dumping logic after we have completely read the cache
        # otherwise, an interactive session might close the shell before the cache has been read completely, resulting
        # in a (partially) empty cache being written back to disk thanks to atexit. We don't want this.

        if not self.offline_file.exists():
            # if the file does not exist, nothing has been cached, yet. But this might change in the course of this
            # cache's lifetime. Therefore, we still need to register something to dump our cache after the fact.
            atexit.register(self._dump_cache)
            return

        with open(self.offline_file) as f:
            self._cache = json.load(f, cls=_DBCacheJsonDecoder)

        atexit.register(self._dump_cache)

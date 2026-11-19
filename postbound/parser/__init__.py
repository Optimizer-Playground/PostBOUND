"""The parser constructs `SqlQuery` objects from query strings.

Other than the parsing itself, the process will also execute a basic column binding process. For example, consider
a query like *SELECT \\* FROM R WHERE R.a = 42*. In this case, the binding only affects the column reference *R.a*
and sets the table of that column to *R*. This binding based on column and table names is always performed.

If the table cannot be inferred based on the column name (e.g. for a query like *SELECT * FROM R, S WHERE a = 42*), a
second binding phase can be executed. This binding needs a working database connection and queries the database schema
to detect the correct tables for each column. Whether the second phase should also be executed by default can be
configured system-wide by setting the `auto_bind_columns` variable.

Notes
-----
Please beware that SQL parsing is a very challenging undertaking and there might be bugs in some lesser-used features.
If you encounter any issues, please report them on the GitHub issue tracker.
We test the parser based on some popular benchmarks, namely JOB and Stats to ensure that result sets from the raw SQL queries
match result sets from the parsed queries. However, we cannot guarantee that the parser will work for all SQL queries.

The parsing itself is based on the pglast project that implements a SQL -> JSON/dict conversion, based on the actual Postgres
query parser. Our parser implementation takes such a JSON representation as input and generates the more verbose structures of
the qal. There exists a Jupyter notebook called *PglastParsingTests* in the *tests* directory that shows the output emitted by
pglast for different SQL query features.

References
----------

.. pglast project: https://github.com/lelit/pglast Thanks a lot for maintaining this fantastic tool and the great support!
"""

from ._parser import (
    ParserError,
    ParserWarning,
    auto_bind_columns,
    load_column_json,
    load_expression_json,
    load_operator_json,
    load_predicate_json,
    load_query,
    load_table_json,
    parse_query,
)

__all__ = [
    "ParserError",
    "ParserWarning",
    "auto_bind_columns",
    "load_column_json",
    "load_expression_json",
    "load_operator_json",
    "load_predicate_json",
    "load_query",
    "load_table_json",
    "parse_query",
]

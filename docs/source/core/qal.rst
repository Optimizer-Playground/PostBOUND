Query Abstraction
=================

The query abstraction is used to represent SQL queries in a unified way, and to make accessing different parts of them
easier. This section walks you through the most important parts of the query abstraction.
Everything query-related is contained in the *query abstraction layer*, or :mod:`~postbound.qal` for short.

In the next sections, we are going to use the following example query:

.. ipython:: python

    import postbound as pb
    raw_query = """
        SELECT u.Id, u.DisplayName, avg(p.Score)
        FROM Users u
            JOIN Posts p ON u.Id = p.OwnerUserId
        WHERE p.PostTypeId = 2
            AND p.AcceptedAnswerId > 0
        GROUP BY u.Id, u.DisplayName
        ORDER BY avg(p.Score) DESC
    """

You can parse this query into a proper :class:`~postbound.SqlQuery` object using the :func:`~postbound.parse_query`
function:

.. ipython:: python

    query = pb.parse_query(raw_query)
    print(pb.qal.format_quick(query))


Basic query structure
---------------------

PostBOUND uses a query abstraction that consists of three main components:

1. The top-level :class:`~postbound.SqlQuery` object, which represents an entire, ready-to-run SQL query
2. Each query consists of one or multiple clauses, such as *SELECT*, *FROM*, *WHERE*, etc. These are represented by
   subclasses of the abstract :class:`~postbound.qal.BaseClause`.
3. Clauses are composed of expressions. These can be simple predicates, function calls, or even subqueries (which in turn
   contain :class:`~postbound.SqlQuery` instances). Expressions are represented by subclasses of the abstract
   :class:`~postbound.qal.SqlExpression`.

Use the :meth:`~postbound.SqlQuery.ast` method to inspect the structure of a query. For example, our example query
looks like this:

.. ipython:: python

    print(query.ast())

The query structure is quite flexible and we try to model a large portion of the scope of SQL features with it.
For example, we support (recursive) :class:`CTEs <postbound.qal.CommonTableExpression>`,
:class:`window functions <postbound.qal.WindowExpression>`, :class:`CASE expressions <postbound.qal.CaseExpression>`, or
even :class:`qualified star expressions <postbound.qal.StarExpression>` (e.g., ``SELECT t.* FROM t``).

.. important::

    The query parser is based on the actual Postgres parser (thanks to `pglast <https://github.com/lelit/pglast>`_!).
    While this means that we have pretty good coverage of the SQL standard, it also means that we cannot parse queries that
    Postgres does not understand.


Working with queries
--------------------

An important design decision of our query abstraction is that all queries are **immutable**.
This means that once created, a query cannot be changed anymore.
Instead, you need to create a new query object that contains your desired changes.
The :mod:`~postbound.transform` module has a large suite of functions that make these updates much easier:

.. ipython:: python

    pb.transform.as_count_star_query(query)
    pb.transform.drop_clause(query, pb.qal.Where)
    pb.transform.add_clause(query, pb.qal.Limit(limit=10))

All of the qal building blocks provide a visitor-based interface that allows you to traverse the query structure in a
consistent way. These are defined in the :class:`~postbound.qal.ClauseVisitor` and
:class:`~postbound.qal.SqlExpressionVisitor`. There is also a dedicated :class:`~postbound.qal.PredicateVisitor`, even
though predicates are just a special case of expressions. You can make use of Python multiple inheritance to create single
visitor class that traverses an entire query.

.. tip::

    Many parts of PostBOUND that represent queries or query plans have a ``tables()`` method and a ``columns()`` method.
    These methods return sets of all tables and columns that are referenced by the object and any "children" of it.


Working with joins and filters
------------------------------

A core part of query optimization tasks is to analyze which join conditions and filter predicates are present in the query.
You can either analyze queries manually and traverse the :class:`~postbound.qal.Where` clause. At the same time, the query
abstraction also provides :class:`~postbound.qal.QueryPredicates` for a more high-level access:

.. ipython:: python

    query.from_clause
    query.where_clause
    query.predicates()

The query predicates can be used to directly retrieve predicates that are relevant for specific tables, e.g.,

.. ipython:: python

    query.predicates().joins()
    query.predicates().joins_for(pb.TableReference("users", "u"))
    query.predicates().filters_for(pb.TableReference("posts", "p"))
    query.predicates().joins_between(
        pb.TableReference("users", "u"),
        pb.TableReference("posts", "p")
    )

.. attention::

    When traversing the query manually, don't forget to also check the :class:`~postbound.qal.From` clause! It might also
    contain join conditions that where specified with the ``JOIN ... ON ...`` syntax, e.g.,
    ``SELECT * FROM t1 JOIN t2 ON t1.id = t2.id``.

The query abstraction uses a full-blown recursive structure to represent predicates. While this approach allows for a large
expressivity, it makes extracting specific bits of information a bit cumbersome. For example, to get any
:class:`~postbound.TableReference` from a join predicate, one would need to do something like the following:

.. ipython:: python

    full_pred = pb.util.collections.get_any(query.predicates().joins())
    full_pred.join_partners()
    single_pred = pb.util.simplify(full_pred.join_partners())
    single_pred
    any_table = single_pred[0]
    any_table

This is because the query abstraction needs to handle cases of complex conjunctiontive or disjunctive predicates accross
multiple tables such as ``R.a = S.b OR R.a = T.c``. However, such complicated structures do not occur in the commonly used
benchmarks.

To ease the development experience, PostBOUND also has a **simplified version of query predicates** for cases where the
predicates follow simple structures:

- the :class:`~postbound.qal.SimpleFilter` can be used for filter predicates that roughly match the structure
  ``<column> <operator> <value>``.
- the :class:`~postbound.qal.SimpleJoin` can be used for join predicates that are plain inner equi-joins, i.e.
  ``<column 1> = <column 2>``.

Since these simplifications only apply to a subset of all possible predicates, you need to check whether a predicate is
actually of a supported form before creating the simplified version. See the class documentations for more details.
Once you have obtained a simplified predicate, its components can be accessed in a more straightforward way:

.. ipython:: python

    simple_filters = pb.qal.SimpleFilter.wrap_all(query.predicates().filters())
    filter_pred = pb.util.collections.get_any(simple_filters)
    filter_pred.column
    filter_pred.operation
    filter_pred.value
    simple_joins = pb.qal.SimpleJoin.wrap_all(query.predicates().joins())
    join_pred = pb.util.simplify(simple_joins)
    join_pred.lhs, join_pred.rhs

Compare this output to the listing of the full AST above.

.. attention::

    :class:`~postbound.qal.QueryPredicates` also has a convenience method :meth:`~postbound.qal.QueryPredicates.simplify`
    that returns simplified version of all predicates that can actually be simplified. However, if some predicates are more
    complicated than the simplification can handle, these are silently dropped form the result. Never forget to check
    :meth:`~postbound.qal.QueryPredicates.all_simple` first to be sure you don't lose any important predicates!

Many query optimizers derive **equivalence classes** from the query predicates to detect more worthwhile joins that are not
explicitly listed in the query. You can do the same (currently somewhat clunkily) by adding all predicates that can be
derived from equivalence classes to the query. Use :func:`~postbound.qal.determine_join_equivalence_classes` and
:func:`~postbound.qal.generate_predicates_for_equivalence_classes` or the shorthand transformation
:func:`~postbound.transform.add_ec_predicates`.


DML and DDL queries
-------------------

Sadly, the :class:`~postbound.SqlQuery` abstraction is currently limited to plain ``SELECT`` queries.
Since large portions of the code base rely on this assumption, it is unlikely to change in the future. This also applies to
queries with set operations such as ``UNION`` or ``INTERSECT``. These are handled by a dedicated
:class:`~postbound.SetQuery`. See the class documentations for more details on how to use them.

If, at some point in the future, PostBOUND has proper support for DML or DDL queries, these will properly be represented
by separate query classes similar to the :class:`~postbound.qal.SetQuery`. To make clear that API functions can work with
queries beyond plain ``SELECT``, we use the :class:`~postbound.qal.SqlStatement`. If you only want
``SELECT`` queries but are fine with set operations, use :class:`~postbound.qal.SelectStatement`.


Relational algebra
------------------

PostBOUND also provides a simple version of relational algebra. Check out the :mod:`postbound.relalg` module for more
details. In short, use :func:`~postbound.relalg.parse_relalg` to convert an :class:`~postbound.SqlQuery` to an
equivalent tree of :class:`~postbound.relalg.RelNode` instances.

.. note::

    The relational algebra is currently not integrated with the optimization pipelines. Instead, you can use it internalliy
    within the different optimization stages when calculating the query plan.

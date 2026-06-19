from __future__ import annotations

import copy
import math
import re
from enum import Enum
from numbers import Number
from typing import (
    Literal,
    Optional,
    SupportsFloat,
    SupportsIndex,
    TypeGuard,
    TypeVar,
    overload,
)

from .util._errors import StateError
from .util.jsonize import jsondict

VisitorResult = TypeVar("VisitorResult")
"""Result of visitor invocations."""

TimeMs = float

Cost = float
"""Type alias for a cost estimate."""


class Cardinality(Number):
    """Cardinlities represent the number of tuples/row in a relation.

    Our cardinality model can be in one of three states:

    - A valid cardinality can be any non-negative integer. This is the default and most common state.
    - An unknown cardinality is represented by NaN.
    - A prohibitively large cardinality is represented by inf.

    Basically, cardinality instances are just wrappers around their integer value that also catch the two special cases.
    Use cardinalities as you would use normal numbers. Notice that cardinalities are immutable, so all mathematical operators
    return a new cardinality instance.

    To check for the state of a cardinality, you can either use `is_valid()` or the more specific `isnan()` and `isinf()`.
    Furthermore, a more expressive alias for `isnan()` exists in the form of `is_unknown()`.

    You can access the raw cardinality value vie the `value` property. However, this property requires that the cardinality is
    indeed in a valid state. If you want to handle invalid cardinalities yourself, `get()` returns a general float value (that
    can also be *NaN* or *inf*).

    To construct valid cardinalities, it is probably easiest to just create a new instance and passing the desired value.
    The `of()` factory method can be used for better readability. Additionally, the `unknown()` and `infinite()` factory
    methods can be used to create cardinalities in the special states.

    Lastly, cardinalities can be used in *match* statements. They match the following pattern: *(is_valid, value)*. If the
    cardinality is invalid, the value is set to -1.
    """

    @staticmethod
    def of(value: int | float | Cardinality) -> Cardinality:
        """Creates a new cardinality with a specific value. This is just a shorthand for `Cardinality(value)`."""
        if isinstance(value, Cardinality):
            return value
        return Cardinality(value)

    @staticmethod
    def unknown() -> Cardinality:
        """Creates a new cardinality with an unknown value."""
        return Cardinality(math.nan)

    @staticmethod
    def infinite() -> Cardinality:
        """Creates a new cardinality with an infinite value."""
        return Cardinality(math.inf)

    def __init__(self, value: int | float) -> None:
        self._nan = math.isnan(value)
        self._inf = math.isinf(value)
        self._valid = not self._nan and not self._inf
        self._value = round(value) if self._valid else -1

    __slots__ = ("_nan", "_inf", "_valid", "_value")
    __match_args__ = ("_valid", "_value")

    @property
    def value(self) -> int:
        """Get the value wrapped by this cardinality instance. If the cardinality is invalid, a `StateError` is raised."""
        if not self._valid:
            raise StateError(
                "Not a valid cardinality. Use is_valid() to check, or get() to handle unknown values yourself."
            )
        return self._value

    def isnan(self) -> bool:
        """Checks, whether cardinality value is *NaN*."""
        # We call this method isnan instead of is_nan to be consistent with math.isnan and np.isnan
        return self._nan

    def isinf(self) -> bool:
        """Checks, whether cardinality value is infinite."""
        # We call this method isinf instead of is_inf to be consistent with math.isinf and np.isinf
        return self._inf

    def is_unknown(self) -> bool:
        """Checks, whether cardinality value is unknown (i.e. *NaN*).

        This is just a more expressive alias for `isnan()`. It is not a synonym for `is_valid()`.
        """
        return self._nan

    def is_valid(self) -> bool:
        """Checks, whether this cardinality is valid, i.e. neither *NaN* nor infinite.

        If the cardinality is valid, the value can be safely accessed via the `value` property.
        """
        return self._valid

    def get(self) -> float:
        """Provides the value of this cardinality.

        In contrast to accessing the `value` property, this method always returns a float and does not raise an error for
        invalid cardinalities. Instead, it returns *NaN* for unknown cardinalities and *inf* for infinite cardinalities.
        """
        return float(self)

    def __json__(self) -> jsondict:
        return float(self)

    def __bool__(self) -> bool:
        return self._valid

    def __add__(self, other: object) -> Cardinality:
        if isinstance(other, Cardinality):
            return Cardinality(self.get() + other.get())
        if isinstance(other, (int, float)):
            return Cardinality(self.get() + other)
        return NotImplemented

    def __radd__(self, other: object) -> Cardinality:
        if isinstance(other, (int, float)):
            return Cardinality(other + self.get())
        return NotImplemented

    def __neg__(self) -> Cardinality:
        # What's a negative cardinality supposed to be?
        return NotImplemented

    def __sub__(self, other: object) -> Cardinality:
        if isinstance(other, Cardinality):
            return Cardinality(self.get() - other.get())
        if isinstance(other, (int, float)):
            return Cardinality(self.get() - other)
        return NotImplemented

    def __rsub__(self, other: object) -> Cardinality:
        if isinstance(other, (int, float)):
            return Cardinality(other - self.get())
        return NotImplemented

    def __mul__(self, other: object) -> Cardinality:
        if isinstance(other, Cardinality):
            return Cardinality(self.get() * other.get())
        if isinstance(other, (int, float)):
            return Cardinality(self.get() * other)
        return NotImplemented

    def __rmul__(self, other: object) -> Cardinality:
        if isinstance(other, (int, float)):
            return Cardinality(other * self.get())
        return NotImplemented

    def __truediv__(self, other: object) -> Cardinality:
        if isinstance(other, Cardinality):
            return Cardinality(self.get() / other.get())
        if isinstance(other, (int, float)):
            return Cardinality(self.get() / other)
        return NotImplemented

    def __rtruediv__(self, other: object) -> Cardinality:
        if isinstance(other, (int, float)):
            return Cardinality(other / self.get())
        return NotImplemented

    def __pow__(self, other: object) -> Cardinality:
        if isinstance(other, Cardinality):
            # TODO: should we allow exponentiation by a cardinality? What would that mean?
            # For now, I can't really think of a use case, but it's probably not a good idea to restrict the
            # allowed operations too much. Therefore, we leave it allowed for now.
            return Cardinality(self.get() ** other.get())
        if isinstance(other, (int, float)):
            return Cardinality(self.get() ** other)
        return NotImplemented

    def __rpow__(self, other: object) -> Cardinality:
        # See comment on __pow__. Not sure, whether this is a good idea or not.
        if isinstance(other, (int, float)):
            return Cardinality(other ** self.get())
        return NotImplemented

    def __abs__(self) -> Cardinality:
        # Cardinalities are always positive (and -1 is only used internally)
        return self

    def __trunc__(self) -> Cardinality:
        # Cardinalities are always positive integers, so truncating does nothing
        return self

    def __ceil__(self) -> Cardinality:
        # Cardinalities are always positive integers, so ceiling does nothing
        return self

    def __floor__(self) -> Cardinality:
        # Cardinalities are always positive integers, so flooring does nothing
        return self

    def __round__(self, ndigits: int = 0) -> Cardinality:
        # Cardinalities are always positive integers, so rounding does nothing
        return self

    def __divmod__(self, other: SupportsFloat | SupportsIndex):
        if not self._valid:
            return math.nan, math.nan
        if isinstance(other, Cardinality):
            if other._nan:
                return math.nan, math.nan
            if other._inf:
                return 0, self.value
            return divmod(self.value, other.value)
        if isinstance(other, (int, float)):
            return divmod(self.value, other)
        return NotImplemented

    def __rdivmod__(
        self, other: SupportsFloat | SupportsIndex
    ) -> tuple[Number, Number]:
        if self._nan:
            own_value = math.nan
        elif self._inf:
            own_value = math.inf
        else:
            own_value = self.value
        return divmod(float(other), own_value)

    def __floordiv__(self, other: object) -> int:
        return math.floor(self / other)

    def __rfloordiv__(self, other: SupportsFloat | SupportsIndex) -> int:
        return math.floor(other / self)

    def __mod__(self, other: object) -> Cardinality:
        if not self._valid:
            return Cardinality.unknown()

        match other:
            case Cardinality(_, otherval):
                if other._nan:
                    return Cardinality.unknown()
                if other._inf:
                    return self
                return Cardinality(self.value % otherval)

            case int():
                return Cardinality(self.value % other)

            case float():
                if math.isnan(other):
                    return Cardinality.unknown()
                if math.isinf(other):
                    return self
                return Cardinality(self.value % other)

        return NotImplemented

    def __rmod__(self, other: SupportsFloat | SupportsIndex) -> Cardinality:
        if math.isnan(other) or math.isinf(other):
            return Cardinality.unknown()

        if self._nan:
            return Cardinality.unknown()
        if self._inf:
            return Cardinality(float(other))
        return Cardinality(float(other) % self.value)

    def __lt__(self, other: object) -> bool:
        if not self._valid:
            return False

        match other:
            case Cardinality(_, otherval):
                if other._nan:
                    return False
                if other._inf:
                    return True
                return self.value < otherval

            case int() | float():
                return self.value < other

        return NotImplemented

    def __le__(self, other: object) -> bool:
        if not self._valid:
            return False

        match other:
            case Cardinality(_, otherval):
                if other._nan:
                    return False
                if other._inf:
                    return True
                return self.value <= otherval

            case int() | float():
                return self.value <= other

        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if not self._valid:
            return False

        match other:
            case Cardinality(_, otherval):
                if other._nan:
                    return False
                if other._inf:
                    return True
                return self.value > otherval

            case int() | float():
                return self.value > other

        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if not self._valid:
            return False

        match other:
            case Cardinality(_, otherval):
                if other._nan:
                    return False
                if other._inf:
                    return True
                return self.value >= otherval

            case int() | float():
                return self.value >= other

        return NotImplemented

    def __float__(self) -> float:
        if self._nan:
            return math.nan
        if self._inf:
            return math.inf
        return float(self.value)

    def __int__(self) -> int:
        if not self._valid:
            raise StateError(
                "Not a valid cardinality. Use is_valid() to check, or get() to handle unknown values yourself."
            )
        return self.value

    def __complex__(self) -> complex:
        if self._nan:
            return complex(math.nan)
        if self._inf:
            return complex(math.inf)
        return complex(float(self.value))

    def __eq__(self, other: object) -> bool:
        match other:
            case Cardinality():
                if self._nan and other._nan:
                    return True
                if self._inf and other._inf:
                    return True
                return self.value == other.value

            case int():
                return self._valid and self.value == other

            case float():
                if self._nan and math.isnan(other):
                    return True
                if self._inf and math.isinf(other):
                    return True
                return self._valid and self.value == other

        return NotImplemented

    def __hash__(self) -> int:
        # There is no need to hash _valid, since it is directly derived from _nan and _inf
        return hash((self._nan, self._inf, self._value))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self._nan:
            return "NaN"
        if self._inf:
            return "inf"
        return str(self.value)


class ScanOperator(Enum):
    """The scan operators supported by PostBOUND.

    These can differ from the scan operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """

    SequentialScan = "Seq. Scan"
    IndexScan = "Idx. Scan"
    IndexOnlyScan = "Idx-only Scan"
    BitmapScan = "Bitmap Scan"

    def __json__(self) -> str:
        return self.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


class JoinOperator(Enum):
    """The join operators supported by PostBOUND.

    These can differ from the join operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """

    NestedLoopJoin = "NLJ"
    HashJoin = "Hash Join"
    SortMergeJoin = "Sort-Merge Join"
    IndexNestedLoopJoin = "Idx. NLJ"

    def __json__(self) -> str:
        return self.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


class IntermediateOperator(Enum):
    """The intermediate operators supported by PostBOUND.

    Intermediate operators are those that do not change the contents of their input relation, but only the way in which it is
    available. For example, a sort operator changes the order of the tuples.
    """

    Sort = "Sort"
    Memoize = "Memoize"
    Materialize = "Materialize"

    def __json__(self) -> str:
        return self.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


PhysicalOperator = ScanOperator | JoinOperator | IntermediateOperator
"""Supertype to model all physical operators supported by PostBOUND.

These can differ from the operators that are actually available in the selected target database system.
"""

_IdentifierPattern = re.compile(r"^[a-z_][a-z0-9_\$]*$")
"""Regular expression to check for valid identifiers.

In line with Postgres' way of name resolution, we only permit identifiers with lower case characters. This forces all
identifiers which contain at least one upper case character to be quoted.

References
----------
- Postgres documentation on identifiers: https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS
- Regex tester: https://regex101.com/r/TtrNQg/1
"""

SqlKeywords = frozenset(
    {
        "ALL",
        "AND",
        "ANY",
        "ARRAY",
        "AS",
        "ASC",
        "ASYMMETRIC",
        "AT",
        "BINARY",
        "BOTH",
        "CASE",
        "CAST",
        "CHECK",
        "COLLATE",
        "COLUMN",
        "CONSTRAINT",
        "CREATE",
        "CROSS",
        "CURRENT_CATALOG",
        "CURRENT_DATE",
        "CURRENT_ROLE",
        "CURRENT_SCHEMA",
        "CURRENT_TIME",
        "CURRENT_TIMESTAMP",
        "CURRENT_USER",
        "DEFAULT",
        "DEFERRABLE",
        "DESC",
        "DISTINCT",
        "DO",
        "ELSE",
        "END",
        "EXCEPT",
        "FALSE",
        "FETCH",
        "FOR",
        "FOREIGN",
        "FROM",
        "FULL",
        "GRANT",
        "GROUP",
        "HAVING",
        "ILIKE",
        "IN",
        "INITIALLY",
        "INNER",
        "INTERSECT",
        "INTO",
        "IS",
        "JOIN",
        "LATERAL",
        "LEADING",
        "LEFT",
        "LIKE",
        "LIMIT",
        "LOCALTIME",
        "LOCALTIMESTAMP",
        "NATURAL",
        "NOT",
        "NULL",
        "OFFSET",
        "ON",
        "ONLY",
        "OR",
        "ORDER",
        "OUTER",
        "OVERLAPS",
        "PLACING",
        "PRIMARY",
        "REFERENCES",
        "RETURNING",
        "RIGHT",
        "SELECT",
        "SESSION_USER",
        "SIMILAR",
        "SOME",
        "SYMMETRIC",
        "TABLE",
        "THEN",
        "TO",
        "TRAILING",
        "TRUE",
        "UNION",
        "UNIQUE",
        "USER",
        "USING",
        "VARIADIC",
        "VERBOSE",
        "WHEN",
        "WHERE",
        "WINDOW",
        "WITH",
    }
)
"""An (probably incomplete) list of reserved SQL keywords that must be quoted before being used as identifiers."""


def quote(identifier: str) -> str:
    """Quotes an identifier if necessary.

    Valid identifiers can be used as-is, e.g. *title* or *movie_id*. Invalid identifiers will be wrapped in quotes, such as
    *"movie title"* or *"movie-id"*.

    Parameters
    ----------
    identifier : str
        The identifier to quote. Note that empty strings are treated as valid identifiers.

    Returns
    -------
    str
        The identifier, potentially wrapped in quotes.
    """
    if not identifier:
        return ""
    valid_identifier = (
        _IdentifierPattern.fullmatch(identifier)
        and identifier.upper() not in SqlKeywords
    )
    return identifier if valid_identifier else f'"{identifier}"'


def normalize(identifier: str) -> str:
    """Generates a normalized version of an identifier.

    Normalization is based on the Postgres rules of performing all comparisons in a case-insensitive manner.

    Parameters
    ----------
    identifier : str
        The identifier to normalize. Notice that empty strings can be normalized as well (without doing anything).

    Returns
    -------
    str
        The normalized identifier
    """
    return identifier.lower()


class TableReference:
    """A table reference represents a database table.

    It can either be a physical table, a CTE, or an entirely virtual query created via subqueries. Note that a table
    reference is indeed just a reference and not a 1:1 "representation" since each table can be sourced multiple times
    in a query. Therefore, in addition to the table name, each instance can optionally also contain an alias to
    distinguish between different references to the same table. In case of virtual tables, the full name will usually be empty
    and only the alias set. An exception are table references that refer to CTEs: their full name is set to the CTE name, the
    alias to the alias from the FROM clause (if present) and the table is still treated as virtual.

    Table references can be sorted lexicographically. All instances should be treated as immutable objects.

    Parameters
    ----------
    full_name : str
        The name of the table, corresponding to the name of a physical database table (or a view)
    alias : str, optional
        Alternative name that is in queries to refer to the table, or to refer to the same table multiple times.
        Defaults to an empty string
    virtual : bool, optional
        Whether the table is virtual or not. As a rule of thumb, virtual tables cannot be part of a *FROM* clause on
        their own, but need some sort of context. For example, the alias of a subquery is typically represented as a
        virtual table in PostBOUND. One cannot directly reference that alias in a *FROM* clause, without also
        specifying the subquery. Defaults to *False* since most tables will have direct physical counterparts.
    schema : str, optional
        The schema in which the table is located. Defaults to an empty string if the table is in the default schema or the
        schema is unknown.
    catalog : str, optional
        The catalog in which the table is located. Defaults to an empty string if the table is in the default catalog or the
        catalog is unknown.

    Raises
    ------
    ValueError
        If neither full name nor an alias are provided, or if a schema without a full name is provided.
    """

    @staticmethod
    def create_virtual(
        alias: str,
        *,
        full_name: str = "",
        schema: str = "",
        catalog: str = "",
    ) -> TableReference:
        """Generates a new virtual table reference with the given alias.

        Parameters
        ----------
        alias : str
            The alias of the virtual table. Cannot be *None*.
        full_name : str, optional
            An optional full name for the entire table. This is mostly used to create references to CTE tables.
        schema : str, optional
            The schema in which the table is located. Defaults to an empty string if the table is in the default schema or the
            schema is unknown.
        catalog : str, optional
            The catalog in which the table is located. Defaults to an empty string if the table is in the default catalog or the
            catalog is unknown.

        Returns
        -------
        TableReference
            The virtual table reference
        """
        return TableReference(
            full_name, alias, virtual=True, schema=schema, catalog=catalog
        )

    def __init__(
        self,
        full_name: str,
        alias: str = "",
        *,
        virtual: bool = False,
        schema: str = "",
        catalog: str = "",
    ) -> None:
        if not full_name and not alias:
            raise ValueError("Full name or alias required")
        if schema and not full_name:
            raise ValueError(
                "Schema can only be set if a full name is provided"
            )
        if catalog and not full_name:
            raise ValueError(
                "Catalog can only be set if a full name is provided"
            )

        # NB: we use conditional assignments to protect against accidental
        # None parameters
        self._catalog = catalog if catalog else ""
        self._schema = schema if schema else ""
        self._full_name = full_name if full_name else ""
        self._alias = alias if alias else ""
        self._virtual = virtual

        self._identifier = self._alias if self._alias else self._full_name

        self._normalized_catalog = normalize(self._catalog)
        self._normalized_schema = normalize(self._schema)
        self._normalized_full_name = normalize(self._full_name)
        self._normalized_alias = normalize(self._alias)
        self._normalized_identifier = normalize(self._identifier)
        self._hash_val = hash(
            (
                self._normalized_full_name,
                self._normalized_alias,
                self._normalized_schema,
                self._normalized_catalog,
            )
        )

        txt_components: list[str] = []
        if self._catalog:
            txt_components.append(quote(self._catalog))
        if self._schema:
            txt_components.append(quote(self._schema))
        txt_components.append(quote(self._full_name))
        table_txt = ".".join(txt_components)

        if table_txt and self._alias:
            self._sql_repr = f"{table_txt} AS {quote(self._alias)}"
        elif self._alias:
            self._sql_repr = quote(self._alias)
        else:
            raise ValueError("Full name or alias required")

    __slots__ = (
        "_catalog",
        "_schema",
        "_full_name",
        "_alias",
        "_virtual",
        "_identifier",
        "_normalized_catalog",
        "_normalized_schema",
        "_normalized_full_name",
        "_normalized_alias",
        "_normalized_identifier",
        "_hash_val",
        "_sql_repr",
    )
    __match_args__ = ("full_name", "alias", "virtual", "schema", "catalog")

    @property
    def full_name(self) -> str:
        """Get the full name of this table. If empty, alias is guaranteed to be set.

        Returns
        -------
        str
            The name of the table
        """
        return self._full_name

    @property
    def alias(self) -> str:
        """Get the alias of this table. If empty, the full name is guaranteed to be set.

        The precise semantics of alias usage differ from database system to system. For example, in Postgres an alias
        shadows the original table name, i.e. once an alias is specified, it *must* be used to reference to the table
        and its columns.

        Returns
        -------
        str
            The alias of the table
        """
        return self._alias

    @property
    def virtual(self) -> bool:
        """Checks whether this table is virtual. In this case, only the alias and not the full name is set.

        Returns
        -------
        bool
            Whether this reference describes a virtual table
        """
        return self._virtual

    @property
    def schema(self) -> str:
        """Get the schema in which this table is located.

        Returns
        -------
        str
            The schema or an empty string if the schema is either unknown or the table is located in the default schema.
        """

        return self._schema

    @property
    def catalog(self) -> str:
        """Get the catalog in which this table is located.

        Returns
        -------
        str
            The catalog or an empty string if the catalog is either unknown or the table is located in the default catalog.
        """
        return self._catalog

    def identifier(self) -> str:
        """Provides a shorthand key that columns can use to refer to this table reference.

        For example, a table reference for *movie_companies AS mc* would have *mc* as its identifier (i.e. the
        alias), whereas a table reference without an alias such as *company_type* would provide the full table name
        as its identifier, i.e. *company_type*.

        Returns
        -------
        str
            The shorthand
        """
        return self._identifier

    def qualified_name(self) -> str:
        """Provides the fully qualified name (i.e. including schema or catalog) of this table.

        Notice that virtual tables do not have a qualified name, since they do not correspond to a physical table.

        Returns
        -------
        str
            The qualified name, quoted as necessary.
        """
        if self.virtual:
            raise VirtualTableError(self)

        name_components: list[str] = []
        if self._catalog:
            name_components.append(quote(self._catalog))
        if self._schema:
            name_components.append(quote(self._schema))
        name_components.append(quote(self._full_name))
        return ".".join(name_components)

    def assign_column(self, name: str) -> BoundColumnReference:
        """Creates a new column that is bound to this table."""
        return BoundColumnReference(name, self)

    def drop_alias(self) -> TableReference:
        """Removes the alias from the current table if there is one. Returns the tabel as-is otherwise.

        Returns
        -------
        TableReference
            This table, but without an alias. Since table references are immutable, the original reference is not
            modified

        Raises
        ------
        StateError
            If this table is a virtual table, since virtual tables only have an alias and no full name.
        """
        if self.virtual:
            raise StateError("An alias cannot be dropped from a virtual table!")
        return TableReference(
            self.full_name, schema=self._schema, catalog=self._catalog
        )

    def with_alias(self, alias: str) -> TableReference:
        """Creates a new table reference for the same table but with a different alias.

        Parameters
        ----------
        alias : str
            The new alias

        Returns
        -------
        TableReference
            The updated table reference

        Raises
        ------
        StateError
            If the current table does not have a full name.
        """
        if not self.full_name:
            raise StateError("Cannot add an alias to a table without full name")

        return TableReference(
            self._full_name,
            alias,
            virtual=self._virtual,
            schema=self._schema,
            catalog=self._catalog,
        )

    def with_schema(self, schema: str) -> TableReference:
        """Creates a new table reference for the same table but with a different schema.

        Parameters
        ----------
        schema : str
            The new schema

        Returns
        -------
        TableReference
            The updated table reference
        """
        return TableReference(
            self._full_name,
            self._alias,
            virtual=self.virtual,
            schema=schema,
            catalog=self._catalog,
        )

    def with_catalog(self, catalog: str) -> TableReference:
        """Creates a new table reference for the same table but with a different catalog.

        Parameters
        ----------
        catalog : str
            The new catalog

        Returns
        -------
        TableReference
            The updated table reference
        """
        return TableReference(
            self._full_name,
            self._alias,
            virtual=self._virtual,
            schema=self._schema,
            catalog=catalog,
        )

    def make_virtual(self) -> TableReference:
        """Creates a new virtual table reference for the same table.

        Returns
        -------
        TableReference
            The updated table reference
        """
        return TableReference(
            self.full_name,
            self.alias,
            virtual=True,
            schema=self._schema,
            catalog=self._catalog,
        )

    def update(
        self,
        *,
        full_name: Optional[str] = None,
        alias: Optional[str] = None,
        virtual: Optional[bool] = None,
        schema: Optional[str] = None,
        catalog: Optional[str] = None,
    ) -> TableReference:
        full_name = self._full_name if full_name is None else full_name
        alias = self._alias if alias is None else alias
        virtual = self._virtual if virtual is None else virtual
        schema = self._schema if schema is None else schema
        catalog = self._catalog if catalog is None else catalog
        return TableReference(
            full_name, alias, virtual=virtual, schema=schema, catalog=catalog
        )

    def __json__(self) -> object:
        return {
            "full_name": self._full_name,
            "alias": self._alias,
            "virtual": self._virtual,
            "schema": self._schema,
            "catalog": self._catalog,
        }

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TableReference):
            return NotImplemented
        return self._normalized_identifier < other._normalized_identifier

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._normalized_full_name == other._normalized_full_name
            and self._normalized_alias == other._normalized_alias
            and self._normalized_schema == other._normalized_schema
            and self._normalized_catalog == other._normalized_catalog
        )

    def __repr__(self) -> str:
        return (
            f"TableReference(full_name='{self.full_name}', alias='{self.alias}', "
            f"virtual={self.virtual}, schema='{self.schema}', catalog='{self.catalog}')"
        )

    def __str__(self) -> str:
        return self._sql_repr


class ColumnReference:
    """A column reference represents a specific column of a specific database table.

    This reference always consists of the name of the "physical" column (see below for special
    cases). In addition, each column can be bound to the table to which it belongs by providing the
    associated table reference.

    Column references can be sorted lexicographically and are designed as immutable data objects.

    For situations where columns must be bound to a table, the `BoundColumnReference` subclass
    exists. It has the same interface like normal column references, but with one difference: its
    ``table`` property is guaranteed to always be a valid `TableReference`. Use the static
    `assert_bound` method to ensure that you are dealing with a bound reference.
    It returns a *TypeGuard* so type checkers should work with the narrowed type after a successful
    check.

    Parameters
    ----------
    name : str
        The name of the column. Cannot be empty.
    table : Optional[TableReference], optional
        The table which provides the column. Can be *None* if the table is unknown.

    Raises
    ------
    ValueError
        If the name is empty (or *None*)

    Notes
    -----
    A number of special cases arise when dealing with subqueries and common table expressions. The
    first one is the fact that columns can be bound to virtual tables, e.g. if they are exported by
    subqueries, etc. In the same vein, columns also do not always need to refer directly to physical
    columns. Consider the following example query:

    ::

        WITH cte_table AS (SELECT foo.f_id, foo.a + foo.b AS 'sum' FROM foo)
        SELECT *
        FROM bar JOIN cte_table ON bar.b_id = cte_table.f_id
        WHERE cte_table.sum < 42

    In this case, the CTE exports a column *sum* that is constructed based on two "actual" columns.
    Hence, the sum column itself does not have any physical representation but will be modelled as a
    column reference nevertheless.

    Due to these situations, we use the full table reference when comparing columns. BUt this can
    have unintended side effects as well. For example, consider a physical column *production_year*
    of a *title* relation. In one query, the column is referenced as *title.production_year*,
    whereas in the other query, *title* is aliased and the column referenced as *t.production_year*.
    While both columns are bound to the *title* relation, only one of the table references contains
    an alias. Therefore, the table references do not compare equal and by extension the columns do
    not compare equal either. To mitigate this issue (at least to some extend), the
    `drop_table_alias` method can be used to obtain a "normalized" version of a column reference.

    See Also
    --------
    BoundColumnReference : A subclass of `ColumnReference` that is always bound to a valid table.
    """

    @staticmethod
    def create(column: str, *, table: str) -> BoundColumnReference:
        """Shortcut method to create a column along with its table."""
        return BoundColumnReference(column, TableReference(table))

    @staticmethod
    def assert_bound(col: ColumnReference) -> TypeGuard[BoundColumnReference]:
        """Checks whether a specific column is bound to (any) table.

        If it is, the column can be treated as an instance of `BoundColumnReference` and the type checker will be able
        to narrow the type accordingly. Most importantly, the `table` property of the column will be guaranteed to be a
        valid `TableReference` and not *None*.

        Notes
        -----
        Sadly, the current spec of the TypeGuard explicitly excludes self from the type check. Therefore, we cannot simply
        modify `is_bound` to return a TypeGuard and have to use an additional method. Ugh.
        """
        return col.is_bound()

    def __init__(
        self, name: str, table: Optional[TableReference] = None
    ) -> None:
        if not name:
            raise ValueError("Column name is required")
        self._name = name
        self._table = table
        self._normalized_name = normalize(self._name)
        self._hash_val = hash((self._normalized_name, self._table))

        if self._table:
            self._sql_repr = (
                f"{quote(self._table.identifier())}.{quote(self._name)}"
            )
        else:
            self._sql_repr = quote(self._name)

    @overload
    def __new__(
        cls, name: str, table: TableReference
    ) -> BoundColumnReference: ...

    @overload
    def __new__(cls, name: str, table: Literal[None]) -> ColumnReference: ...

    def __new__(
        cls, name: str, table: Optional[TableReference] = None
    ) -> ColumnReference:
        if cls is not ColumnReference or table is None:
            return super().__new__(cls)
        return BoundColumnReference(name, table)

    __slots__ = (
        "_name",
        "_table",
        "_normalized_name",
        "_hash_val",
        "_sql_repr",
    )
    __match_args__ = ("name", "table")

    @property
    def name(self) -> str:
        """Get the name of this column. This is guaranteed to be set and will never be empty

        Returns
        -------
        str
            The name
        """
        return self._name

    @property
    def table(self) -> Optional[TableReference]:
        """Get the table to which this column belongs, if specified.

        Returns
        -------
        Optional[TableReference]
            The table or *None*. The table can be an arbitrary reference, i.e. virtual or physical.
        """
        return self._table

    def is_bound(self) -> bool:
        """Checks, whether this column is bound to a table.

        Returns
        -------
        bool
            Whether a valid table reference is set

        See Also
        --------
        assert_bound : A static method that performs the same check but also serves as a type guard to narrow
                       the type to `BoundColumnReference` if the check is successful.
        """
        return self.table is not None

    def belongs_to(self, table: TableReference) -> bool:
        """Checks, whether the column is part of the given table.

        This check does not consult the schema of the actual database or the like, it merely matches the given table
        reference with the `table` attribute of this column.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the table's column is the same as the given one
        """
        return table == self.table

    def bind_to(self, table: TableReference) -> BoundColumnReference:
        """Binds this column to a new table.

        Parameters
        ----------
        table : TableReference
            The new table

        Returns
        -------
        BoundColumnReference
            The updated column reference, the original reference is not modified.
        """
        return BoundColumnReference(self.name, table)

    def as_unbound(self) -> ColumnReference:
        """Removes the table binding from this column.

        Returns
        -------
        ColumnReference
            The updated column reference, the original reference is not modified.
        """
        return ColumnReference(self.name, None)

    def drop_table_alias(self) -> ColumnReference:
        """Removes the alias from the table this column is bound to, if there is one.

        For example, if this column is *t.production_year* based on a table alias *title as t*,
        calling this method turns the column into *title.production_year*, effectively dropping the
        *t* alias.

        This can be useful to obtain a "normalized" version of a column reference.
        """
        if not self.assert_bound(self):
            return self
        unaliased = self.table.drop_alias()
        return self.bind_to(unaliased)

    def __json__(self) -> object:
        return {"column": self._name, "table": self._table}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ColumnReference):
            return NotImplemented
        if self.table == other.table:
            return self._normalized_name < other._normalized_name
        if not self.table:
            return True
        if not other.table:
            return False
        return self.table < other.table

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, (ColumnReference, BoundColumnReference))
            and self._normalized_name == other._normalized_name
            and self.table == other.table
        )

    def __copy__(self) -> ColumnReference:
        return ColumnReference(self.name, self.table)

    def __deepcopy__(self, memo: dict) -> ColumnReference:
        tab = copy.deepcopy(self.table, memo)
        return ColumnReference(str(self.name), tab)

    def __reduce__(self) -> tuple:
        return (self.__class__, (self._name, self._table))

    def __repr__(self) -> str:
        return f"ColumnReference(name='{self.name}', table={repr(self.table)})"

    def __str__(self) -> str:
        return self._sql_repr


class BoundColumnReference(ColumnReference):
    """A column reference that is guaranteed to be bound to a table.

    The main purpose of this class is communication: it conveys the requirement of consumers to have a valid
    table reference on the column - or the guarantee of producers to only return columns that are bound to a
    table.
    """

    def __init__(self, name: str, table: TableReference) -> None:
        super().__init__(name, table)

    __match_args__ = ("name", "table")

    @property
    def table(self) -> TableReference:
        """Get the table to which this column belongs. This is guaranteed to be set and will never be *None*."""
        return self._table  # type: ignore

    def is_bound(self) -> bool:
        return True

    def drop_table_alias(self) -> BoundColumnReference:
        return super().drop_table_alias()  # type: ignore

    __lt__ = ColumnReference.__lt__
    __hash__ = ColumnReference.__hash__
    __eq__ = ColumnReference.__eq__
    __repr__ = ColumnReference.__repr__
    __str__ = ColumnReference.__str__


class UnboundColumnError(StateError):
    """Indicates that a column is required to be bound to a table, but the provided column was not.

    Parameters
    ----------
    column : ColumnReference
        The column without the necessary table binding
    """

    def __init__(self, column: ColumnReference) -> None:
        super().__init__("Column is not bound to any table: " + str(column))
        self.column = column


class VirtualTableError(StateError):
    """Indicates that a table is required to correspond to a physical table, but the provided reference was not.

    Parameters
    ----------
    table : TableReference
        The virtual table
    """

    def __init__(self, table: TableReference) -> None:
        super().__init__("Table is virtual: " + str(table))
        self.table = table

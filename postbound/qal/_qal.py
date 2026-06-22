from __future__ import annotations

import abc
import collections
import enum
import functools
import itertools
import numbers
import warnings
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from types import NoneType
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

import networkx as nx

from .. import util
from .._base import T
from .._core import ColumnReference, TableReference, VisitorResult, quote
from ..util._errors import StateError
from ..util.jsonize import jsondict


class MathOperator(enum.Enum):
    """The supported mathematical operators."""

    Add = "+"
    Subtract = "-"
    Multiply = "*"
    Divide = "/"
    Modulo = "%"
    Concatenate = "||"
    TrueDiv = "//"


class LogicalOperator(enum.Enum):
    """The supported unary and binary operators."""

    Equal = "="
    NotEqual = "<>"
    Less = "<"
    LessEqual = "<="
    Greater = ">"
    GreaterEqual = ">="
    Like = "LIKE"
    NotLike = "NOT LIKE"
    ILike = "ILIKE"
    NotILike = "NOT ILIKE"
    In = "IN"
    Exists = "EXISTS"
    Is = "IS"
    IsNot = "IS NOT"
    Between = "BETWEEN"
    DistinctFrom = "IS DISTINCT FROM"
    NotDistinctFrom = "IS NOT DISTINCT FROM"

    # Postgres-style operators
    Contains = "@>"
    ContainedBy = "<@"
    Overlaps = "&&"


UnarySqlOperators = frozenset(
    {LogicalOperator.Exists, LogicalOperator.Is, LogicalOperator.IsNot}
)
"""The `LogicalSqlOperators` that can be used as unary operators."""


class CompoundOperator(enum.Enum):
    """The supported compound operators."""

    And = "AND"
    Or = "OR"
    Not = "NOT"


SqlOperator = Union[MathOperator, LogicalOperator, CompoundOperator]
"""Captures all different kinds of operators in one type."""


class SetOperator(enum.Enum):
    """The supported set operators."""

    Union = "UNION"
    UnionAll = "UNION ALL"
    Intersect = "INTERSECT"
    Except = "EXCEPT"


class QuantifierOperator(enum.Enum):
    """The supported quantifier operators."""

    All = "ALL"
    Any = "ANY"


class SqlExpression(abc.ABC):
    """Base class for all expressions.

    Expressions form one of the central building blocks of representing a SQL query in the QAL. They specify how values
    from different columns are modified and combined, thereby forming larger (hierarchical) structures.

    Expressions can be inserted in many different places in a SQL query. For example, a *SELECT* clause produces
    columns such as in ``SELECT R.a FROM R``, but it can also modify the column values slightly, such as in
    ``SELECT R.a + 42 FROM R``. To account for all  these different situations, the `SqlExpression` is intended to form
    hierarchical trees and chains of expressions. In the first case, a `ColumnExpression` is used, whereas a
    `MathExpression` can model the second case. Whereas column expressions represent leaves in the expression tree,
    mathematical expressions are intermediate nodes.

    As a more advanced example, a complicated expressions such as `my_udf(R.a::interval + 42)` which consists of a
    user-defined function, a value cast and a mathematical operation is represented the following way:
    `FunctionExpression(MathExpression(CastExpression(ColumnExpression), StaticValueExpression))`. The methods provided by all
    expression instances enable a more convenient use and access to the expression hierarchies.

    The different kinds of expressions are represented using different subclasses of the `SqlExpression` interface.
    This really is an abstract interface, not a usable expression. All inheriting expression have to provide their own
    `__eq__` method and re-use the `__hash__` method provided by the base expression. Remember to explicitly set this
    up! The concrete hash value is constant since the clause itself is immutable. It is up to the implementing class to
    make sure that the equality/hash consistency is enforced.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete expression object
    """

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

    __slots__ = ("_hash_val",)

    @abc.abstractmethod
    def tables(self) -> set[TableReference]:
        """Provides all tables that are accessed by this expression.

        Returns
        -------
        set[TableReference]
            All tables. This includes virtual tables if such tables are present in the expression.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced by this expression.

        Returns
        -------
        set[ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides all columns that are referenced by this expression.

        If a column is referenced multiple times, it is also returned multiple times.

        Returns
        -------
        Iterable[ColumnReference]
            All columns in exactly the order in which they are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterchildren(self) -> Iterable[SqlExpression]:
        """Provides unified access to all child expressions of the concrete expression type.

        For *leaf* expressions such as static values, the iterable will not contain any elements. Otherwise, all
        *direct* children will be returned. For example, a mathematical expression could return both the left, as well
        as the right operand. This allows for access to nested expressions in a recursive manner.

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        """Enables processing of the current expression by an expression visitor.

        Parameters
        ----------
        visitor : SqlExpressionVisitor[VisitorResult]
            The visitor
        args
            Additional arguments that are passed to the visitor
        kwargs
            Additional keyword arguments that are passed to the visitor
        """
        raise NotImplementedError

    def __json__(self) -> jsondict:
        return {
            "node_type": "expression",
            "tables": self.tables(),
            "expression": str(self),
        }

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class StaticValueExpression(SqlExpression, Generic[T]):
    """An expression that wraps a literal/static value.

    This is one of the leaf expressions that does not contain any further child expressions.

    *NULL* values are represented by *None*, you can use `StaticValueExpression.null()` to create such an expression and
    `is_null()` to check whether the value is *NULL*.

    Parameters
    ----------
    value : T
        The value that is wrapped by the expression

    Examples
    --------
    Consider the following SQL query: ``SELECT * FROM R WHERE R.a = 42``. In this case the comparison value of 42 will
    be represented as a static value expression. The reference to the column *R.a* cannot be a static value since its
    values depend on the actual column values. Hence, a `ColumnExpression` is used for it.
    """

    @staticmethod
    def null() -> StaticValueExpression[None]:
        """Create a static value expression that represents a *NULL* value."""
        return StaticValueExpression(None)

    def __init__(self, value: T) -> None:
        self._value = value
        super().__init__(hash(value))

    __slots__ = ("_value",)
    __match_args__ = ("value",)

    @property
    def value(self) -> T:
        """Get the value.

        Returns
        -------
        T
            The value, duh!
        """
        return self._value

    def is_null(self) -> bool:
        """Checks, whether the value is *NULL*."""
        return self._value is None

    def tables(self) -> set[TableReference]:
        return set()

    def columns(self) -> set[ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_static_value_expr(self, *args, **kwargs)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.value == other.value

    def __str__(self) -> str:
        if self.value is None:
            return "NULL"
        match self.value:
            case numbers.Number():
                return str(self.value)
            case _:
                escaped = str(self.value).replace("'", "''")
                return f"'{escaped}'"


class StarExpression(SqlExpression):
    """A special expression that is only used in *SELECT* clauses to select all columns.

    Parameters
    ----------
    from_table : Optional[TableReference], optional
        The table from which to select all columns. Defaults to **None**, in which case all columns of all tables are being
        selected.
    """

    def __init__(self, *, from_table: Optional[TableReference] = None) -> None:
        self._table = from_table
        super().__init__(hash(("*", self._table)))

    __slots__ = ("_table",)
    __match_args__ = ("from_table",)

    @property
    def from_table(self) -> Optional[TableReference]:
        """Get the table from which to select all columns.

        If no such table was selected, all columns of all tables are being selected.

        Returns
        -------
        Optional[TableReference]
            The table, or **None** if all columns are selected
        """
        return self._table

    def tables(self) -> set[TableReference]:
        if self._table:
            return {self._table}
        return set()

    def columns(self) -> set[ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_star_expr(self, *args, **kwargs)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __str__(self) -> str:
        if not self._table:
            return "*"
        return f"{quote(self._table.identifier())}.*"


class CastExpression(SqlExpression):
    """An expression that casts the type of another nested expression.

    Note that PostBOUND itself does not know about the semantics of the actual types or casts. Eventual errors due to
    illegal casts are only caught at runtime by the actual database system.

    Parameters
    ----------
    expression : SqlExpression
        The expression that is casted to a different type.
    target_type : str
        The type to which the expression should be converted to. This cannot be empty.
    type_params: Optional[Sequence[SqlExpression]], optional
        Additional arguments to parameterize the type, such as in *NUMERIC(4, 2)* or *VARCHAR(255)*. For example, when casting
        to *VARCHAR(255)*, the *255* would be an additional parameter, represented as a single static value expression. When
        casting to *NUMERIC(4, 2)*, the *4* and *2* would be the additional parameters (in that order).
    array_type : bool, optional
        Whether the target type is an array type.

    Raises
    ------
    ValueError
        If the `target_type` is empty.
    """

    def __init__(
        self,
        expression: SqlExpression,
        target_type: str,
        *,
        type_params: Optional[Sequence[SqlExpression]] = None,
        array_type: bool = False,
    ) -> None:
        if not expression or not target_type:
            raise ValueError("Expression and target type are required")
        self._casted_expression = expression
        self._target_type = target_type
        self._type_params = tuple(type_params) if type_params else ()
        self._array_type = array_type

        hash_val = hash(
            (
                self._casted_expression,
                self._target_type,
                self._type_params,
                self._array_type,
            )
        )
        super().__init__(hash_val)

    __slots__ = (
        "_casted_expression",
        "_target_type",
        "_type_params",
        "_array_type",
    )
    __match_args__ = (
        "casted_expression",
        "target_type",
        "type_params",
        "array_type",
    )

    @property
    def casted_expression(self) -> SqlExpression:
        """Get the expression that is being casted.

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._casted_expression

    @property
    def target_type(self) -> str:
        """Get the type to which to cast to.

        Returns
        -------
        str
            The desired type. This is never empty.
        """
        return self._target_type

    @property
    def type_params(self) -> Sequence[SqlExpression]:
        """Get additional arguments that parameterize the type.

        For example, when casting to *VARCHAR(255)*, the *255* would be an additional parameter, represented as a single static
        value expression. When casting to *NUMERIC(4, 2)*, the *4* and *2* would be the additional parameters (in that order).

        Returns
        -------
        Sequence[SqlExpression]
            The type parameters or an empty sequence if there are none.
        """
        return self._type_params

    @property
    def array_type(self) -> bool:
        """Get whether the target type is an array type."""
        return self._array_type

    def tables(self) -> set[TableReference]:
        return self._casted_expression.tables() | util.set_union(
            expr.tables() for expr in self._type_params
        )

    def columns(self) -> set[ColumnReference]:
        return self.casted_expression.columns() | util.set_union(
            expr.columns() for expr in self.type_params
        )

    def itercolumns(self) -> Iterable[ColumnReference]:
        casted_cols = list(self.casted_expression.itercolumns())
        param_cols = util.flatten([expr.itercolumns() for expr in self.type_params])
        return casted_cols + param_cols

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.casted_expression] + list(self.type_params)

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_cast_expr(self, *args, **kwargs)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.casted_expression == other.casted_expression
            and self.target_type == other.target_type
            and self.type_params == other.type_params
            and self.array_type == other.array_type
        )

    def __str__(self) -> str:
        if self.type_params:
            type_args = ", ".join(str(arg) for arg in self.type_params)
            type_str = f"{self.target_type}({type_args})"
        else:
            type_str = self.target_type

        if self.array_type:
            type_str = f"{type_str}[]"

        casted_str = (
            str(self.casted_expression)
            if isinstance(
                self.casted_expression,
                (ColumnExpression, StaticValueExpression),
            )
            else f"({self.casted_expression})"
        )
        return f"CAST({casted_str} AS {type_str})"


class MathExpression(SqlExpression):
    """A mathematical expression computes a result value based on some formula.

    The formula is based on an arbitrary expression, an operator and potentially a number of additional
    expressions/arguments.

    If it is necessary to represent boolean expressions outside of the **WHERE** and **HAVING** clauses, a `AbstractPredicate`
    should be used instead of a mathematical expression.

    Parameters
    ----------
    operator : MathOperator
        The operator that is used to combine the arguments.
    first_argument : SqlExpression
        The first argument. For unary expressions, this can also be the only argument
    second_argument : SqlExpression | Sequence[SqlExpression] | None, optional
        Additional arguments. For the most common case of a binary expression, this will be exactly one argument.
        Defaults to *None* to accomodate for unary expressions.

        .. deprecated:: 0.21.1
            Passing a sequence of expressions is deprecated and will be removed in v0.22.0.
            To encode operations involving more than two operands, a hierarchy of binary expressions
            should be used instead.
    """

    @staticmethod
    def create_binary(
        lhs: Any, operator: MathOperator | str, rhs: Any
    ) -> MathExpression:
        """Shortcut method to create a new math expression.

        Lhs and rhs are processed depending on their actual types:

        - `SqlExpression` instances are left as-is
        - `ColumnReference` instances are mapped to `ColumnExpression`s
        - Anything else is wrapped in a `StaticValueExpression`
        """
        if isinstance(operator, str):
            operator = MathOperator(operator)
        lhs, rhs = as_expression(lhs), as_expression(rhs)
        return MathExpression(operator, lhs, rhs)

    def __init__(
        self,
        operator: MathOperator,
        first_argument: SqlExpression,
        second_argument: SqlExpression | Sequence[SqlExpression] | None = None,
    ) -> None:
        if isinstance(second_argument, Sequence):
            warnings.warn(
                "Passing a sequence of expressions is deprecated and will be removed in "
                "v0.22.0. To encode operations involving more than two operands, a hierarchy of "
                "binary expressions should be used instead.",
                category=DeprecationWarning,
            )
        if not operator or not first_argument:
            raise ValueError("Operator and first argument are required!")

        self._operator = operator
        self._first_arg = first_argument

        match second_argument:
            case SqlExpression():
                self._second_arg = second_argument
            case None:
                self._second_arg = None
            case Sequence():
                self._second_arg = util.simplify(second_argument)
                warnings.warn(
                    "Passing a sequence of expressions is deprecated and will be removed in "
                    "v0.22.0. To encode operations involving more than two operands, a"
                    "hierarchy of binary expressions should be used instead.",
                    category=DeprecationWarning,
                )
                if isinstance(self._second_arg, Sequence):
                    self._second_arg = tuple(self._second_arg)
            case _:
                raise ValueError(
                    "Second argument must be a single expression or a sequence of expressions, "
                    f"not '{second_argument}'"
                )

        if isinstance(self._second_arg, tuple) and len(self._second_arg) == 1:
            self._second_arg = self._second_arg[0]

        hash_val = hash((self._operator, self._first_arg, self._second_arg))
        super().__init__(hash_val)

    __slots__ = ("_operator", "_first_arg", "_second_arg")
    __match_args__ = ("operator", "first_arg", "second_arg")

    @property
    def operator(self) -> MathOperator:
        """Get the operation to combine the input value(s).

        Returns
        -------
        MathOperator
            The operator
        """
        return self._operator

    @property
    def first_arg(self) -> SqlExpression:
        """Get the first argument to the operator. This is always specified.

        Returns
        -------
        SqlExpression
            The argument
        """
        return self._first_arg

    @property
    def second_arg(self) -> SqlExpression | Sequence[SqlExpression] | None:
        """Get the second argument to the operator.

        Depending on the operator, this can be a single expression (the most common case), but also a sequence of
        expressions (e.g. sum of multiple values) or no value at all (e.g. negation).

        Returns
        -------
        SqlExpression | Sequence[SqlExpression] | None
            The argument(s)
        """
        return self._second_arg

    def is_unary(self) -> bool:
        """Checks, whether the expression is a unary one (e.g. a negation as in *-42*)."""
        return not self.second_arg

    def tables(self) -> set[TableReference]:
        all_tables = set(self.first_arg.tables())
        if isinstance(self.second_arg, list):
            all_tables |= util.set_union(expr.tables() for expr in self.second_arg)
        elif isinstance(self.second_arg, SqlExpression):
            all_tables |= self.second_arg.tables()
        return all_tables

    def columns(self) -> set[ColumnReference]:
        all_columns = set(self.first_arg.columns())
        if isinstance(self.second_arg, list):
            for expression in self.second_arg:
                all_columns |= expression.columns()
        elif isinstance(self.second_arg, SqlExpression):
            all_columns |= self.second_arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        first_columns = list(self.first_arg.itercolumns())
        if not self.second_arg:
            return first_columns

        match self.second_arg:
            case SqlExpression():
                second_columns = list(self.second_arg.itercolumns())
            case Sequence():
                second_columns = util.flatten(
                    [sub_arg.itercolumns() for sub_arg in self.second_arg]
                )
            case None:
                second_columns = []
        return first_columns + second_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        children = [self.first_arg]
        match self.second_arg:
            case SqlExpression():
                children.append(self.second_arg)
            case Sequence():
                children.extend(self.second_arg)
            case None:
                pass
        return children

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_math_expr(self, *args, **kwargs)

    def _requires_brackets(self, child: SqlExpression) -> bool:
        """Checks, whether some expression must be wrapped in brackets to ensure correct evaluation order."""
        if not isinstance(child, MathExpression):
            return False
        if child.operator == self.operator:
            return False
        if (
            self.operator == MathOperator.Concatenate
            or child.operator == MathOperator.Concatenate
        ):
            return True

        lazy_ops = {
            MathOperator.Add,
            MathOperator.Multiply,
            MathOperator.Modulo,
        }
        eager_ops = {MathOperator.Subtract, MathOperator.Divide}

        strict_brackets = self.operator in eager_ops and child.operator in lazy_ops

        # these brackets are not really required, but we still use them for better readability
        pretty_brackets = self.operator in lazy_ops and child.operator in eager_ops

        return strict_brackets or pretty_brackets

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.operator == other.operator
            and self.first_arg == other.first_arg
            and self.second_arg == other.second_arg
        )

    def __str__(self) -> str:
        operator_str = self.operator.value
        if not self.second_arg:
            return f"{operator_str}{self.first_arg}"
        if isinstance(self.second_arg, tuple):
            all_args = [self.first_arg] + list(self.second_arg)
            return operator_str.join(f"({arg})" for arg in all_args)
        first_str = (
            f"({self.first_arg})"
            if self._requires_brackets(self.first_arg)
            else str(self.first_arg)
        )

        assert isinstance(self.second_arg, SqlExpression)
        second_str = (
            f"({self.second_arg})"
            if self._requires_brackets(self.second_arg)
            else str(self.second_arg)
        )
        return f"{first_str} {operator_str} {second_str}"


class ColumnExpression(SqlExpression):
    """A column expression wraps the reference to a column.

    This is a leaf expression, i.e. a column expression cannot have any more child expressions. It corresponds directly
    to an access to the values of the wrapped column with no modifications.

    Parameters
    ----------
    column : ColumnReference
        The column being wrapped
    """

    @staticmethod
    def of(column_name: str, *, table: str | TableReference) -> ColumnExpression:
        """Shortcut method to create a new column reference + expression."""
        table = TableReference(table) if isinstance(table, str) else table
        return ColumnExpression(ColumnReference(column_name, table))

    def __init__(self, column: ColumnReference) -> None:
        if column is None:
            raise ValueError("Column cannot be none")
        self._column = column
        super().__init__(hash(self._column))

    __slots__ = ("_column",)
    __match_args__ = ("column",)

    @property
    def column(self) -> ColumnReference:
        """Get the column that is wrapped by this expression.

        Returns
        -------
        ColumnReference
            The column
        """
        return self._column

    def tables(self) -> set[TableReference]:
        if (table := self.column.table) is None:
            return set()
        return {table}

    def columns(self) -> set[ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[ColumnReference]:
        return [self.column]

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_column_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.column == other.column

    def __str__(self) -> str:
        return str(self.column)


AggregateFunctions = {
    # SQL standard aggregates
    "COUNT",
    "SUM",
    "MIN",
    "MAX",
    "AVG",
    "EVERY",
    "CORR",
    "STDDEV",
    # Postgres additions
    "ANY_VALUE",
    "ARRAY_AGG",
    "BIT_AND",
    "BIT_OR",
    "BIT_XOR",
    "BOOL_AND",
    "BOOL_OR",
    "BOOL_XOR",
    "STRING_AGG",
    "JSON_AGG",
    "XML_AGG",
}
"""All aggregate functions specified in standard SQL and Postgres."""


class FunctionExpression(SqlExpression):
    """The function expression indicates a call to an arbitrary function.

    The actual function might be one of the standard SQL functions, an aggregation function or a user-defined one.
    PostBOUND treats them all the same and it is up to the user to differentiate e.g. between UDFs and aggregations if
    this distinction is important. This can be easily achieved by introducing additional subclasses of the function
    expression and updating the queries to use the new function expressions where appropriate. The `transform` module
    provides utilities to make such updates easy.

    Parameters
    ----------
    function : str
        The name of the function that should be called. Cannot be empty.
    arguments : Optional[Sequence[SqlExpression]], optional
        The parameters that should be passed to the function. Can be *None* if the function does not take or does not
        need any arguments (e.g. ``CURRENT_TIME()``)
    distinct : bool, optional
        Whether the (aggregation) function should only operate on distinct column values and hence a duplicate
        elimination needs to be performed before passing the argument values (e.g. ``COUNT(DISTINCT *)``). Defaults to
        *False*
    filter_where : Optional[AbstractPredicate], optional
        An optional filter expression that restricts the values included in an aggregation function.

    See Also
    --------
    postbound.transform.replace_expressions
    """

    @staticmethod
    def create_count(
        column: ColumnReference
        | SqlExpression
        | Iterable[SqlExpression] = StarExpression(),
        *,
        distinct: bool = False,
    ) -> FunctionExpression:
        """Shortcut method to create a new *COUNT()*  expression over (one or multiple) columns.

        If no column is given, a *COUNT(\\*)* is created.
        """
        match column:
            case ColumnReference():
                args = [ColumnExpression(column)]
            case SqlExpression():
                args = [column]
            case Iterable():
                args = list(column)
        return FunctionExpression("count", args, distinct=distinct)

    @staticmethod
    def create_max(
        column: ColumnReference | SqlExpression | Iterable[SqlExpression],
    ) -> FunctionExpression:
        """Shortcut method to create a new *MAX()*  expression over (one or multiple) columns."""
        match column:
            case ColumnReference():
                args = [ColumnExpression(column)]
            case SqlExpression():
                args = [column]
            case Iterable():
                args = list(column)
        return FunctionExpression("max", args)

    @staticmethod
    def create_min(
        column: ColumnReference | SqlExpression | Iterable[SqlExpression],
    ) -> FunctionExpression:
        """Shortcut method to create a new *MIN()*  expression over (one or multiple) columns."""
        match column:
            case ColumnReference():
                args = [ColumnExpression(column)]
            case SqlExpression():
                args = [column]
            case Iterable():
                args = list(column)
        return FunctionExpression("min", args)

    @staticmethod
    def create_sum(
        column: ColumnReference | SqlExpression | Iterable[SqlExpression],
    ) -> FunctionExpression:
        """Shortcut method to create a new *SUM()*  expression over (one or multiple) columns."""
        match column:
            case ColumnReference():
                args = [ColumnExpression(column)]
            case SqlExpression():
                args = [column]
            case Iterable():
                args = list(column)
        return FunctionExpression("sum", args)

    def __init__(
        self,
        function: str,
        arguments: Optional[Sequence[SqlExpression]] = None,
        *,
        keyword_args: Optional[Mapping[str, SqlExpression]] = None,
        distinct: bool = False,
        filter_where: Optional[AbstractPredicate] = None,
    ) -> None:
        if not function:
            raise ValueError("Function is required")
        if function.upper() not in AggregateFunctions and (distinct or filter_where):
            raise ValueError(
                "DISTINCT keyword or FILTER expressions are only valid for aggregate functions"
            )

        self._function = function.upper()
        self._arguments: tuple[SqlExpression, ...] = (
            () if arguments is None else tuple(arguments)
        )
        self._kw_args: Mapping[str, SqlExpression] = (
            {}
            if keyword_args is None
            else {k.upper(): v for k, v in keyword_args.items()}
        )
        self._distinct = distinct
        self._filter_expr = filter_where

        kw_hashable = tuple((k, v) for k, v in self._kw_args.items())
        hash_val = hash(
            (
                self._function,
                self._distinct,
                self._arguments,
                kw_hashable,
                self._filter_expr,
            )
        )
        super().__init__(hash_val)

    __slots__ = (
        "_function",
        "_arguments",
        "_kw_args",
        "_distinct",
        "_filter_expr",
    )
    __match_args__ = (
        "function",
        "arguments",
        "keyword_args",
        "distinct",
        "filter_where",
    )

    @property
    def function(self) -> str:
        """Get the function name.

        Returns
        -------
        str
            The function name. Will never be empty
        """
        return self._function

    @property
    def arguments(self) -> Sequence[SqlExpression]:
        """Get all arguments that are supplied to the function.

        Returns
        -------
        Sequence[SqlExpression]
            The arguments. Can be empty if no arguments are passed (but will never be *None*).
        """
        return self._arguments

    @property
    def keyword_args(self) -> Mapping[str, SqlExpression]:
        """Get all keyword arguments that are supplied to the function.

        The order in the mapping matches the order of occurrence in the
        function call.
        """
        return self._kw_args

    @property
    def distinct(self) -> bool:
        """Get whether the function should only operate on distinct values.

        Whether this makes any sense for the function at hand is entirely dependend on the specific function and not
        enfored by PostBOUND. The runtime DBS has to check this.

        Generally speaking, this argument is intended for aggregation functions.

        Returns
        -------
        bool
            Whether a duplicate elimination has to be performed on the function arguments
        """
        return self._distinct

    @property
    def filter_where(self) -> Optional[AbstractPredicate]:
        """Get the filter expression for an aggregate function.

        Filters restrict the values that are actually included in the aggregate.

        Returns
        -------
        Optional[AbstractPredicate]
            The filter expression or *None* if no filter is applied (or the function is not an aggregate).
        """
        return self._filter_expr

    def is_aggregate(self) -> bool:
        """Checks, whether the function is a well-known SQL aggregation function.

        Only standard functions are considered (e.g. no CORR for computing correlations).

        Returns
        -------
        bool
            Whether the function is a known aggregate function.
        """
        return self._function.upper() in AggregateFunctions

    def tables(self) -> set[TableReference]:
        args_tables = util.set_union(arg.tables() for arg in self.arguments)
        kwargs_tables = util.set_union(
            arg.tables() for arg in self.keyword_args.values()
        )
        filter_tables = self._filter_expr.tables() if self._filter_expr else set()
        return args_tables | kwargs_tables | filter_tables

    def columns(self) -> set[ColumnReference]:
        all_columns = util.set_union(arg.columns() for arg in self.arguments)
        all_columns |= util.set_union(
            arg.columns() for arg in self.keyword_args.values()
        )
        if self._filter_expr:
            all_columns |= self._filter_expr.columns()
        return all_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        all_columns = []
        for arg in self.arguments:
            all_columns.extend(arg.itercolumns())
        for arg in self.keyword_args.values():
            all_columns.extend(arg.itercolumns())
        if self._filter_expr:
            all_columns.extend(self._filter_expr.itercolumns())
        return all_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        return list(self.arguments) + list(self._kw_args.values())

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_function_expr(self, *args, **kwargs)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.function == other.function
            and self.arguments == other.arguments
            and self.keyword_args == other.keyword_args
            and self.distinct == other.distinct
            and self.filter_where == other.filter_where
        )

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._arguments)
        if self.keyword_args:
            keywords = [f"{kw} {val}" for kw, val in self.keyword_args.items()]
            keyword_str = " ".join(keywords)
            args_str = f"{args_str} {keyword_str}" if args_str else keyword_str

        distinct_str = "DISTINCT " if self._distinct else ""
        if len(self._arguments) > 1 and self._distinct:
            # Postgres, DuckDB (and others?) require brackets around the arguments if DISTINCT is used with multiple arguments,
            # e.g. COUNT(DISTINCT (a, b))
            # Notice that this is NOT documented in the aggregate syntax under
            # https://www.postgresql.org/docs/current/sql-expressions.html#SYNTAX-AGGREGATES
            args_str = f"({args_str})"

        parameterization = f"({distinct_str}{args_str})"
        filter_str = f" FILTER (WHERE {self._filter_expr})" if self._filter_expr else ""
        return f"{self._function}{parameterization}{filter_str}"


class ArrayExpression(SqlExpression):
    """Models an array literal expression, such as ``ARRAY[1, 2, 3]``.

    Our array abstraction also permits the array to contain arbitrary expressions, as long as they are all of the same type
    (which we assume but cannot check), e.g. ``ARRAY[41, (SELECT 42), 43]``.

    Parameters
    ----------
    elements: Sequence[SqlExpression]
        The elements of the array. Notice that all elements have to be valid `SqlExpression` instances, raw values are not
        permitted.
    """

    def __init__(self, elements: Sequence[SqlExpression]) -> None:
        raw_elements = [
            (idx, elem)
            for idx, elem in enumerate(elements)
            if not isinstance(elem, SqlExpression)
        ]
        if raw_elements:
            details = ", ".join(
                f"{idx}: {type(elem)} ({elem})" for idx, elem in raw_elements
            )
            raise TypeError(
                "Cannot create ArrayExpression. ",
                f"All elements must be SqlExpression instances, but found invalid elements at positions: {details}",
            )

        self._elements = tuple(elements)
        hash_val = hash(self._elements)
        super().__init__(hash_val)

    __slots__ = ("_elements",)
    __match_args__ = ("elements",)

    @property
    def elements(self) -> Sequence[SqlExpression]:
        """Get the elements of the array."""
        return self._elements

    def tables(self) -> set[TableReference]:
        return util.set_union(elem.tables() for elem in self._elements)

    def columns(self) -> set[ColumnReference]:
        return util.set_union(elem.columns() for elem in self._elements)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(elem.itercolumns() for elem in self._elements)

    def iterchildren(self) -> Iterable[SqlExpression]:
        return list(self._elements)

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_array_expr(self, *args, **kwargs)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.elements == other.elements

    def __str__(self) -> str:
        elements_str = ", ".join(str(elem) for elem in self._elements)
        return f"ARRAY[{elements_str}]"


class ArrayAccessExpression(FunctionExpression):
    """Models index-based access to an array column.

    Due to its oftentimes special syntax, this is modeled as a special case of a function expression (using ``ARRAY_GET`` as
    the function name). The text representation is based on Postgres and should be adapted for other systems during query
    formatting if necessary.

    Depending on the specific kind of access, different parameters can be set. For simple element access, the `index` attribute
    is used. For slices, the `lower_index` and `upper_index` attributes are available. It is also possible to only set one
    boundary to create a half-open slice. Whether the index is 0-based or 1-based is not enforced by PostBOUND and depends
    on the actual database system.

    Notice that all indexes are represented as expressions rather than simple integers. This allows for "variable" indexes, as
    in ``SELECT R.a[R.b] FROM R``.

    Parameters
    ----------
    array_expr : SqlExpression
        The array being accessed
    idx : Optional[SqlExpression], optional
        For point-based access, the index of the element to access. Defaults to *None*.
    lower_idx : Optional[SqlExpression], optional
        For slice-based access, the lower boundary of the slice. Defaults to *None*.
    upper_idx : Optional[SqlExpression], optional
        For slice-based access, the upper boundary of the slice. Defaults to *None*.
    """

    def __init__(
        self,
        array_expr: SqlExpression,
        *,
        idx: Optional[SqlExpression] = None,
        lower_idx: Optional[SqlExpression] = None,
        upper_idx: Optional[SqlExpression] = None,
    ) -> None:
        if idx is None and lower_idx is None and upper_idx is None:
            raise ValueError("At least one index has to be specified")
        if idx is not None and (lower_idx is not None or upper_idx is not None):
            raise ValueError("Cannot specify both a single index and a slice")

        self._array = array_expr
        self._idx = idx
        self._lower_idx = lower_idx
        self._upper_idx = upper_idx
        self._hash_val = hash(
            (self._array, self._idx, self._lower_idx, self._upper_idx)
        )

        args = [
            StaticValueExpression.null() if arg is None else arg
            for arg in (array_expr, idx, lower_idx, upper_idx)
        ]
        super().__init__("ARRAY_GET", args)

    __slots__ = ("_array", "_idx", "_lower_idx", "_upper_idx", "_hash_val")
    __match_args__ = ("array", "index", "lower_index", "upper_index")

    @property
    def array(self) -> SqlExpression:
        """Get the array that is being accessed.

        Returns
        -------
        SqlExpression
            The array
        """
        return self._array

    @property
    def index(self) -> Optional[SqlExpression]:
        """Get the index of the element to access.

        Returns
        -------
        Optional[SqlExpression]
            The index or *None* if sliced access is used
        """
        return self._idx

    @property
    def lower_index(self) -> Optional[SqlExpression]:
        """Get the lower boundary of the slice.

        Returns
        -------
        Optional[SqlExpression]
            The lower boundary or *None* if either point-based access is used, or the slice is open at the lower end
        """
        return self._lower_idx

    @property
    def upper_index(self) -> Optional[SqlExpression]:
        """Get the upper boundary of the slice.

        Returns
        -------
        Optional[SqlExpression]
            The upper boundary or *None* if either point-based access is used, or the slice is open at the upper end
        """
        return self._upper_idx

    @property
    def index_slice(
        self,
    ) -> Optional[tuple[Optional[SqlExpression], Optional[SqlExpression]]]:
        """Get the boundaries of the slice.

        Returns
        -------
        Optional[tuple[Optional[SqlExpression], Optional[SqlExpression]]]
            The slice interval. Any boundaries can be none if the interval is open at that end. If the array access is not
            sliced, the entire tuple is *None*.
        """
        if self._idx:
            return None
        return self._lower_idx, self._upper_idx

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_array_access_expr(self, *args, **kwargs)

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self._array == other._array
            and self._idx == other._idx
            and self._lower_idx == other._lower_idx
            and self._upper_idx == other._upper_idx
        )

    def __hash__(self):
        return self._hash_val

    def __str__(self) -> str:
        if self._idx is not None:
            index_str = f"[{self._idx}]"
        elif self._lower_idx is not None and self._upper_idx is not None:
            index_str = f"[{self._lower_idx}:{self._upper_idx}]"
        elif self._lower_idx is not None:
            index_str = f"[{self._lower_idx}:]"
        else:
            index_str = f"[:{self._upper_idx}]"
        return f"({self._array}){index_str}"


class SubqueryExpression(SqlExpression):
    """A subquery expression wraps an arbitrary subquery.

    This expression can be used in two different contexts: as table source to produce a virtual temporary table for
    reference in the query (see the `clauses` module), or as a part of a predicate. In the latter scenario the
    subqueries' results are transient for the rest of the query. Therefore, this expression only represents the
    subquery part but no name under which the query result can be accessed. This is added by the different parts of the
    `clauses` module (e.g. `WithQuery` or `SubqueryTableSource`).

    This is a leaf expression, i.e. a subquery expression cannot have any more child expressions. However, the subquery itself
    likely consists of additional expressions.

    Parameters
    ----------
    subquery : SqlQuery
        The subquery that forms this expression

    """

    def __init__(self, subquery: SqlQuery) -> None:
        self._query = subquery
        super().__init__(hash(subquery))

    __slots__ = ("_query",)
    __match_args__ = ("query",)

    @property
    def query(self) -> SqlQuery:
        """The (sub)query that is wrapped by this expression.

        Returns
        -------
        SqlQuery
            The query
        """
        return self._query

    def tables(self) -> set[TableReference]:
        return self._query.tables()

    def columns(self) -> set[ColumnReference]:
        return self._query.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._query.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_subquery_expr(self, *args, **kwargs)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._query == other._query

    def __str__(self) -> str:
        query_str = str(self._query).removesuffix(";")
        return f"({query_str})"


class WindowExpression(SqlExpression):
    """Represents a window expression in SQL.

    Parameters
    ----------
    window_function : FunctionExpression | str
        The window function to be applied. If this is a string, it will be wrapped in a plain
        function expression.
    partitioning : Optional[Sequence[SqlExpression | ColumnReference]], optional
        The expressions used for partitioning the window. Defaults to None.
    ordering : Optional[OrderBy], optional
        The ordering of the window. Defaults to None.
    filter_condition : Optional[AbstractPredicate], optional
        The filter condition for the window. Defaults to None.
    """

    def __init__(
        self,
        window_function: FunctionExpression | str,
        *,
        partitioning: Optional[Sequence[SqlExpression | ColumnReference]] = None,
        ordering: Optional[OrderBy] = None,
        filter_condition: Optional[AbstractPredicate] = None,
    ) -> None:
        if isinstance(window_function, str):
            window_function = FunctionExpression(window_function)
        if isinstance(partitioning, Sequence):
            partitioning = [
                ColumnExpression(p) if isinstance(p, ColumnReference) else p
                for p in partitioning
            ]

        self._window_function = window_function
        self._partitioning = tuple(partitioning) if partitioning else tuple()
        self._ordering = ordering
        self._filter_condition = filter_condition

        hash_val = hash(
            (
                self._window_function,
                self._partitioning,
                self._ordering,
                self._filter_condition,
            )
        )
        super().__init__(hash_val)

    __slots__ = (
        "_window_function",
        "_partitioning",
        "_ordering",
        "_filter_condition",
    )
    __match_args__ = (
        "window_function",
        "partitioning",
        "ordering",
        "filter_condition",
    )

    @property
    def window_function(self) -> FunctionExpression:
        """Get the window function of the window expression.

        Returns
        -------
        FunctionExpression
            The window function.
        """
        return self._window_function

    @property
    def partitioning(self) -> Sequence[SqlExpression]:
        """Get the expressions used for partitioning the window.

        Returns
        -------
        Sequence[SqlExpression]
            The expressions used for partitioning the window. Can be empty if no partitioning is used.
        """
        return self._partitioning

    @property
    def ordering(self) -> Optional[OrderBy]:
        """Get the ordering of tuples in the current window.

        Returns
        -------
        Optional[OrderBy]
            The ordering of the tuples, or *None* if no ordering is specified.
        """
        return self._ordering

    @property
    def filter_condition(self) -> Optional[AbstractPredicate]:
        """Get the filter condition for tuples in the current window.

        Returns:
            Optional[AbstractPredicate]:
            The filter condition for the expression, or *None* if all tuples are aggegrated.
        """
        return self._filter_condition

    def tables(self) -> set[TableReference]:
        return util.set_union(child.tables() for child in self.iterchildren())

    def columns(self) -> set[ColumnReference]:
        return util.set_union(child.columns() for child in self.iterchildren())

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(expr.itercolumns() for expr in self.iterchildren())

    def iterchildren(self) -> Iterable[SqlExpression]:
        function_children = self.window_function.iterchildren()
        partitioning_children = util.flatten(
            expr.iterchildren() for expr in self.partitioning
        )
        ordering_children = self.ordering.iterexpressions() if self.ordering else []
        filter_children = (
            self.filter_condition.iterexpressions() if self.filter_condition else []
        )
        return util.flatten(
            [
                function_children,
                partitioning_children,
                ordering_children,
                filter_children,
            ]
        )

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_window_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.window_function == other.window_function
            and self.partitioning == other.partitioning
            and self.ordering == other.ordering
            and self.filter_condition == other.filter_condition
        )

    def __str__(self) -> str:
        filter_str = (
            f" FILTER (WHERE {self.filter_condition})" if self.filter_condition else ""
        )
        function_str = f"{self.window_function}{filter_str} OVER"
        window_grouping: list[str] = []
        if self.partitioning:
            partitioning_str = ", ".join(
                str(partition) for partition in self.partitioning
            )
            window_grouping.append(f"PARTITION BY {partitioning_str}")
        if self.ordering:
            window_grouping.append(str(self.ordering))
        window_str = " ".join(window_grouping)
        window_str = f"({window_str})" if window_str else "()"
        return f"{function_str} {window_str}"


class CaseExpression(SqlExpression):
    """Represents a case expression in SQL.

    Parameters:
    -----------
    cases : Sequence[tuple[SqlExpression, SqlExpression]]
        A sequence of tuples representing the cases in the case expression. The cases are passed as a sequence rather than a
        dictionary, because the evaluation order of the cases is important. The first case that evaluates to true determines
        the result of the entire case statement.
    simple_expr: Optional[SqlExpression], optional
        The expression to evaluate against the cases. This "simple form" compares the `expression` directly against each
        of the values in `cases`, similar to a switch statement.
    else_expr : Optional[SqlExpression], optional
        The expression to be evaluated if none of the cases match. If no case matches and no else expression is provided, the
        entire case expression should evaluate to NULL.
    """

    def __init__(
        self,
        cases: Sequence[tuple[SqlExpression, SqlExpression]],
        *,
        simple_expr: Optional[SqlExpression] = None,
        else_expr: Optional[SqlExpression] = None,
    ) -> None:
        if not cases:
            raise ValueError("At least one case is required")
        self._cases = tuple(cases)
        self._simple_expr = simple_expr
        self._else_expr = else_expr

        hash_val = hash((self._cases, self._simple_expr, self._else_expr))
        super().__init__(hash_val)

    __slots__ = ("_cases", "_simple_expr", "_else_expr")
    __match_args__ = ("cases", "simple_expression", "else_expression")

    @property
    def cases(self) -> Sequence[tuple[SqlExpression, SqlExpression]]:
        """Get the different cases.

        Returns
        -------
        Sequence[tuple[SqlExpression]]
            The cases. At least one case will be present.
        """
        return self._cases

    @property
    def simple_expression(self) -> Optional[SqlExpression]:
        """Get the expression to evaluate against the cases.

        This is only set for the "simple form" of the case expression, where the expression is compared directly against
        the values in the cases. In this form, each case has to be a plain value instead of a full predicate, similar to a
        switch statement:

        .. code-block:: sql

            SELECT  CASE R.a
                        WHEN 1 THEN 'one'
                        WHEN 2 THEN 'two'
                        ELSE 'other'
                    END AS foo
            FROM R;

        """
        return self._simple_expr

    @property
    def else_expression(self) -> Optional[SqlExpression]:
        """Get the expression to use if none of the cases match.

        Returns
        -------
        Optional[SqlExpression]
            The expression. Can be *None*, in which case the case expression evaluates to *NULL*.
        """
        return self._else_expr

    def tables(self) -> set[TableReference]:
        return util.set_union(child.tables() for child in self.iterchildren())

    def columns(self) -> set[ColumnReference]:
        return util.set_union(child.columns() for child in self.iterchildren())

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(expr.itercolumns() for expr in self.iterchildren())

    def iterchildren(self) -> Iterable[SqlExpression]:
        case_children = util.flatten(
            list(pred.iterchildren()) + list(expr.iterchildren())
            for pred, expr in self.cases
        )
        expression_children = (
            list(self.simple_expression.iterchildren())
            if self.simple_expression
            else []
        )
        else_children = (
            list(self.else_expression.iterchildren()) if self.else_expression else []
        )
        return case_children + expression_children + else_children

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_case_expr(self, *args, **kwargs)

    def _braketify(self, expression: SqlExpression) -> str:
        """Wraps the given expression in brackets if necessary."""
        if isinstance(expression, (CaseExpression, MathExpression)):
            return f"({expression})"
        return str(expression)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.cases == other.cases
            and self.simple_expression == other.simple_expression
            and self.else_expression == other.else_expression
        )

    def __str__(self) -> str:
        expression = f"{self.simple_expression} " if self.simple_expression else ""
        cases_str = " ".join(
            f"WHEN {pred} THEN {self._braketify(expr)}" for pred, expr in self.cases
        )
        else_str = (
            f" ELSE {self._braketify(self.else_expression)}"
            if self.else_expression
            else ""
        )
        return f"CASE {expression}{cases_str}{else_str} END"


class QuantifierExpression(SqlExpression):
    """An ANY/ALL expression.

    For a predicate such as ``R.a > ALL (SELECT b FROM S)`` this expression is used to represent the right-hand side of the
    predicate. It typically appears as a child expression of a `BinaryPredicate`.

    Parameters
    ----------
    expression : SqlExpression
        The expression being quantified. Typically this is a `SubqueryExpression`, but some database systems also allow
        comparing against arrays or other collection types.
    quantifier : QuantifierOperator
        The quantifier operator (_ANY_ or _ALL_).
    """

    @staticmethod
    def any(expression: SqlExpression | SqlQuery) -> QuantifierExpression:
        """Create an ANY expression."""
        expression = (
            SubqueryExpression(expression)
            if isinstance(expression, SqlQuery)
            else expression
        )
        return QuantifierExpression(expression, quantifier=QuantifierOperator.Any)

    @staticmethod
    def all(expression: SqlExpression | SqlQuery) -> QuantifierExpression:
        """Create an ALL expression."""
        expression = (
            SubqueryExpression(expression)
            if isinstance(expression, SqlQuery)
            else expression
        )
        return QuantifierExpression(expression, quantifier=QuantifierOperator.All)

    def __init__(
        self, expression: SqlExpression, *, quantifier: QuantifierOperator
    ) -> None:
        self._expression = expression
        self._quantifier = quantifier

        hash_val = hash((self._expression, self._quantifier))
        super().__init__(hash_val)

    __slots__ = ("_expression", "_quantifier")
    __match_args__ = ("expression", "quantifier")

    @property
    def expression(self) -> SqlExpression:
        """Get the expression that is being compared.

        Typically, this will be a `SubqueryExpression` that produces a relation with a single column. Some database systems
        also allow comparing against arrays or other collection types.
        """
        return self._expression

    @property
    def quantifier(self) -> QuantifierOperator:
        """Get the actual quantifier."""
        return self._quantifier

    def tables(self) -> set[TableReference]:
        return self._expression.tables()

    def columns(self) -> set[ColumnReference]:
        return self._expression.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._expression.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self._expression]

    def accept_visitor(
        self, visitor: SqlExpressionVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_quantifier_expr(self, *args, **kwargs)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.expression == other.expression
            and self.quantifier == other.quantifier
        )

    def __str__(self) -> str:
        return f"{self.quantifier.value} ({self.expression})"


class SqlExpressionVisitor(abc.ABC, Generic[VisitorResult]):
    """Basic visitor to operator on arbitrary expression trees.

    See Also
    --------
    SqlExpression

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_static_value_expr(
        self, expr: StaticValueExpression, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_cast_expr(self, expr: CastExpression, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_math_expr(self, expr: MathExpression, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_column_expr(
        self, expr: ColumnExpression, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_function_expr(
        self, expr: FunctionExpression, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_subquery_expr(
        self, expr: SubqueryExpression, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_star_expr(self, expr: StarExpression, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_window_expr(
        self, expr: WindowExpression, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_case_expr(self, expr: CaseExpression, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_quantifier_expr(
        self, expr: QuantifierExpression, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_predicate_expr(
        self, expr: AbstractPredicate, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_array_expr(self, expr: ArrayExpression, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    def visit_array_access_expr(
        self, expr: ArrayAccessExpression, *args, **kwargs
    ) -> VisitorResult:
        return self.visit_function_expr(expr, *args, **kwargs)

    def visit_substring_expr(
        self, expr: SubstringExpression, *args, **kwargs
    ) -> VisitorResult:
        return self.visit_function_expr(expr, *args, **kwargs)


class ExpressionCollector(SqlExpressionVisitor[set[SqlExpression]]):
    """Utility to traverse an arbitrarily deep expression hierarchy in order to collect specific expressions.

    Parameters
    ----------
    matcher : Callable[[SqlExpression], bool]
        Function to determine whether a specific expression matches the collection predicate. Should return *True* for matches
        and *False* otherwise.
    continue_after_match : bool, optional
        Whether the traversal of the current expression element should be continued if the current element matches the
        collection predicate. By default, traversal is stopped for the current element (but other branches in the expression
        tree could still produce more matches).
    """

    def __init__(
        self,
        matcher: Callable[[SqlExpression], bool],
        *,
        continue_after_match: bool = False,
    ) -> None:
        self.matcher = matcher
        self.continue_after_match = continue_after_match

    def visit_column_expr(
        self, expr: ColumnExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_cast_expr(
        self, expr: CastExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_function_expr(
        self, expr: FunctionExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_math_expr(
        self, expr: MathExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_star_expr(
        self, expr: StarExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_static_value_expr(
        self, expr: StaticValueExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_subquery_expr(
        self, expr: SubqueryExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_window_expr(
        self, expr: WindowExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_case_expr(
        self, expr: CaseExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_quantifier_expr(
        self, expr: QuantifierExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_array_expr(
        self, expr: ArrayExpression, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def visit_predicate_expr(
        self, expr: AbstractPredicate, *args, **kwargs
    ) -> set[SqlExpression]:
        return self._check_match(expr)

    def _check_match(self, expression: SqlExpression) -> set[SqlExpression]:
        """Handler to perform the actual traversal.

        Parameters
        ----------
        expression : SqlExpression
            The current expression

        Returns
        -------
        set[SqlExpression]
            All matching expressions
        """
        own_match = {expression} if self.matcher(expression) else set()
        if own_match and not self.continue_after_match:
            return own_match
        return own_match | util.set_union(
            child.accept_visitor(self) for child in expression.iterchildren()
        )


def as_expression(
    value: object,
    allow_star: bool = True,
) -> SqlExpression:
    """Transforms the given value into the most appropriate `SqlExpression` instance.

    This is a heuristic utility method that applies the following rules:

    - All instances of `SqlExpression` are left unmodified.
    - `ColumnReference` becomes `ColumnExpression`
    - `SqlQuery` becomes `SubqueryExpression`
    - The star-string ``*`` becomes a `StarExpression` if `allow_star` is enabled, otherwise
      it becomes a static value of the literal "star-string" ('\\*')

    All other values become a `StaticValueExpression` of that value.

    Parameters
    ----------
    value : object
        The object to be transformed into an expression
    allow_star : bool, optional
        Whether a literal \\* argument should be treated as an SQL \\* (as in
        *SELECT \\**) expression or as a string literal (as in *SELECT '\\*'*).
        By default, it is treated as the expression rather than the string
        literal.

    Returns
    -------
    SqlExpression
        The most appropriate expression object according to the transformation rules

    Examples
    --------
    >>> as_expression(ColumnReference("a"))
    a
    >>> as_expression("a")
    'a'
    """
    if isinstance(value, SqlExpression):
        return value

    if isinstance(value, ColumnReference):
        return ColumnExpression(value)

    if isinstance(value, SqlQuery):
        return SubqueryExpression(value)

    if value == "*" and allow_star:
        return StarExpression()
    return StaticValueExpression(value)


def as_func_expr(func_name: str, *args, **kwargs) -> FunctionExpression:
    """Utility to create a function expression from a function name and arbitrary arguments.

    All positional arguments are automatically transformed into the appropriate expression using
    `as_expression` with one notable exception: mapping types (e.g., dictionaries) are instead
    treated as collections of keyword arguments (see below). This allows to specify keyword
    arguments that are not allowed as keyword arguments in Python.

    All keyword arguments are treated as keyword arguments for the function expression. Their values
    are transformed using `as_expression` whereas the keys become the keywords.

    Examples
    --------

    >>> as_func_expr("sum", ColumnReference("a"))
    sum(a)
    >>> as_func_expr("substring", ColumnReference("a"), {"from": 1, "for": 3})
    substring(a FROM 1 FOR 3)
    """
    positional_args: list[SqlExpression] = []
    keyword_args: dict[str, SqlExpression] = {}

    for arg in args:
        if isinstance(arg, Mapping):
            parsed_args = {key: as_expression(val) for key, val in arg.items()}
            keyword_args.update(parsed_args)
        else:
            positional_args.append(as_expression(arg))

    parsed_kwargs = {key: as_expression(val) for key, val in kwargs.items()}
    keyword_args.update(parsed_kwargs)

    return FunctionExpression(func_name, positional_args, keyword_args=keyword_args)


def as_math_expr(
    lhs: object, operator: MathOperator | str, rhs: object
) -> MathExpression:
    """Utility to create a binary math expression.

    The left-hand side and right-hand side arguments are transformed using `as_expression`. The
    operator can be specified as a full operator object, or as the corresponding string value.

    Examples
    --------
    >>> as_math_expr(ColumnReference("a"), "+", ColumnReference("b"))
    a + b
    >>> as_math_expr(ColumnReference("a"), MathOperator.Subtract, 42)
    a - 42
    """
    operator = MathOperator(operator) if isinstance(operator, str) else operator
    return MathExpression(operator, as_expression(lhs), as_expression(rhs))


def _normalize_join_pair(
    columns: tuple[ColumnReference, ColumnReference],
) -> tuple[ColumnReference, ColumnReference]:
    """Normalizes the given join such that a pair ``(R.a, S.b)`` and ``(S.b, R.a)`` can be recognized as equal.

    Normalization in this context means that the order in which two appear is always the same. Therefore, this method
    essentially forces a swap of the columns if necessary, making use of the ability to sort columns lexicographically.

    Parameters
    ----------
    columns : tuple[ColumnReference, ColumnReference]
        The join pair to normalize

    Returns
    -------
    tuple[ColumnReference, ColumnReference]
        The normalized join pair.
    """
    first_col, second_col = columns
    return (second_col, first_col) if second_col < first_col else columns


class NoJoinPredicateError(StateError):
    """Error to indicate that a filter predicate was supplied at a place where a join predicate was expected.

    Parameters
    ----------
    predicate : AbstractPredicate | None, optional
        The predicate that caused the error, defaults to *None*
    """

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


class NoFilterPredicateError(StateError):
    """Error to indicate that a join predicate was supplied at a place where a filter predicate was expected.

    Parameters
    ----------
    predicate : AbstractPredicate | None, optional
        The predicate that caused the error, defaults to *None*.
    """

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


BaseExpression = ColumnExpression | StaticValueExpression | SubqueryExpression
"""Supertype that captures all expression types that can be considered base expressions for predicates."""


def _collect_base_expressions(
    expression: SqlExpression,
) -> Iterable[BaseExpression]:
    """Provides all base expressions that are contained in a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    Iterable[BaseExpression]
        The base expressions
    """
    if isinstance(
        expression,
        (ColumnExpression, StaticValueExpression, SubqueryExpression),
    ):
        return [expression]
    return util.flatten(
        _collect_base_expressions(child_expr)
        for child_expr in expression.iterchildren()
    )


def _collect_subquery_expressions(
    expression: SqlExpression,
) -> Iterable[SubqueryExpression]:
    """Provides all subquery expressions that are contained in a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    Iterable[SubqueryExpression]
        All subqueries that are contained in some level in the expression
    """
    return [
        child_expr
        for child_expr in _collect_base_expressions(expression)
        if isinstance(child_expr, SubqueryExpression)
    ]


def _collect_column_expression_columns(
    expression: SqlExpression,
) -> set[ColumnReference]:
    """Provides all columns that are directly contained in `ColumnExpression` instances with a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree. Notice that it ignores all
    expressions that are part of subqueries.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    set[ColumnReference]
        All columns that are referenced in `ColumnExpression`s
    """
    return util.set_union(
        base_expr.columns()
        for base_expr in _collect_base_expressions(expression)
        if isinstance(base_expr, ColumnExpression)
    )


def _collect_column_expression_tables(
    expression: SqlExpression,
) -> set[TableReference]:
    """Provides all tables that are linked directly in `ColumnExpression` instances with a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree. Notice that it ignores all
    expressions that are part of subqueries.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    set[TableReference]
        All tables that are referenced in the columns of `ColumnExpression`s
    """
    return {
        column.table
        for column in _collect_column_expression_columns(expression)
        if column.table is not None
    }


def _generate_join_pairs(
    first_columns: Iterable[ColumnReference],
    second_columns: Iterable[ColumnReference],
) -> set[tuple[ColumnReference, ColumnReference]]:
    """Provides all possible pairs of columns where each column comes from a different iterable.

    Essentially, this produces the cross product of the two column sets. The join pairs are normalized and duplicate
    elimination is performed (this is necessary since columns can appear in both iterables). Likewise, "joins" over the same
    logical relations are also skipped.

    Parameters
    ----------
    first_columns : Iterable[ColumnReference]
        Candidate columns to be joined with `second_columns`
    second_columns : Iterable[ColumnReference]
        The join partner columns

    Returns
    -------
    set[tuple[ColumnReference, ColumnReference]]
        All normalized join pairs
    """
    return {
        _normalize_join_pair((first_col, second_col))
        for first_col, second_col in itertools.product(first_columns, second_columns)
        if first_col.table != second_col.table
    }


class AbstractPredicate(SqlExpression, abc.ABC):
    """Base class for all predicates.

    Predicates constitute the central building block for *WHERE* and *HAVING* clauses and model the join conditions in
    explicit joins using the *JOIN ON* syntax.

    The different kinds of predicates are represented as subclasses of the `AbstractPredicate` interface. This really is an
    abstract interface, not a usable predicate and it only specifies the behaviour that is shared among all specific predicate
    implementations. All inheriting classes have to implement their own `__eq__` method and inherit the `__hash__` method
    specified by the abstract predicate. Remember to explicitly set this up! The concrete hash value is constant since the
    clause itself is immutable. It is up to the implementing class to make sure that the equality/hash consistency is enforced.

    Possible implementations of the abstract predicate can model basic binary predicates such as  ``R.a = S.b`` or
    ``R.a = 42``, as well as compound predicates that are build form base predicates, e.g. conjunctions, disjunctions or
    negations.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete predicate object
    """

    def __init__(self, hash_val: int) -> None:
        super().__init__(hash_val)

    @abc.abstractmethod
    def is_compound(self) -> bool:
        """Checks, whether this predicate combines the evaluation of other predicates to compute the overall evaluation.

        Operators to combine such predicates can be standard logical operators like conjunction, disjunction and negation. This
        method serves as a high-level check, preventing the usage of dedicated ``isinstance`` calls in some use-cases.

        Returns
        -------
        bool
            Whethter this predicate is a composite of other predicates.
        """
        raise NotImplementedError

    def is_base(self) -> bool:
        """Checks, whether this predicate forms a leaf in the predicate tree, i.e. does not contain any more child predicates.

        This is the case for basic binary predicates, *IN* predicates, etc. This method serves as a high-level check,
        preventing the usage of dedicated ``isinstance`` calls in some use-cases.

        Returns
        -------
        bool
            Whether this predicate is a base predicate
        """
        return not self.is_compound()

    @abc.abstractmethod
    def is_join(self) -> bool:
        """Checks, whether this predicate encodes a join between two tables.

        PostBOUND uses the following criteria to determine, whether a predicate is join or not:

        1. all predicates of the form ``<col 1> <operator> <col 2>`` where ``<col 1>`` and ``<col 2>`` come from different
           tables are joins. The columns can optionally be modified by value casts or static expressions, e.g.
           ``R.a::integer + 7``
        2. all functions that access columns from multiple tables are joins, e.g. ``my_udf(R.a, S.b)``
        3. all subqueries are treated as filters, no matter whether they are dependent subqueries or not. This means that both
           ``R.a = (SELECT MAX(S.b) FROM S)`` and ``R.a = (SELECT MAX(S.b) FROM S WHERE R.c = S.d)`` are treated as filters and
           not as joins, even though the second subquery will require some sort of the join in the query plan.
        4. *BETWEEN* and *IN* predicates are treated according to rule 1 since they can be emulated via base predicates
           (subqueries in *IN* predicates are evaluated according to rule 3.)

        Although these rules might seem a bit arbitrary at first, there is actually no clear consensus of what constitutes a
        join and the query optimizers of different industrial database systems treat different predicates as joins. For
        example, some systems might not recognize function calls that access columns form two or more tables as joins or do not
        recognize predicates that use non-equi joins as opertors as actual joins.

        If the specific join and filter recognition procedure breaks a specific use-case, subclasses of the predicate classes
        can be implemented. These subclasses can then apply the required rules. Using the tools in the `transformation`
        module, the queries can be updated. For some use-cases it can also be sufficient to change the join/filter recognition
        rules of the `QueryPredicates` objects. Consult its documentation for more details.

        Lastly, notice that the distinction between join and filter is not entirely binary. There may also be a third class of
        predicates, potentially called "post-join filters". These are filters that are applied after a join but cannot be
        included in the join predicate itself. This is usually the case due to limitations in the operator implementation of
        the actual database system. For example, invocations of user defined functions (case 2 above) usually fall in this
        category. Since the query abstraction layer is agnostic to specific details of database systems, we apply the binary
        categorization outlined above.

        Returns
        -------
        bool
            Whether the predicate is a join of different relations.
        """
        raise NotImplementedError

    def is_filter(self) -> bool:
        """Checks, whether this predicate encodes a filter on a base table rather than a join of base tables.

        This is the inverse method to `is_join`. Consult it for more details on the join/filter recognition procedure.

        Returns
        -------
        bool
            Whether the predicate is a filter of some relation
        """
        return not self.is_join()

    def tables(self) -> set[TableReference]:
        """Provides all tables that are accessed by this predicate.

        Notice that even for filters, the provided set might contain multiple entries, e.g. if the predicate contains
        subqueries.

        Returns
        -------
        set[TableReference]
            All tables. This can include virtual tables if such tables are referenced in the predicate.
        """
        return util.set_union(e.tables() for e in self.iterexpressions())

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        Returns
        -------
        set[ColumnReference]
            The columns. If the predicate contains a subquery, all columns of that query are included.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        If a column is referenced multiple times, it is also returned multiple times.

        Returns
        -------
        Iterable[ColumnReference]
            All columns in exactly the order in which they are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all expressions that are directly contained in this predicate.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    def iterchildren(self) -> Iterable[SqlExpression]:
        return self.iterexpressions()

    def contains_table(self, table: TableReference) -> bool:
        """Checks, whether this predicate filters or joins a column of a specific table.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the given `table` is referenced by any of the columns in the predicate.
        """
        return any(table == tab for tab in self.tables())

    def joins_table(self, table: TableReference) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is a specific table.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the given `table` is one of the join partners in the predicate.
        """
        if not self.is_join():
            return False
        return any(
            first_col.belongs_to(table) or second_col.belongs_to(table)
            for first_col, second_col in self.join_partners()
        )

    def joins_tables(self, a: TableReference, b: TableReference) -> bool:
        """Checks, whether this predicate describes a join between the given tables.

        The order of the tables does not matter.
        """
        if not self.is_join():
            return False
        given = {a, b}
        return given.issubset(self.tables())

    def joins_columns(self, a: ColumnReference, b: ColumnReference) -> bool:
        """Checks, whether this predicate describes a join between the given columns.

        The order of the columns does not matter.
        """
        if not self.is_join():
            return False
        given = {a, b}
        return given.issubset(self.columns())

    def columns_of(self, table: TableReference) -> set[ColumnReference]:
        """Retrieves all columns of a specific table that are referenced by this predicate.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        set[ColumnReference]
            All columns in this predicate that belong to the given `table`
        """
        return {col for col in self.columns() if col.belongs_to(table)}

    def join_partners_of(self, table: TableReference) -> set[ColumnReference]:
        """Retrieves all columns that are joined with a specific table.

        Parameters
        ----------
        table : TableReference
            The table for which the join partners should be searched

        Returns
        -------
        set[ColumnReference]
            The columns that are joined with the given `table`

        Raises
        ------
        NoJoinPredicateError
        """
        partners = []
        for first_col, second_col in self.join_partners():
            if first_col.belongs_to(table):
                partners.append(second_col)
            elif second_col.belongs_to(table):
                partners.append(first_col)
        return set(partners)

    @abc.abstractmethod
    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        """Provides all pairs of columns that are joined within this predicate.

        If multiple columns are joined or it is unclear which columns are involved in a join exactly, this method falls back to
        returning the cross-product of all potential join partners. For example, consider the following query:
        ``SELECT * FROM R, S WHERE my_udf(R.a, R.b, S.c)``. In this case, it cannot be determined which columns of *R* take
        part in the join. Therefore, `join_partners` will return the set ``{(R.a, S.c), (R.b, S.c)}``.

        Returns
        -------
        set[tuple[ColumnReference, ColumnReference]]
            The pairs of joined columns. These pairs are normalized, such that two predicates which join the same columns
            provide the join partners in the same order.

        Raises
        ------
        NoJoinPredicateError
        """
        raise NotImplementedError

    @abc.abstractmethod
    def base_predicates(self) -> Iterable[AbstractPredicate]:
        """Provides all base predicates that form this predicate.

        This allows to iterate over all leaves of a compound predicate, for base predicates it simply returns the predicate
        itself.

        Returns
        -------
        Iterable[AbstractPredicate]
            The base predicates, in an arbitrary order. If the predicate is a base predicate already, it will be the only item
            in the iterable.
        """
        raise NotImplementedError

    def required_tables(self) -> set[TableReference]:
        """Provides all tables that have to be "available" in order for this predicate to be executed.

        Availability in this context means that the table has to be scanned already. Therefore it can be accessed either as-is,
        or as part of an intermediate relation.

        The output of this method differs from the `tables` method in one central aspect: `tables` provides all tables that are
        accessed, which includes all tables from subqueries. In contrast, the `required_tables` remove all tables that are
        scanned by the subquery and only include those that must be "provided" by the query execution engine.

        Consider the following example predicate: ``R.a = (SELECT MIN(S.b) FROM S)``. Calling `tables` on this predicate would
        return the set ``{R, S}``. However, table *S* is already provided by the subquery. Therefore, `required_tables` only
        returns ``{R}``, since this is the only table that has to be provided by the context of this method.

        Returns
        -------
        set[TableReference]
            The tables that need to be provided by the query execution engine in order to run this predicate
        """
        subqueries = util.flatten(
            _collect_subquery_expressions(child_expr)
            for child_expr in self.iterexpressions()
        )
        subquery_tables = util.set_union(
            subquery.query.unbound_tables() for subquery in subqueries
        )
        column_tables = util.set_union(
            _collect_column_expression_tables(child_expr)
            for child_expr in self.iterexpressions()
        )
        return column_tables | subquery_tables

    @abc.abstractmethod
    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        """Enables processing of the current predicate by a predicate visitor.

        Parameters
        ----------
        visitor : PredicateVisitor[VisitorResult]
            The visitor
        args
            Additional arguments to pass to the visitor
        kwargs
            Additional keyword arguments to pass to the visitor
        """
        raise NotImplementedError

    def _assert_join_predicate(self) -> None:
        """Raises a `NoJoinPredicateError` if this predicate is not a join.

        Raises
        ------
        NoJoinPredicateError
        """
        if not self.is_join():
            raise NoJoinPredicateError(self)

    def _assert_filter_predicate(self) -> None:
        """Raises a `NoFilterPredicateError` if this predicate is not a filter.

        Raises
        ------
        NoFilterPredicateError
        """
        if not self.is_filter():
            raise NoFilterPredicateError(self)

    def __json__(self) -> jsondict:
        return {
            "node_type": "predicate",
            "tables": self.tables(),
            "predicate": str(self),
        }

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class BasePredicate(AbstractPredicate, abc.ABC):
    """A base predicate is a predicate that is not composed of any additional child predicates, such as a binary predicate.

    It represents the smallest kind of condition that evaluates to *TRUE* or *FALSE*.

    Parameters
    ----------
    operation : Optional[SqlOperator]
        The operation that compares the column value(s). For unary base predicates, this may be *None* if a
        predicate function is used to determine matching tuples.
    hash_val : int
        The hash of the entire predicate
    """

    def __init__(self, operation: Optional[SqlOperator], *, hash_val: int) -> None:
        self._operation = operation
        super().__init__(hash_val)

    __slots__ = ("_operation",)
    __match_args__ = ("operation",)

    @property
    def operation(self) -> Optional[SqlOperator]:
        """Get the operation that is used to obtain matching (pairs of) tuples.

        Most of the time, this operation will be set to one of the SQL operators. However, for unary predicates that filter
        based on a predicate function this might be *None* (e.g. a user-defined function such as in
        `SELECT * FROM R WHERE my_udf_predicate(R.a, R.b)`).

        Returns
        -------
        Optional[SqlOperator]
            The operation if it exists
        """
        return self._operation

    def is_compound(self) -> bool:
        return False

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]

    __hash__ = AbstractPredicate.__hash__


class BinaryPredicate(BasePredicate):
    """A binary predicate combines exactly two base expressions with a comparison operation.

    This is the most typical kind of predicate and appears in most joins (e.g. ``R.a = S.b``) and many filters
    (e.g. ``R.a = 42``).

    Parameters
    ----------
    operation : SqlOperator
        The operation that combines the input arguments
    first_argument : SqlExpression
        The first comparison value
    second_argument : SqlExpression
        The second comparison value
    """

    @staticmethod
    def equal(
        first_argument: SqlExpression, second_argument: SqlExpression
    ) -> BinaryPredicate:
        """Generates an equality predicate between two arguments."""
        return BinaryPredicate(LogicalOperator.Equal, first_argument, second_argument)

    def __init__(
        self,
        operation: SqlOperator,
        first_argument: SqlExpression,
        second_argument: SqlExpression,
    ) -> None:
        if not first_argument or not second_argument:
            raise ValueError("First argument and second argument are required")
        self._first_argument = first_argument
        self._second_argument = second_argument

        hash_val = hash((operation, first_argument, second_argument))
        super().__init__(operation, hash_val=hash_val)

    __slots__ = ("_first_argument", "_second_argument")
    __match_args__ = ("operation", "first_argument", "second_argument")

    @property
    def operation(self) -> SqlOperator:
        return self._operation  # type: ignore

    @property
    def first_argument(self) -> SqlExpression:
        """Get the first argument of the predicate.

        Returns
        -------
        SqlExpression
            The argument
        """
        return self._first_argument

    @property
    def second_argument(self) -> SqlExpression:
        """Get the second argument of the predicate.

        Returns
        -------
        SqlExpression
            The argument
        """
        return self._second_argument

    def is_join(self) -> bool:
        first_tables = _collect_column_expression_tables(self.first_argument)
        if len(first_tables) > 1:
            return True

        second_tables = _collect_column_expression_tables(self.second_argument)
        if len(second_tables) > 1:
            return True

        return (
            bool(first_tables)
            and bool(second_tables)
            and len(first_tables ^ second_tables) > 0
        )

    def columns(self) -> set[ColumnReference]:
        return self.first_argument.columns() | self.second_argument.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return list(self.first_argument.itercolumns()) + list(
            self.second_argument.itercolumns()
        )

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.first_argument, self.second_argument]

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        self._assert_join_predicate()
        first_columns = _collect_column_expression_columns(self.first_argument)
        second_columns = _collect_column_expression_columns(self.second_argument)

        partners = _generate_join_pairs(first_columns, first_columns)
        partners |= _generate_join_pairs(second_columns, second_columns)
        partners |= _generate_join_pairs(first_columns, second_columns)
        return partners

    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        if _safe_recurse_expression_visitor(visitor):
            return visitor.visit_predicate_expr(self, *args, **kwargs)
        return visitor.visit_binary_predicate(self, *args, **kwargs)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.operation == other.operation
            and self.first_argument == other.first_argument
            and self.second_argument == other.second_argument
        )

    def __str__(self) -> str:
        return f"{self.first_argument} {self.operation.value} {self.second_argument}"


class BetweenPredicate(BasePredicate):
    """A *BETWEEN* predicate is a special case of a conjunction of two binary predicates.

    Each *BETWEEN* predicate has a structure of ``<col> BETWEEN <a> AND <b>``, where ``<col>`` describes the (column)
    expression to which the condition should apply and ``<a>`` and ``<b>`` are the expressions that denote the valid bounds.

    Each BETWEEN predicate can be represented by a conjunction of binary predicates: ``<col> BETWEEN <a> AND <b>`` is
    equivalent to ``<col> >= <a> AND <col> <= <b>``.

    Parameters
    ----------
    column : SqlExpression
        The value that is checked by the predicate
    interval : tuple[SqlExpression, SqlExpression]
        The allowed range in which the `column` values must lie. The range is inclusive at both endpoints. This has to be a
        pair (2-tuple) of expressions.

    Raises
    ------
    ValueError
        If the `interval` is not a pair of values.

    Notes
    -----
    A *BETWEEN* predicate can be a join predicate as in ``R.a BETWEEN 42 AND S.b``.
    Furthermore, some systems even allow the ``<col>`` part to be an arbitrary expression. For example, in Postgres this is a
    valid query:

    .. code-block:: sql

        SELECT *
        FROM R JOIN S ON R.a = S.b
        WHERE 42 BETWEEN R.c AND S.d
    """

    def __init__(
        self,
        column: SqlExpression,
        interval: tuple[SqlExpression, SqlExpression],
    ) -> None:
        if not column or not interval or len(interval) != 2:
            raise ValueError("Column and interval must be set")
        self._column = column
        self._interval = interval
        self._interval_start, self._interval_end = self._interval

        hash_val = hash(
            (
                LogicalOperator.Between,
                self._column,
                self._interval_start,
                self._interval_end,
            )
        )
        super().__init__(LogicalOperator.Between, hash_val=hash_val)

    __slots__ = ("_column", "_interval", "_interval_start", "_interval_end")
    __match_args__ = ("column", "interval_start", "interval_end")

    @property
    def column(self) -> SqlExpression:
        """Get the column that is tested (*R.a* in ``SELECT * FROM R WHERE R.a BETWEEN 1 AND 42``).

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    @property
    def interval(self) -> tuple[SqlExpression, SqlExpression]:
        """Get the interval (as ``(lower, upper)``) that is tested against.

        Returns
        -------
        tuple[SqlExpression, SqlExpression]
            The allowed range of values. This interval is inclusive at both endpoints.
        """
        return self._interval

    @property
    def interval_start(self) -> SqlExpression:
        """Get the lower bound of the interval that is tested against.

        Returns
        -------
        SqlExpression
            The lower value. This value is inclusive, i.e. the comparison values must be greater or equal.
        """
        return self._interval_start

    @property
    def interval_end(self) -> SqlExpression:
        """Get the upper bound of the interval that is tested against.

        Returns
        -------
        SqlExpression
            The upper value. This value is inclusive, i.e. the comparison values must be less or equal.
        """
        return self._interval_end

    def is_join(self) -> bool:
        column_tables = _collect_column_expression_tables(self.column)
        interval_start_tables = _collect_column_expression_tables(self.interval_start)
        interval_end_tables = _collect_column_expression_tables(self.interval_end)
        return (
            len(column_tables) > 1
            or len(column_tables | interval_start_tables) > 1
            or len(column_tables | interval_end_tables) > 1
        )

    def columns(self) -> set[ColumnReference]:
        return (
            self.column.columns()
            | self.interval_start.columns()
            | self.interval_end.columns()
        )

    def itercolumns(self) -> Iterable[ColumnReference]:
        return (
            list(self.column.itercolumns())
            + list(self.interval_start.itercolumns())
            + list(self.interval_end.itercolumns())
        )

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.column, self.interval_start, self.interval_end]

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        self._assert_join_predicate()
        predicate_columns = _collect_column_expression_columns(self.column)
        start_columns = _collect_column_expression_columns(self.interval_start)
        end_columns = _collect_column_expression_columns(self.interval_end)

        partners = _generate_join_pairs(predicate_columns, predicate_columns)
        partners |= _generate_join_pairs(predicate_columns, start_columns)
        partners |= _generate_join_pairs(predicate_columns, end_columns)
        return set(partners)

    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        if _safe_recurse_expression_visitor(visitor):
            return visitor.visit_predicate_expr(self, *args, **kwargs)
        return visitor.visit_between_predicate(self, *args, **kwargs)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.column == other.column
            and self.interval == other.interval
        )

    def __str__(self) -> str:
        interval_start, interval_end = self.interval
        return f"{self.column} BETWEEN {interval_start} AND {interval_end}"


class InPredicate(BasePredicate):
    """An *IN* predicate lists the allowed values for a column.

    In most cases, such a predicate with *n* allowed values can be transformed into a disjunction of *n* equality
    predicates, i.e. ``R.a IN (1, 2, 3)`` is equivalent to ``R.a = 1 OR R.a = 2 OR R.a = 3``. Depending on the allowed
    values, an IN predicate can denote a join, e.g. ``R.a IN (S.b, S.c)``. An important special case arises if the
    allowed values are produced by a subquery, e.g. ``R.a IN (SELECT S.b FROM S)``. This does no longer allow for a
    transformation into binary predicates since it is unclear how many rows the subquery will produce.

    Parameters
    ----------
    column : SqlExpression
        The value that is checked by the predicate
    values : Sequence[SqlExpression]
        The allowed column values. The individual expressions are not limited to `StaticValueExpression` instances, but can
        also include subqueries, columns or complicated mathematical expressions.

    Raises
    ------
    ValueError
        If `values` is empty.

    Notes
    -----
    Some systems even allow the `column` part to be an arbitrary expression. For example, in Postgres this is a valid query:

    .. code-block:: sql

        SELECT *
        FROM R JOIN S ON R.a = S.b
        WHERE 42 IN (R.c, 40 * S.d)
    """

    @staticmethod
    def subquery(
        column: SqlExpression, subquery: SubqueryExpression | SqlQuery
    ) -> InPredicate:
        """Generates an *IN* predicate that is based on a subquery.

        Such a predicate is of the form ``R.a IN (SELECT S.b FROM S)``.

        Parameters
        ----------
        column : SqlExpression
            The column that should be checked for being contained by the subquerie's result set.
        subquery : SubqueryExpression | SqlQuery
            The subquery to produce the allowed values.

        Returns
        -------
        InPredicate
            The predicate
        """
        subquery = (
            subquery
            if isinstance(subquery, SubqueryExpression)
            else SubqueryExpression(subquery)
        )
        return InPredicate(column, (subquery,))

    def __init__(self, column: SqlExpression, values: Sequence[SqlExpression]) -> None:
        if not column or not values:
            raise ValueError("Both column and values must be given")
        if not all(val for val in values):
            raise ValueError("No empty value allowed")
        self._column = column
        self._values = tuple(values)
        hash_val = hash((LogicalOperator.In, self._column, self._values))
        super().__init__(LogicalOperator.In, hash_val=hash_val)

    __slots__ = ("_column", "_values")
    __match_args__ = ("column", "values")

    @property
    def column(self) -> SqlExpression:
        """Get the expression that is tested (*R.a* in ``SELECT * FROM R WHERE R.a IN (1, 2, 3)``).

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    @property
    def values(self) -> Sequence[SqlExpression]:
        """Get the allowed values of the tested expression.

        Returns
        -------
        Sequence[SqlExpression]
            The allowed values. This sequence always contains at least one entry.
        """
        return self._values

    def is_subquery_predicate(self) -> bool:
        """Checks, if this is a subquery-based **IN** predicate, i.e. a predicate of the form ``R.a IN (SELECT S.b FROM S)``.

        Returns
        -------
        bool
            Whether this predicate is based on a subquery
        """
        return len(self._values) == 1 and isinstance(
            self._values[0], SubqueryExpression
        )

    def is_join(self) -> bool:
        column_tables = _collect_column_expression_tables(self.column)
        if len(column_tables) > 1:
            return True
        for value in self.values:
            value_tables = _collect_column_expression_tables(value)
            if len(column_tables | value_tables) > 1:
                return True
        return False

    def columns(self) -> set[ColumnReference]:
        all_columns = self.column.columns()
        for val in self.values:
            all_columns |= val.columns()
        return all_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        return list(self.column.itercolumns()) + util.flatten(
            val.itercolumns() for val in self.values
        )

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.column] + list(self.values)

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        self._assert_join_predicate()
        predicate_columns = _collect_column_expression_columns(self.column)

        partners = _generate_join_pairs(predicate_columns, predicate_columns)
        for value in self.values:
            value_columns = _collect_column_expression_columns(value)
            partners |= _generate_join_pairs(predicate_columns, value_columns)
        return partners

    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        if _safe_recurse_expression_visitor(visitor):
            return visitor.visit_predicate_expr(self, *args, **kwargs)
        return visitor.visit_in_predicate(self, *args, **kwargs)

    def _stringify_values(self) -> str:
        """Converts the allowed values into a valid string representation."""
        # NOTE: part of this implementation is re-used in the __str__ method for NOT predicates to format NOT IN predicates
        # appropriately. These methods should be kept in sync.
        if len(self.values) == 1:
            value = util.simplify(self.values)
            vals = str(value) if isinstance(value, SubqueryExpression) else f"({value})"
        else:
            vals = "(" + ", ".join(str(val) for val in self.values) + ")"
        return vals

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.column == other.column
            and set(self.values) == set(other.values)
        )

    def __str__(self) -> str:
        vals = self._stringify_values()
        return f"{self.column} IN {vals}"


class UnaryPredicate(BasePredicate):
    """A unary predicate is applied directly to an expression, evaluating to TRUE or FALSE.

    Examples of such predicates include ``R.a IS NOT NULL``, ``EXISTS (SELECT S.b FROM S WHERE R.a = S.b)``, or
    ``my_udf(R.a)``. In the last case, ``my_udf`` has to produce a boolean return value.

    Parameters
    ----------
    column : SqlExpression
        The expression that is tested. This can also be a user-defined function that produces a boolen return value.
    operation : Optional[SqlOperator], optional
        The operation that is used to generate the unary predicate. Only a small subset of operators can actually be used in
        this context (e.g. *EXISTS* or *MISSING*). If the predicate does not require an operator (e.g. in the case of
        filtering UDFs), the operation can be *None*. Notice however, that PostBOUND has no knowledge of the semantics of
        UDFs and can therefore not enforce, whether UDFs is actually valid in this context. This has to be done at runtime by
        the actual database system.

    Raises
    ------
    ValueError
        If the given operation is not a valid unary operator
    """

    @staticmethod
    def exists(subquery: SqlQuery | SubqueryExpression) -> UnaryPredicate:
        """Creates an *EXISTS* predicate for a subquery.

        Parameters
        ----------
        subquery : SqlQuery | SubqueryExpression
            The subquery. Will be wrapped in a `SubqueryExpression` if it is not already one.

        Returns
        -------
        UnaryPredicate
            The *EXISTS* predicate
        """
        subquery = (
            subquery
            if isinstance(subquery, SubqueryExpression)
            else SubqueryExpression(subquery)
        )
        return UnaryPredicate(subquery, LogicalOperator.Exists)

    def __init__(self, column: SqlExpression, operation: Optional[SqlOperator] = None):
        if not column:
            raise ValueError("Column must be set")
        if operation is not None and operation not in UnarySqlOperators:
            raise ValueError(f"Not an allowed unary operator: {operation}")
        self._column = column
        super().__init__(operation, hash_val=hash((operation, column)))

    __slots__ = ("_column",)
    __match_args__ = ("column", "operation")

    @property
    def column(self) -> SqlExpression:
        """The column that is checked by this predicate

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    def is_exists(self) -> bool:
        """Checks, whether this predicate is an *EXISTS* predicate.

        Returns
        -------
        bool
            Whether this predicate is an *EXISTS* predicate
        """
        return self.operation == LogicalOperator.Exists

    def is_join(self) -> bool:
        return len(_collect_column_expression_tables(self.column)) > 1

    def columns(self) -> set[ColumnReference]:
        return self.column.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.column.itercolumns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.column]

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        columns = _collect_column_expression_columns(self.column)
        return _generate_join_pairs(columns, columns)

    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        if _safe_recurse_expression_visitor(visitor):
            return visitor.visit_predicate_expr(self, *args, **kwargs)
        return visitor.visit_unary_predicate(self, *args, **kwargs)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.operation == other.operation
            and self.column == other.column
        )

    def __str__(self) -> str:
        if not self.operation:
            return str(self.column)

        if self.operation == LogicalOperator.Exists:
            assert isinstance(self.column, SubqueryExpression)
            return f"EXISTS {self.column}"

        col_str = (
            str(self.column)
            if isinstance(self.column, (StaticValueExpression, ColumnExpression))
            else f"({self.column})"
        )
        return f"{self.operation.value}{col_str}"


class CompoundPredicate(AbstractPredicate):
    """A compound predicate creates a composite hierarchical structure of other predicates.

    Currently, PostBOUND supports 3 kinds of compound predicates: negations, conjunctions and disjunctions. Depending on the
    specific compound operator, a diferent number of child predicates is allowed.

    Parameters
    ----------
    operation : LogicalSqlCompoundOperators
        The operation that glues together the individual child predicates.
    children : AbstractPredicate | Sequence[AbstractPredicate]
        The predicates that are combined by this composite. For conjunctions and disjunctions, at least two children are
        required. For negations, exactly one child is permitted (either directly or in the sequence).

    Raises
    ------
    ValueError
        If `operation` is a negation and a number of children unequal to 1 is passed
    ValueError
        If `operation` is a conjunction or a disjunction and less than 2 children are passed

    .. deprecated:: 0.20.2
        `CompoundPredicate` will only handle AND/OR predicates in the future. NOT predicates will be represented by a proper
        `NotPredicate` class.
    """

    @staticmethod
    def create(
        operation: CompoundOperator, parts: Sequence[AbstractPredicate]
    ) -> AbstractPredicate:
        """Creates an arbitrary compound predicate for a number of child predicates.

        If just a single child predicate is provided, but the operation requires multiple children, that child is returned
        directly instead of the compound predicate.

        Parameters
        ----------
        operation : LogicalSqlCompoundOperators
            The logical operator to combine the child predicates.
        parts : Sequence[AbstractPredicate]
            The child predicates

        Returns
        -------
        AbstractPredicate
            A composite predicate of the given `parts`, if parts contains the appropriate number of items. Otherwise the
            supplied child predicate.

        Raises
        ------
        ValueError
            If a negation predicate should be created but a number child predicates unequal to one are supplied. Likewise, if
            a conjunction or disjunction is requested, but no child predicates are supplied.
        """
        if operation == CompoundOperator.Not and len(parts) != 1:
            raise ValueError(
                f"Can only create negations for exactly one predicate but received: '{parts}'"
            )
        elif operation != CompoundOperator.Not and not parts:
            raise ValueError("Conjunctions/disjunctions require at least one predicate")

        match operation:
            case CompoundOperator.Not:
                return CompoundPredicate.create_not(parts[0])
            case CompoundOperator.And | CompoundOperator.Or:
                if len(parts) == 1:
                    return parts[0]
                return CompoundPredicate(operation, parts)
            case _:
                raise ValueError(f"Unknown operator: '{operation}'")

    @staticmethod
    def create_and(parts: Collection[AbstractPredicate]) -> AbstractPredicate:
        """Creates an *AND* predicate, combining a number of child predicates.

        If just a single child predicate is provided, that child is returned directly instead of wrapping it in an
        *AND* predicate.

        Parameters
        ----------
        parts : Collection[AbstractPredicate]
            The children that should be combined

        Returns
        -------
        AbstractPredicate
            A conjunctive predicate of the given `parts`, if `parts` contains at least two items. Otherwise the only passed
            predicate is returned.

        Raises
        ------
        ValueError
            If `parts` is empty
        """
        parts = list(parts)
        if not parts:
            raise ValueError("No predicates supplied.")
        if len(parts) == 1:
            return parts[0]
        return CompoundPredicate(CompoundOperator.And, parts)

    @staticmethod
    def create_not(predicate: AbstractPredicate) -> CompoundPredicate:
        """Builds a *NOT* predicate, wrapping a specific child predicate.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate that should be negated

        Returns
        -------
        CompoundPredicate
            The negated predicate. No logic checks or simplifications are performed. For example, it is possible to negate a
            negation and this will still be represented in the predicate hierarchy.
        """
        return CompoundPredicate(CompoundOperator.Not, predicate)

    @staticmethod
    def create_or(parts: Collection[AbstractPredicate]) -> AbstractPredicate:
        """Creates an *OR* predicate, combining a number of child predicates.

        If just a single child predicate is provided, that child is returned directly instead of wrapping it in an
        *OR* predicate.

        Parameters
        ----------
        parts : Collection[AbstractPredicate]
            The children that should be combined

        Returns
        -------
        AbstractPredicate
            A disjunctive predicate of the given `parts`, if `parts` contains at least two items. Otherwise the only passed
            predicate is returned.

        Raises
        ------
        ValueError
            If `parts` is empty
        """
        parts = list(parts)
        if not parts:
            raise ValueError("No predicates supplied.")
        if len(parts) == 1:
            return parts[0]
        return CompoundPredicate(CompoundOperator.Or, parts)

    def __init__(
        self,
        operation: CompoundOperator,
        children: AbstractPredicate | Sequence[AbstractPredicate],
    ) -> None:
        if not operation or not children:
            raise ValueError("Operation and children must be set")
        if operation == CompoundOperator.Not and len(util.enlist(children)) > 1:
            raise ValueError("NOT predicates can only have one child predicate")
        if operation != CompoundOperator.Not and len(util.enlist(children)) < 2:
            raise ValueError("AND/OR predicates require at least two child predicates.")
        self._operation = operation
        self._children = tuple(util.enlist(children))
        super().__init__(hash((self._operation, self._children)))

    __slots__ = ("_operation", "_children")
    __match_args__ = ("operation", "children")

    @property
    def operation(self) -> CompoundOperator:
        """Get the operation used to combine the individual evaluations of the child predicates.

        Returns
        -------
        LogicalSqlCompoundOperators
            The operation
        """
        return self._operation

    @property
    def children(self) -> Sequence[AbstractPredicate] | AbstractPredicate:
        """Get the child predicates that are combined in this compound predicate.

        For conjunctions and disjunctions this will be a sequence of children with at least two children. For negations
        the child predicate will be returned directly (i.e. without being wrapped in a sequence).

        Returns
        -------
        Sequence[AbstractPredicate] | AbstractPredicate
            The sequence of child predicates for *AND* and *OR* predicates, or the negated predicate for *NOT*
            predicates.
        """
        return (
            self._children[0]
            if self.operation == CompoundOperator.Not
            else self._children
        )

    def is_negation(self) -> bool:
        """Checks whether this is a NOT predicate."""
        return self.operation == CompoundOperator.Not

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self._children)

    def columns(self) -> set[ColumnReference]:
        return util.set_union(child.columns() for child in self._children)

    def iterchildren(self) -> Sequence[AbstractPredicate]:
        """Provides all children contained in this predicate.

        In contrast to the `children` property, this method always returns an iterable, even for *NOT* predicates. In the
        latter case the iterable contains just a single item.

        Returns
        -------
        Sequence[AbstractPredicate]
            The children
        """
        return self._children

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(child.itercolumns() for child in self._children)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(child.iterexpressions() for child in self._children)

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        return util.set_union(child.join_partners() for child in self._children)

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return util.set_union(set(child.base_predicates()) for child in self._children)

    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        if _safe_recurse_expression_visitor(visitor):
            return visitor.visit_predicate_expr(self, *args, **kwargs)

        match self.operation:
            case CompoundOperator.Not:
                return visitor.visit_not_predicate(self, self.children, *args, **kwargs)
            case CompoundOperator.And:
                return visitor.visit_and_predicate(self, self.children, *args, **kwargs)
            case CompoundOperator.Or:
                return visitor.visit_or_predicate(self, self.children, *args, **kwargs)
            case _:
                raise ValueError(f"Unknown operation: '{self.operation}'")

    def _stringify_not(self) -> str:
        if not isinstance(self.children, InPredicate):
            return f"NOT {self.children}"
        return f"{self.children.column} NOT IN {self.children._stringify_values()}"

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.operation == other.operation
            and self.children == other.children
        )

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        if self.operation == CompoundOperator.Not:
            return self._stringify_not()
        elif self.operation == CompoundOperator.Or:
            components = " OR ".join(
                f"({child})" if child.is_compound() else str(child)
                for child in self.iterchildren()
            )
            return f"({components})"
        elif self.operation == CompoundOperator.And:
            return " AND ".join(str(child) for child in self.iterchildren())
        else:
            raise ValueError(f"Unknown operation: '{self.operation}'")


class PredicateVisitor(abc.ABC, Generic[VisitorResult]):
    """Basic visitor to operator on arbitrary predicate trees.

    As a modification to a strict vanilla interpretation of the design pattern, we provide dedicated matching methods for the
    different composite operators (i.e. for *AND*, *OR* and *NOT* predicates), rather than just matching on
    `CompoundPredicate`.

    See Also
    --------
    AbstractPredicate

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    def visit_query_predicates(
        self, query: SqlQuery | QueryPredicates, *args, **kwargs
    ) -> VisitorResult:
        if isinstance(query, SqlQuery):
            predicates = query.predicates()
        else:
            predicates = query
        return predicates.root.accept_visitor(self, *args, **kwargs)

    @abc.abstractmethod
    def visit_binary_predicate(
        self, predicate: BinaryPredicate, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_between_predicate(
        self, predicate: BetweenPredicate, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_in_predicate(
        self, predicate: InPredicate, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_unary_predicate(
        self, predicate: UnaryPredicate, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_not_predicate(
        self,
        predicate: CompoundPredicate,
        child_predicate: AbstractPredicate,
        *args,
        **kwargs,
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_or_predicate(
        self,
        predicate: CompoundPredicate,
        components: Sequence[AbstractPredicate],
        *args,
        **kwargs,
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_and_predicate(
        self,
        predicate: CompoundPredicate,
        components: Sequence[AbstractPredicate],
        *args,
        **kwargs,
    ) -> VisitorResult:
        raise NotImplementedError


def _safe_recurse_expression_visitor(visitor) -> bool:
    return (  #
        isinstance(visitor, SqlExpressionVisitor)  #
        and not isinstance(visitor, PredicateVisitor)  #
    )


@overload
def as_predicate(
    column: ColumnReference,
    operation: Literal[LogicalOperator.In, "in", "IN"],
    *arguments,
) -> InPredicate: ...


@overload
def as_predicate(
    column: ColumnReference,
    operation: Literal[
        LogicalOperator.Between,
        "between",
        "BETWEEN",
    ],
    *arguments,
) -> BetweenPredicate: ...


@overload
def as_predicate(
    column: ColumnReference, operation: LogicalOperator | str, *arguments
) -> BinaryPredicate: ...


def as_predicate(
    column: ColumnReference, operation: LogicalOperator | str, *arguments
) -> BasePredicate:
    """Utility method to quickly construct instances of base predicates.

    The given arguments are transformed into appropriate expression objects as necessary.

    The specific type of generated predicate is determined by the given operation. The following rules are applied:

    - for *BETWEEN* predicates, the arguments can be either two values, or a tuple of values
      (additional arguments are ignored)
    - for *IN* predicates, the arguments can be either a number of arguments, or a (nested) iterable of arguments
    - for all other binary predicates exactly one additional argument must be given (and an error is raised if that
      is not the case)

    Parameters
    ----------
    column : ColumnReference
        The column that should become the first operand of the predicate
    operation : LogicalSqlOperators | str
        The operation that should be used to build the predicate. The actual return type depends on this value.
        As an alternative to `LogicalOperator`, the operation can also be provided as a string (e.g. `"="` for `LogicalOperator.Equal`).
    *arguments
        Further operands for the predicate. The allowed values and their structure depend on the precise predicate (see rules
        above).

    Returns
    -------
    BasePredicate
        A predicate representing the given operation on the given operands

    Raises
    ------
    ValueError
        If a binary predicate is requested, but `*arguments` does not contain a single value
    """
    if isinstance(operation, str):
        operation = operation.upper()
        aliases = {"!=": "<>", "==": "="}
        operation = aliases.get(operation, operation)
        operation = LogicalOperator(operation)

    column: ColumnExpression = ColumnExpression(column)

    if operation == LogicalOperator.Between:
        if len(arguments) == 1:
            lower, upper = arguments[0]
        else:
            lower, upper, *__ = arguments
        return BetweenPredicate(
            column,
            (
                as_expression(lower, allow_star=False),
                as_expression(upper, allow_star=False),
            ),
        )
    elif operation == LogicalOperator.In:
        arguments = util.flatten(arguments)
        return InPredicate(
            column,
            [as_expression(value, allow_star=False) for value in arguments],
        )
    elif len(arguments) != 1:
        raise ValueError("Too many arguments for binary predicate: " + str(arguments))

    argument = arguments[0]
    return BinaryPredicate(operation, column, as_expression(argument, allow_star=False))


def determine_join_equivalence_classes(
    predicates: Iterable[BinaryPredicate],
) -> set[frozenset[ColumnReference]]:
    """Calculates all equivalence classes of equijoin predicates.

    Columns are in an equivalence class if they can all be compared with matching equality predicates. For example, consider
    two predicates *a = b* and *a = c*. From these predicates it follows that *b = c* and hence the set of columns *{a, b, c}*
    is an equivalence class. Likewise, the predicates *a = b* and *c = d* form two equivalence classes, namely *{a, b}* and
    *{c, d}*.

    Parameters
    ----------
    predicates : Iterable[BinaryPredicate]
        The predicates to check. Non-equijoin predicates are discarded automatically.

    Returns
    -------
    set[frozenset[ColumnReference]]
        The equivalence classes. Each element of the set describes a complete equivalence class.
    """
    join_predicates = {
        pred
        for pred in predicates
        if isinstance(pred, BinaryPredicate)
        and pred.is_join()
        and pred.operation == LogicalOperator.Equal
    }

    equivalence_graph = nx.Graph()
    for predicate in join_predicates:
        columns = predicate.columns()
        if not len(columns) == 2:
            continue
        col_a, col_b = columns
        equivalence_graph.add_edge(col_a, col_b)

    equivalence_classes: set[frozenset[ColumnReference]] = set()
    for equivalence_class in nx.connected_components(equivalence_graph):
        equivalence_classes.add(frozenset(equivalence_class))
    return equivalence_classes


def generate_predicates_for_equivalence_classes(
    equivalence_classes: set[frozenset[ColumnReference]],
) -> set[BinaryPredicate]:
    """Provides all possible equijoin predicates for a set of equivalence classes.

    This function can be used in combination with `determine_join_equivalence_classes` to expand join predicates to also
    include additional joins that can be derived from the predicates.

    For example, consider two joins *a = b* and *b = c*. These joins form one equivalence class *{a, b, c}*. Based on the
    equivalence class, the predicates *a = b*, *b = c* and *a = c* can be generated.

    Parameters
    ----------
    equivalence_classes : set[frozenset[ColumnReference]]
        The equivalence classes. Each class is described by the columns it contains.

    Returns
    -------
    set[BinaryPredicate]
        The predicates

    See Also
    --------
    determine_join_equivalence_classes
    CompoundPredicate.create_and
    """
    equivalence_predicates: set[BinaryPredicate] = set()
    for equivalence_class in equivalence_classes:
        for first_col, second_col in util.collections.pairs(equivalence_class):
            equivalence_predicates.add(
                as_predicate(first_col, LogicalOperator.Equal, second_col)
            )
    return equivalence_predicates


def _unwrap_expression(expression: SqlExpression) -> ColumnReference | object:
    """Provides the column of a `ColumnExpression` or the value of a `StaticValueExpression`.

    This is a utility method to gain quick access to the values in simple predicates.

    Parameters
    ----------
    expression : SqlExpression
        The expression to unwrap.

    Returns
    -------
    ColumnReference | object
        The column or value contained in the expression.
    """
    match expression:
        case StaticValueExpression(val):
            return val
        case ColumnExpression(col):
            return col
        case CastExpression(castee):
            return _unwrap_expression(castee)
        case _:
            raise ValueError("Cannot unwrap expression " + str(expression))


UnwrappedFilter = tuple[ColumnReference, LogicalOperator, object]
"""Type that captures the main components of a filter predicate."""


def _attempt_filter_unwrap(
    predicate: AbstractPredicate,
) -> tuple[ColumnReference, LogicalOperator, Any] | None:
    """Extracts the main components of a simple filter, making them more directly accessible.

    This is a preparatory step in order to create instances of `SimpleFilter`. Therefore, it only
    works for predicates that match the requirements of "simple" filters. This boils down to being
    of the form ``<column> <operator> <values>``. If this condition is not met, the unwrapping fails.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate that should be simplified

    Returns
    -------
    tuple[ColumnReference, LogicalOperator, Any] | None
        A triple consisting of column, operator and value(s) if the `predicate` could be unwrapped,
        or *None* otherwise.

    """
    if not predicate.is_filter() or not predicate.is_base():
        return None

    match predicate:
        case BinaryPredicate(op, lhs, rhs):
            left, right = _unwrap_expression(lhs), _unwrap_expression(rhs)
            left, right = (
                (left, right) if isinstance(left, ColumnReference) else (right, left)
            )
            assert isinstance(left, ColumnReference)
            return left, op, right

        case BetweenPredicate(lhs, lower, upper):
            lhs = _unwrap_expression(lhs)
            lower, upper = _unwrap_expression(lower), _unwrap_expression(upper)
            assert isinstance(lhs, ColumnReference)
            return lhs, LogicalOperator.Between, (lower, upper)

        case InPredicate(lhs, values):
            lhs = _unwrap_expression(lhs)
            values = [_unwrap_expression(val) for val in values]
            assert isinstance(lhs, ColumnReference)
            return lhs, LogicalOperator.In, tuple(values)

        case _:
            return None


def _unwrap_filter_or_raise(
    predicate: AbstractPredicate,
) -> tuple[ColumnReference, LogicalOperator, Any]:
    """Tries to unwrap a filter predicate or raises an error if this is not possible.

    See Also
    --------
    _attempt_filter_unwrap
    """
    unwrapped = _attempt_filter_unwrap(predicate)
    if unwrapped is None:
        raise ValueError(f"Could not simplify predicate {predicate}")
    return unwrapped


class SimpleFilter(AbstractPredicate):
    """A simple filter provides more streamlined and direct access to filter predicates.

    A simple filter is a read-only predicate, i.e. it cannot be created on its own and has to be
    derived from a base predicate (either a binary predicate, a *BETWEEN* predicate, or an *IN*
    predicate). Afterward, it provides read-only access to the predicate being filtered, the filter
    operation, as well as the values used to restrict the allowed column instances.
    All these attributes are made directly available instead of via hierarchies of `SqlExpression`
    instances.

    Note that not all base predicates can be represented as a simplified view. In order for the view
    to work, both the column as well as the filter values cannot be modified by other expressions
    such as function calls or mathematical expressions. However, cast expressions are tolerated and
    will simply be dropped. As a rule of thumb, if an expression modifies a value (such as a function
    call), this cannot be unwrapped. Therefore, a filter approximately has to be of the form
    ``<column reference> <operator> <static values>`` in order for the representation to work.

    The static methods `attempt_wrap`, `wrap`, `can_wrap` and `wrap_all` serve as high-level access
    points into the view. The components of the view are accessible via properties.
    In general, using `attempt_wrap` and `wrap_all` should be sufficient for most purposes.

    Parameters
    ----------
    column : ColumnReference | AbstractPredicate
        The column being filtered.

        For legacy purposes, this can (counterintuitively) also be a full predicate. This is treated
        as follows: if a predicate is given, all other parameters need to be omitted (they are
        ignored either way). The predicate must be of simple form (see explanation above) and its
        components will be automatically unwrapped and used as attribute values for the simplified
        version. The introduction of the `attempt_wrap` method has made this mode unnecessary. It is
        only available to keep backwards compatibility.
    operation : LogicalOperator | None
        The filter operation (e.g. *IN* or ``<>``). This cannot be *EXISTS* or *MISSING*, since
        subqueries cannot be represented in simplified form. Passing *None* only works in legacy
        mode.
    value : Any
        The filter value. Passing *None* is interpreted as SQL NULL.
    predicate : AbstractPredicate | None
        The original predicate that is represented by this simplified view. Passing *None* only
        works in legacy mode.

    See Also
    --------
    SimpleJoin : A similar view dedicated to join predicates

    Examples
    --------
    The best way to construct simplified views is to start with the `QueryPredicates` and extract
    the filter predicates, e.g., by using ``views = SimpleFilter.wrap_all(query.predicates())`` or
    ``filters = SimpleFilter.wrap_all(query.predicates().joins())``. Notice that especially the
    first conversion can be "lossy": all join predicates are dropped. Likewise, all filters that are
    more complex such as disjunctions are ignored. Alternatively, the `QueryPredicates` also
    provides a `simplify()` method that can be used to convert all predicates (filters and joins)
    into their simplified counterparts.

    If a single predicate should be simplified, the ``attempt_wrap(predicate)`` usually works best.

    The following predicates can be represented as a simplified view: ``R.a = 42``,
    ``R.b BETWEEN 1 AND 2`` or ``R.c IN (11, 22, 33)``.
    On the other hand, the following predicates cannot be represented b/c they involve advanced
    operations: ``R.a + 10 = 42`` (contains a mathematical expression) and
    ``some_udf(R.a) < 11 % 2`` (contains a function call, a predicate expressoin, and a mathematical
    expression).

    Notes
    -----
    Simple filters can be used in *match* statements and provide *(column, operation, value)* as
    arguments.
    """

    @staticmethod
    def attempt_wrap(predicate: AbstractPredicate) -> Optional[SimpleFilter]:
        """Transforms a predicate into a simplified view. Returns *None* if that is not possible."""
        unwrapped = _attempt_filter_unwrap(predicate)
        if unwrapped is None:
            return None
        col, op, val = unwrapped
        return SimpleFilter(col, op, val, predicate=predicate)

    @staticmethod
    def wrap(predicate: AbstractPredicate) -> SimpleFilter:
        """Transforms a predicate into a simplified view. Raises an error if that is not possible.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to represent as a simplified view

        Returns
        -------
        SimpleFilter
            A simplified view wrapping the given predicate

        Raises
        ------
        ValueError
            If the predicate cannot be represented as a simplified view.
        """
        simplified = SimpleFilter.attempt_wrap(predicate)
        if simplified is None:
            raise ValueError(f"Could not simplify predicate {predicate}")
        return simplified

    @staticmethod
    def can_wrap(predicate: AbstractPredicate) -> bool:
        """Checks, whether a specific predicate can be represented as a simplified view.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to check

        Returns
        -------
        bool
            Whether a representation as a simplified view is possible.
        """
        unwrapped = _attempt_filter_unwrap(predicate)
        return unwrapped is not None

    @staticmethod
    def wrap_all(
        predicates: Iterable[AbstractPredicate],
    ) -> Sequence[SimpleFilter]:
        """Transforms specific predicates into simplified views.

        If individual predicates cannot be represented as views, they are ignored.

        Parameters
        ----------
        predicates : Iterable[AbstractPredicate]
            The predicates to represent as views. These can be arbitrary predicates, i.e. including
            joins and complex filters.

        Returns
        -------
        Sequence[SimpleFilter]
            The simplified views for all predicates that can be represented this way. The sequence
            of the views matches the sequence in the `predicates`. If the representation fails for
            individual predicates, they simply do not appear in the result. Therefore, this sequence
            may be empty if none of the predicates are valid  simplified views.
        """
        views: list[SimpleFilter | None] = [
            SimpleFilter.attempt_wrap(pred) for pred in predicates
        ]
        return [v for v in views if v is not None]

    def __init__(
        self,
        column: ColumnReference | AbstractPredicate,
        operation: LogicalOperator | None = None,
        value: Any = None,
        *,
        predicate: AbstractPredicate | None = None,
    ) -> None:
        if isinstance(column, AbstractPredicate):
            predicate = column
            column, operation, value = _unwrap_filter_or_raise(column)
        if predicate is None:
            raise ValueError("Original predicate must be given explicitly!")
        if column is None or operation is None:  # value can be None for NULL filters[]
            raise ValueError("Column, operation and value must be given explicitly!")

        self._column = column
        self._operation = operation
        self._value = value
        self._predicate = predicate

        hash_val = hash((column, operation, value))
        super().__init__(hash_val)

    __slots__ = ("_column", "_operation", "_value", "_predicate")
    __match_args__ = ("column", "operation", "value")

    @property
    def column(self) -> ColumnReference:
        """Get the filtered column.

        Returns
        -------
        ColumnReference
            The column
        """
        return self._column

    @property
    def operation(self) -> LogicalOperator:
        """Get the SQL operation that is used for the filter (e.g. *IN* or ``<>``).

        Returns
        -------
        LogicalSqlOperators
            the operator. This cannot be *EXISTS* or *MISSING*, since subqueries cannot be
            represented in simplified views.
        """
        return self._operation

    @property
    def value(self) -> object | tuple[object] | Sequence[object]:
        """Get the filter value.

        Returns
        -------
        object | tuple[object] | Sequence[object]
            The value. For a binary predicate, this is just the value itself. For a *BETWEEN*
            predicate, this is tuple of the form ``(lower, upper)`` and for an *IN* predicate, this
            is a sequence of the allowed values.
        """
        return self._value

    def unwrap(self) -> AbstractPredicate:
        """Get the original predicate that is represented by this view.

        Returns
        -------
        AbstractPredicate
            The original predicate
        """
        return self._predicate

    def is_compound(self) -> bool:
        return False

    def is_join(self) -> bool:
        return False

    def columns(self) -> set[ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[ColumnReference]:
        return [self.column]

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self._predicate.iterexpressions()

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        return set()

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]

    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        if _safe_recurse_expression_visitor(visitor):
            return visitor.visit_predicate_expr(self, *args, **kwargs)  # type: ignore
        return self._predicate.accept_visitor(visitor, *args, **kwargs)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SimpleFilter):
            return self._predicate == other._predicate

        return isinstance(other, type(self._predicate)) and self._predicate == other

    def __str__(self) -> str:
        return str(self._predicate)


def _unwrap_simple_join(
    predicate: AbstractPredicate,
) -> tuple[ColumnReference, ColumnReference] | None:
    if not isinstance(predicate, BinaryPredicate) or not predicate.is_join():
        return None
    if not predicate.operation == LogicalOperator.Equal:
        return None

    lhs, rhs = (
        _unwrap_expression(predicate.first_argument),
        _unwrap_expression(predicate.second_argument),
    )
    if not isinstance(lhs, ColumnReference) or not isinstance(rhs, ColumnReference):
        return None
    return lhs, rhs


def _unwrap_join_or_raise(
    predicate: AbstractPredicate,
) -> tuple[ColumnReference, ColumnReference]:
    unwrapped = _unwrap_simple_join(predicate)
    if unwrapped is None:
        raise ValueError(f"Could not simplify join {predicate}")
    return unwrapped


class SimpleJoin(AbstractPredicate):
    """A simple join provides a more streamlined and direct access to join predicates.

    A simple join is a read-only predicate, i.e. it cannot be created on its own and has to be
    derived from a binary equi-join predicate. Afterward, it provides read-only access to the
    partner columns that are joined.

    Note that not all binary joins can be represented in a simplified view. In order for the view to
    work, the join must be an equi-join, i.e. using `LogicalOperator.Equal`. Furthermore, both sides
    of the join have to cannot be modified by other expressions such as function calls or
    mathematical expressions. However, cast expressions are tolerated and will simply be dropped. As
    a rule of thumb, if an expression modifies a value (such as a function call), this cannot be
    unwrapped. Therefore, a join approximately has to be of the form ``<first col> = <second col>``
    in order for the representation to work.

    The static methods `attempt_wrap`, `wrap`, `can_wrap` and `wrap_all` serve as high-level access
    points into the view. The components of the view are accessible via properties.
    In general, using `attempt_wrap` and `wrap_all` should be sufficient for most purposes.

    Parameters
    ----------
    lhs : ColumnReference | AbstractPredicate
        The left-hand side of the join. For legacy purposes, this can (counterintuitively) also be a
        full predicate. This is treated as follows: if a predicate is given, all other parameters
        need to be omitted (they are ignored either way). The predicate must be of simple form (see
        explanation above) and its components will be automatically unwrapped and used as attribute
        values for the simplified version. The introduction of the `attempt_wrap` method has made
        this mode unnecessary. It is only available to keep backwards compatibility.
    rhs : ColumnReference | None
        The right-hand side of the join. Passing *None* only works in legacy mode.
    predicate : AbstractPredicate | None
        The original predicate that is represented by this simplified view. Passing *None* only works
        in legacy mode.


    See Also
    --------
    SimpleFilter : A similar view dedicated to filter predicates

    Examples
    --------
    The best way to construct simplified views is to start with the `QueryPredicates` and extract the
    joins, e.g. by using ``views = SimpleJoin.wrap_all(query.predicates())`` or
    ``joins = SimpleJoin.wrap_all(query.predicates().joins())``.
    Notice that especially the first conversion can be "lossy": all filter predicates are dropped.
    Likewise, all joins that are not equi-joins are ignored.
    Alternatively, the `QueryPredicates` also provides a `simplify()` method that can be used to
    convert all predicates (filters and joins) into their simplified counterparts.

    Notes
    -----
    Simple joins can be used in *match* statements and provide the `lhs` and `rhs` properties.
    """

    @staticmethod
    def attempt_wrap(predicate: AbstractPredicate) -> Optional[SimpleJoin]:
        """Transforms a predicate into a simplified view. Returns *None* if that is not possible."""
        unwrapped = _unwrap_simple_join(predicate)
        if unwrapped is None:
            return None
        lhs, rhs = unwrapped
        return SimpleJoin(lhs, rhs, predicate=predicate)

    @staticmethod
    def wrap(predicate: AbstractPredicate) -> SimpleJoin:
        """Transforms a specific predicate into a simplified view. Raises an error if that is not possible.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to represent as a simplified view

        Returns
        -------
        SimpleJoin
            A simplified view wrapping the given predicate

        Raises
        ------
        ValueError
            If the predicate cannot be represented as a simplified view.
        """
        simplified = SimpleJoin.attempt_wrap(predicate)
        if simplified is None:
            raise ValueError(f"Could not simplify join {predicate}")
        return simplified

    @staticmethod
    def can_wrap(predicate: AbstractPredicate) -> bool:
        """Checks, whether a specific predicate can be represented as a simplified view.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to check

        Returns
        -------
        bool
            Whether a representation as a simplified view is possible.
        """
        unwrapped = _unwrap_simple_join(predicate)
        return unwrapped is not None

    @staticmethod
    def wrap_all(
        predicates: Iterable[AbstractPredicate],
    ) -> Sequence[SimpleJoin]:
        """Transforms specific predicates into simplified views.

        If individual predicates cannot be represented as views, they are ignored.

        Parameters
        ----------
        predicates : Iterable[AbstractPredicate]
            The predicates to represent as views. These can be arbitrary predicates, i.e. including
            filters and complex joins.

        Returns
        -------
        Sequence[SimpleJoin]
            The simplified views for all predicates that can be represented this way. The sequence of
            the views matches the sequence in the `predicates`. If the representation fails for
            individual predicates, they simply do not appear in the result. Therefore, this sequence
            may be empty if none of the predicates are valid simplified views.
        """
        views: list[SimpleJoin | None] = [
            SimpleJoin.attempt_wrap(pred) for pred in predicates
        ]
        return [v for v in views if v is not None]

    def __init__(
        self,
        lhs: ColumnReference | AbstractPredicate,
        rhs: ColumnReference | None = None,
        *,
        predicate: AbstractPredicate | None = None,
    ) -> None:
        if isinstance(lhs, AbstractPredicate):
            predicate = lhs
            lhs, rhs = _unwrap_join_or_raise(lhs)

        if lhs is None or rhs is None or predicate is None:
            raise ValueError("None of lhs, rhs, and predicate may be None")

        self._lhs = lhs
        self._rhs = rhs
        self._predicate = predicate

        hash_val = hash((lhs, rhs))
        super().__init__(hash_val)

    __slots__ = ("_lhs", "_rhs", "_predicate")
    __match_args__ = ("lhs", "rhs")

    @property
    def lhs(self) -> ColumnReference:
        """Get the left-hand side of the join."""
        return self._lhs

    @property
    def rhs(self) -> ColumnReference:
        """Get the right-hand side of the join."""
        return self._rhs

    def unwrap(self) -> AbstractPredicate:
        """Get the original predicate that is represented by this view."""
        return self._predicate

    def joins(self, partner: TableReference | ColumnReference) -> bool:
        """Checks, whether this predicate joins the given column or table."""
        if isinstance(partner, TableReference):
            return self.lhs.table == partner or self.rhs.table == partner
        elif isinstance(partner, ColumnReference):
            return self.lhs == partner or self.rhs == partner
        else:
            raise TypeError("Unexpected join partner type: " + str(type(partner)))

    @overload
    def partner_of(self, other: TableReference) -> Optional[TableReference]:
        """Provides the join partner of the given table.

        If the table is not joined, *None* is returned.
        """
        ...

    @overload
    def partner_of(self, other: ColumnReference) -> Optional[ColumnReference]:
        """Provides the join partner of the given column.

        If the column is not joined, *None* is returned.
        """
        ...

    def partner_of(
        self, other: TableReference | ColumnReference
    ) -> Optional[TableReference | ColumnReference]:
        """Provides the join partner of the given column or table.

        If the column or table is not joined, *None* is returned.
        """
        if not self.joins(other):
            return None
        if isinstance(other, TableReference):
            return self.rhs.table if self.lhs.table == other else self.lhs.table
        elif isinstance(other, ColumnReference):
            return self.rhs if self.lhs == other else self.lhs
        else:
            raise TypeError("Unexpected join partner type: " + str(type(other)))

    def is_compound(self) -> bool:
        return False

    def is_join(self) -> bool:
        return True

    def columns(self) -> set[ColumnReference]:
        return {self.lhs, self.rhs}

    def itercolumns(self) -> Iterable[ColumnReference]:
        return [self.lhs, self.rhs]

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self._predicate.iterexpressions()

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        return {(self.lhs, self.rhs)}

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]

    def accept_visitor(
        self,
        visitor: PredicateVisitor[VisitorResult] | SqlExpressionVisitor[VisitorResult],
        *args,
        **kwargs,
    ) -> VisitorResult:
        if _safe_recurse_expression_visitor(visitor):
            return visitor.visit_predicate_expr(self, *args, **kwargs)  # type: ignore
        return self._predicate.accept_visitor(visitor, *args, **kwargs)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SimpleJoin):
            return self._predicate == other._predicate

        return isinstance(other, type(self._predicate)) and self.unwrap() == other

    def __str__(self) -> str:
        return str(self._predicate)


def _collect_filter_predicates(
    predicate: AbstractPredicate,
) -> set[AbstractPredicate]:
    """Provides all base filter predicates that are contained in a specific predicate hierarchy.

    To determine, whether a given predicate is a join or a filter, the `AbstractPredicate.is_filter` method is used.

    This method handles compound predicates as follows:

    - conjunctions are un-nested, i.e. all predicates that form an *AND* predicate are collected individually
    - *OR* predicates are included with exactly those predicates from their children that are filters. If this is only true
       for a single predicate, that predicate will be returned directly.
    - *NOT* predicates are included if their child predicate is a filter

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate hierarchy to unwrap.

    Returns
    -------
    set[AbstractPredicate]
        The filter predicates that are contained in the `predicate`.

    Raises
    ------
    ValueError
        If a compound predicate has an unknown operation. This indicates a programming error or a broken invariant.
    ValueError
        If the given `predicate` is neither a `BasePredicate`, nor a `CompoundPredicate`. This indicates a modification of the
        predicate class hierarchy without the necessary adjustments to the consuming methods.

    See Also
    --------
    _collect_join_predicates
    """
    if isinstance(predicate, BasePredicate):
        return {predicate} if predicate.is_filter() else set()
    elif isinstance(predicate, CompoundPredicate):
        if predicate.operation == CompoundOperator.Or:
            or_filter_children = [
                child_pred
                for child_pred in predicate.children
                if child_pred.is_filter()
            ]
            if len(or_filter_children) < 2:
                return set(or_filter_children)
            or_filters = CompoundPredicate(CompoundOperator.Or, or_filter_children)
            return {or_filters}
        elif predicate.operation == CompoundOperator.Not:
            not_filter_children = (
                predicate.children if predicate.children.is_filter() else None
            )
            if not not_filter_children:
                return set()
            return {predicate}
        elif predicate.operation == CompoundOperator.And:
            return util.set_union(
                [_collect_filter_predicates(child) for child in predicate.children]
            )
        else:
            raise ValueError(f"Unknown operation: '{predicate.operation}'")
    else:
        raise ValueError(f"Unknown predicate type: {predicate}")


def _collect_join_predicates(
    predicate: AbstractPredicate,
) -> set[AbstractPredicate]:
    """Provides all join predicates that are contained in a specific predicate hierarchy.

    To determine, whether a given predicate is a join or a filter, the `AbstractPredicate.is_join` method is used.

    This method handles compound predicates as follows:

    - conjunctions are un-nested, i.e. all predicates that form an *AND* predicate are collected individually
    - *OR* predicates are included with exactly those predicates from their children that are joins. If this is only true for
       a single predicate, that predicate will be returned directly.
    - *NOT* predicates are included if their child predicate is a join

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate hierarchy to unwrap.

    Returns
    -------
    set[AbstractPredicate]
        The join predicates that are contained in the `predicate`

    Raises
    ------
    ValueError
        If a compound predicate has an unknown operation. This indicates a programming error or a broken invariant.
    ValueError
        If the given `predicate` is neither a `BasePredicate`, nor a `CompoundPredicate`. This indicates a modification of the
        predicate class hierarchy without the necessary adjustments to the consuming methods.

    See Also
    --------
    _collect_filter_predicates
    """
    if isinstance(predicate, BasePredicate):
        return {predicate} if predicate.is_join() else set()
    elif isinstance(predicate, CompoundPredicate):
        if predicate.operation == CompoundOperator.Or:
            or_join_children = [
                child_pred for child_pred in predicate.children if child_pred.is_join()
            ]
            if len(or_join_children) < 2:
                return set(or_join_children)
            or_joins = CompoundPredicate(CompoundOperator.Or, or_join_children)
            return {or_joins}
        elif predicate.operation == CompoundOperator.Not:
            not_join_children = (
                predicate.children if predicate.children.is_join() else None
            )
            if not not_join_children:
                return set()
            return {predicate}
        elif predicate.operation == CompoundOperator.And:
            return util.set_union(
                [_collect_join_predicates(child) for child in predicate.children]
            )
        else:
            raise ValueError(f"Unknown operation: '{predicate.operation}'")
    else:
        raise ValueError(f"Unknown predicate type: {predicate}")


class QueryPredicates:
    """The query predicates provide high-level access to all the different predicates in a query.

    Generally speaking, this class provides the most user-friendly access into the predicate
    hierarchy and should be sufficient for most use-cases. The provided methods revolve around
    identifying filter and join predicates easily, as well finding the predicates that are specified
    on specific tables.

    Parameters
    ----------
    root : Optional[AbstractPredicate]
        The root predicate of the predicate hierarchy that should be represented by the
        `QueryPredicates`. Typically, this is a conjunction of the actual predicates.
    """

    @staticmethod
    def empty_predicate() -> QueryPredicates:
        """Constructs a new predicates instance without any actual content.

        Returns
        -------
        QueryPredicates
            The predicates wrapper
        """
        return QueryPredicates(None)

    def __init__(self, root: Optional[AbstractPredicate]):
        self._root = root
        self._hash_val = hash(self._root)
        self._join_predicate_map = self._init_join_predicate_map()

    @property
    def root(self) -> AbstractPredicate:
        """Get the root predicate that represents the entire predicate hierarchy.

        Typically, this is a conjunction of the actual predicates. This conjunction can be used to start a custom traversal of
        the predicate hierarchy.

        Returns
        -------
        AbstractPredicate
            The root predicate

        Raises
        ------
        StateError
            If the predicates warpper is empty and there is no root predicate.
        """
        self._assert_not_empty()
        return self._root

    def is_empty(self) -> bool:
        """Checks, whether this predicate wrapper contains any actual predicates.

        Returns
        -------
        bool
            Whether at least one predicate was specified.
        """
        return self._root is None

    @functools.cache
    def filters(self) -> Collection[AbstractPredicate]:
        """Provides all filter predicates that are contained in the predicate hierarchy.

        By default, the distinction between filters and joins that is defined in `AbstractPredicate.is_join` is used. However,
        this behaviour can be changed by subclasses.

        This method handles compound predicates as follows:

        - conjunctions are un-nested, i.e. all predicates that form an *AND* predicate are collected individually
        - *OR* predicates are included with exactly those predicates from their children that are filters. If this is only
          true for a single predicate, that predicate will be returned directly.
        - *NOT* predicates are included if their child predicate is a filter.

        Returns
        -------
        Collection[AbstractPredicate]
            The filter predicates.
        """
        if self.is_empty():
            return []
        return _collect_filter_predicates(self._root)

    @functools.cache
    def joins(self) -> Collection[AbstractPredicate]:
        """Provides all join predicates that are contained in the predicate hierarchy.

        By default, the distinction between filters and joins that is defined in `AbstractPredicate.is_join` is used. However,
        this behaviour can be changed by subclasses.

        This method handles compound predicates as follows:

        - conjunctions are un-nested, i.e. all predicates that form an *AND* predicate are collected individually
        - *OR* predicates are included with exactly those predicates from their children that are joins. If this is only true
          for a single predicate, that predicate will be returned directly.
        - *NOT* predicates are included if their child predicate is a join.

        Returns
        -------
        Collection[AbstractPredicate]
            The join predicates
        """
        if self.is_empty():
            return []
        return _collect_join_predicates(self._root)

    @functools.cache
    def join_graph(self, *, merge_aliases: bool = False) -> nx.Graph:
        """Provides the join graph for the predicates.

        A join graph is an undirected graph, where each node corresponds to a base table and each edge corresponds to a
        join predicate between two base tables. In addition, each node is annotated by a ``predicate`` key which is a
        conjunction of all filter predicates on that table (or *None* if the table is unfiltered). Likewise, each edge is
        annotated by a ``predicate`` key that corresponds to the join predicate (which can never be *None*).

        Parameters
        ----------
        merge_aliases : bool, optional
            Whether to merge aliases of the same base table into one node. Setting this to *True*
            transforms all predicates to dictionaries that map the aliased tables to their
            filter predicates and frozensets of tables to their join predicates.
            Otherwise, the predicates are annotated directly.

        Returns
        -------
        nx.Graph
            The join graph.

        Notes
        -----
        Since our implementation of a join graph is not based on a multigraph, only binary joins can be represented. The
        determination of edges is based on `AbstractPredicate.join_partners`.
        """
        join_graph = nx.Graph()
        if self.is_empty():
            return join_graph
        assert self._root is not None

        if not merge_aliases:
            for table in self._root.tables():
                filter_predicates = self.filters_for(table)
                join_graph.add_node(table, predicate=filter_predicates)

            for join in self.joins():
                for first_col, second_col in join.join_partners():
                    join_graph.add_edge(
                        first_col.table, second_col.table, predicate=join
                    )

            return join_graph

        table2alias = collections.defaultdict(set)
        for tab in self._root.tables():
            table2alias[tab.drop_alias()].add(tab)

        table2preds = {
            tab: {alias: self.filters_for(alias) for alias in aliased}
            for tab, aliased in table2alias.items()
        }
        for table, preds in table2preds.items():
            join_graph.add_node(table, predicate=preds, aliases=table2alias[table])

        col2join: dict[frozenset[ColumnReference], AbstractPredicate] = {}
        for join in self.joins():
            for first_col, second_col in join.join_partners():
                partners = frozenset([first_col, second_col])
                col2join[partners] = join

        tab2join = collections.defaultdict(dict)
        for partners, join in col2join.items():
            tabs = frozenset([col.drop_table_alias() for col in partners])
            tab2join[tabs][partners] = join

        for tabs, joins in tab2join.items():
            first_tab, second_tab = tabs
            join_graph.add_edge(first_tab, second_tab, predicate=joins)

        return join_graph

    @functools.cache
    def filters_for(self, table: TableReference) -> Optional[AbstractPredicate]:
        """Provides all filter predicates that reference a specific table.

        If multiple individual filter predicates are specified in the query, they will be combined in one large
        conjunction.

        The determination of matching filter predicates is the same as for the `filters()` method.

        Parameters
        ----------
        table : TableReference
            The table to retrieve the filters for

        Returns
        -------
        Optional[AbstractPredicate]
            A (conjunction of) the filter predicates of the `table`, or *None* if the table is unfiltered.
        """
        if self.is_empty():
            return None
        applicable_filters = [
            filter_pred
            for filter_pred in self.filters()
            if filter_pred.contains_table(table)
        ]
        return (
            CompoundPredicate.create_and(applicable_filters)
            if applicable_filters
            else None
        )

    @functools.cache
    def joins_for(self, table: TableReference) -> Collection[AbstractPredicate]:
        """Provides all join predicates that reference a specific table.

        Each entry in the resulting collection is a join predicate between the given table and a (set of) partner tables, such
        that the partner tables in different entries in the collection are also different. If multiple join predicates are
        specified between the given table and a specific (set of) partner tables, these predicates are aggregated into one
        large conjunction.

        The determination of matching join predicates is the same as for the `joins()` method.

        Parameters
        ----------
        table : TableReference
            The table to retrieve the joins for

        Returns
        -------
        Collection[AbstractPredicate]
            The join predicates with `table`. If there are no such predicates, the collection is empty.
        """
        if self.is_empty():
            return []

        applicable_joins: list[AbstractPredicate] = [
            join_pred for join_pred in self.joins() if join_pred.contains_table(table)
        ]
        distinct_joins: dict[frozenset[TableReference], list[AbstractPredicate]] = (
            collections.defaultdict(list)
        )
        for join_predicate in applicable_joins:
            partners = {
                column.table for column in join_predicate.join_partners_of(table)
            }
            distinct_joins[frozenset(partners)].append(join_predicate)

        aggregated_predicates = []
        for join_group in distinct_joins.values():
            aggregated_predicates.append(CompoundPredicate.create_and(join_group))
        return aggregated_predicates

    def joins_between(
        self,
        first_table: TableReference | Iterable[TableReference],
        second_table: TableReference | Iterable[TableReference],
        *,
        _computation: Literal["legacy", "graph", "map"] = "map",
    ) -> Optional[AbstractPredicate]:
        """Provides the (conjunctive) join predicate that joins specific tables.

        The precise behaviour of this method depends on the provided parameters: If `first_table` or `second_table` contain
        multiple tables, all join predicates between tables from the different sets are returned (but joins from tables within
        `first_table` or from tables within `second_table` are not).

        Notice that the returned predicate might also include other tables, if they are part of a join predicate that
        also joins the given two tables.

        The performance of this method can be crucial for some applications, since the join check is often part of a very hot
        loop. Therefore, a number of different calculation strategies are implemented. If profiling shows that the application
        is slowed down heavily by the current strategy, it might be a good idea to switch to another algorithm. This can be
        achieved using the `_computation` parameter.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.
        _computation : Literal[&quot;legacy&quot;, &quot;graph&quot;, &quot;map&quot;], optional
            The specific algorithm to use for determining the join partners. The algorithms have very different performance
            and memory charactersistics. The default ``map`` setting is usually the fastest. It should only really be changed
            if there are very good reasons for it. The following settings exist:

            - *legacy*: uses a recursive strategy in case `first_table` or `second_table` contain multiple references.
              This is pretty slow, but easy to debug
            - *graph*: builds an internal join graph and merges the involved nodes from `first_table` and `second_table` to
              extract the overall join predicate directly.
            - *map*: stores a mapping between tables and their join predicates and merges these mappings to determine the
              overall join predicate

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, *None* is returned.

        Raises
        ------
        ValueError
            If the `_computation` strategy is none of the allowed values
        """
        if self.is_empty():
            return None

        if _computation == "legacy":
            return self._legacy_joins_between(first_table, second_table)
        elif _computation == "graph":
            return self._graph_based_joins_between(first_table, second_table)
        elif _computation == "map":
            return self._map_based_joins_between(first_table, second_table)
        else:
            raise ValueError(
                "Unknown computation method. Allowed values are 'legacy', 'graph', or 'map', ",
                f"not '{_computation}'",
            )

    def joins_tables(
        self,
        tables: TableReference | Iterable[TableReference],
        *more_tables: TableReference,
    ) -> bool:
        """Checks, whether specific tables are all joined with each other.

        This does not mean that there has to be a join predicate between each pair of tables, but rather that all pairs
        of tables must at least be connected through a sequence of other tables. From a graph-theory centric point of
        view this means that the join (sub-) graph induced by the given tables is connected.

        Parameters
        ----------
        tables : TableReference | Iterable[TableReference]
            The tables to check.
        *more_tables
            Additional tables that also should be included in the check. This parameter is mainly for convenience usage in
            interactive scenarios.

        Returns
        -------
        bool
            Whether the given tables can be joined without any cross product

        Raises
        ------
        ValueError
            If `tables` and `more_tables` is both empty

        Examples
        --------
        The following calls are exactly equivalent:

        .. code-block:: python

            >>> predicates.joins_tables([table1, table2, table3])
            >>> predicates.joins_tables(table1, table2, table3)
            >>> predicates.joins_table([table1, table2], table3)

        """
        if self.is_empty():
            return False

        tables = [tables] if not isinstance(tables, Iterable) else list(tables)
        tables = frozenset(set(tables) | set(more_tables))

        if not tables:
            raise ValueError("Cannot perform check for empty tables")
        if len(tables) == 1:
            table = util.simplify(tables)
            return table in self._root.tables()

        return self._join_tables_check(tables)

    def simplify(self) -> Sequence[SimpleFilter | SimpleJoin]:
        """Converts all predicates in the hierarchy into their simplified counterparts.

        See Also
        --------
        SimpleFilter : The simplified representation of predicates
        SimpleJoin : The simplified representation of join predicates
        """
        if self.is_empty():
            return []

        filters = SimpleFilter.wrap_all(self.filters())
        joins = SimpleJoin.wrap_all(self.joins())
        return filters + joins

    def all_simple(self) -> bool:
        """Checks, whether all predicates in the hierarchy can be represented as simplified views.

        See Also
        --------
        SimpleFilter : The simplified representation of predicates
        SimpleJoin : The simplified representation of join predicates
        """
        if not all(SimpleFilter.can_wrap(pred) for pred in self.filters()):
            return False
        return all(SimpleJoin.can_wrap(pred) for pred in self.joins())

    def and_(
        self, other_predicate: QueryPredicates | AbstractPredicate
    ) -> QueryPredicates:
        """Combines the current predicates with additional predicates, creating a conjunction of the two predicates.

        The input predicates, as well as the current predicates object are not modified. All changes are applied to the
        resulting predicates object.

        Parameters
        ----------
        other_predicate : QueryPredicates | AbstractPredicate
            The predicates to combine. Can also be an `AbstractPredicate`, in which case this predicate is used as the root
            for the other predicates instance.

        Returns
        -------
        QueryPredicates
            The merged predicates wrapper. Its root is roughly equivalent to ``self.root AND other_predicate.root``.
        """
        if (
            self.is_empty()
            and isinstance(other_predicate, QueryPredicates)
            and other_predicate.is_empty()
        ):
            return self
        elif (
            isinstance(other_predicate, QueryPredicates) and other_predicate.is_empty()
        ):
            return self

        other_predicate = (
            other_predicate._root
            if isinstance(other_predicate, QueryPredicates)
            else other_predicate
        )
        if self.is_empty():
            return QueryPredicates(other_predicate)

        merged_predicate = CompoundPredicate.create_and([self._root, other_predicate])
        return QueryPredicates(merged_predicate)

    @functools.cache
    def _join_tables_check(self, tables: frozenset[TableReference]) -> bool:
        """Constructs the join graph for the given tables and checks, whether it is connected.

        Parameters
        ----------
        tables : frozenset[TableReference]
            The tables to check. Has to be a frozenset to enable caching via `functools.cache`

        Returns
        -------
        bool
            Whether the specified tables induce a connected subgraph of the full join graph.
        """
        join_graph = nx.Graph()
        join_graph.add_nodes_from(tables)
        for table in tables:
            for join in self.joins_for(table):
                partner_tables = (
                    set(col.table for col in join.join_partners_of(table)) & tables
                )
                join_graph.add_edges_from(itertools.product([table], partner_tables))
        return nx.is_connected(join_graph)

    def _assert_not_empty(self) -> None:
        """Ensures that a root predicate is set

        Raises
        ------
        StateError
            If there is no root predicate
        """
        if self._root is None:
            raise StateError("No query predicates!")

    def _init_join_predicate_map(
        self,
    ) -> dict[frozenset[TableReference], AbstractPredicate]:
        """Generates the necessary mapping for `_map_based_joins_between`.

        This is a static data structure and hence can be pre-computed.

        Returns
        -------
        dict[frozenset[TableReference], AbstractPredicate]
            A mapping from a set of tables to the join predicate that is specified between those tables. If a set of tables
            does not appear in the dictionary, there is no join predicate between the specific tables.
        """
        if self.is_empty():
            return {}

        predicate_map: dict[frozenset[TableReference], AbstractPredicate] = {}
        for table in self._root.tables():
            join_partners = self.joins_for(table)
            for join_predicate in join_partners:
                partner_tables = {
                    partner.table for partner in join_predicate.join_partners_of(table)
                }
                map_key = frozenset(partner_tables | {table})
                if map_key in predicate_map:
                    continue
                predicate_map[map_key] = join_predicate
        return predicate_map

    def _legacy_joins_between(
        self,
        first_table: TableReference | Iterable[TableReference],
        second_table: TableReference | Iterable[TableReference],
    ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined using the legacy recursive structure.

        .. deprecated::
            There is no real advantage of using this method, other than slightly easier debugging. Should be removed at some
            later point in time (in which case the old ``legacy`` strategy key will be re-mapped to a different) strategy.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, *None* is returned.
        """
        if isinstance(first_table, TableReference) and isinstance(
            second_table, TableReference
        ):
            if first_table == second_table:
                return None
            first_joins: Collection[AbstractPredicate] = self.joins_for(first_table)
            matching_joins = {
                join for join in first_joins if join.joins_table(second_table)
            }
            return (
                CompoundPredicate.create_and(matching_joins) if matching_joins else None
            )

        matching_joins = set()
        first_table, second_table = (
            util.enlist(first_table),
            util.enlist(second_table),
        )
        for first in frozenset(first_table):
            for second in frozenset(second_table):
                join_predicate = self.joins_between(first, second)
                if not join_predicate:
                    continue
                matching_joins.add(join_predicate)
        return CompoundPredicate.create_and(matching_joins) if matching_joins else None

    def _graph_based_joins_between(
        self,
        first_table: TableReference | Iterable[TableReference],
        second_table: TableReference | Iterable[TableReference],
    ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined using a graph-based approach.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, *None* is returned.

        """
        join_graph = self.join_graph()
        first_table, second_table = (
            util.enlist(first_table),
            util.enlist(second_table),
        )

        if len(first_table) > 1:
            first_first_table, *remaining_first_tables = first_table
            for remaining_first_table in remaining_first_tables:
                join_graph = nx.contracted_nodes(
                    join_graph, first_first_table, remaining_first_table
                )
            first_hook = first_first_table
        else:
            first_hook = util.simplify(first_table)

        if len(second_table) > 1:
            first_second_table, *remaining_second_tables = second_table
            for remaining_second_table in remaining_second_tables:
                join_graph = nx.contracted_nodes(
                    join_graph, first_second_table, remaining_second_table
                )
            second_hook = first_second_table
        else:
            second_hook = util.simplify(second_table)

        if (first_hook, second_hook) not in join_graph.edges:
            return None
        return join_graph.edges[first_hook, second_hook]["predicate"]

    def _map_based_joins_between(
        self,
        first_table: TableReference | Iterable[TableReference],
        second_table: TableReference | Iterable[TableReference],
    ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined together using a map-based approach.

        This method is the preferred way of inferring the join predicate from the two candidate sets. It is based on static
        map data that was precomputed as part of `_init_join_predicate_map`.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, *None* is returned.

        """
        join_predicates = set()
        first_table, second_table = (
            util.enlist(first_table),
            util.enlist(second_table),
        )
        for first in first_table:
            for second in second_table:
                map_key = frozenset((first, second))
                if map_key not in self._join_predicate_map:
                    continue
                current_predicate = self._join_predicate_map[map_key]
                join_predicates.add(current_predicate)
        if not join_predicates:
            return None
        return CompoundPredicate.create_and(join_predicates)

    def __iter__(self) -> Iterator[AbstractPredicate]:
        return (list(self.filters()) + list(self.joins())).__iter__()

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._root == other._root

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._root)


class BaseClause(abc.ABC):
    """Basic interface shared by all supported clauses.

    This really is an abstract interface, not a usable clause. All inheriting clauses have to provide their own
    `__eq__` method and re-use the `__hash__` method provided by the base clause. Remember to explicitly set this up!
    The concrete hash value is constant since the clause itself is immutable. It is up to the implementing class to
    make sure that the equality/hash consistency is enforced.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete clause object
    """

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

    __slots__ = ("_hash_val",)

    def tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in the clause.

        Returns
        -------
        set[TableReference]
            All tables. This includes virtual tables if such tables are present in the clause
        """
        return util.set_union(
            expression.tables() for expression in self.iterexpressions()
        )

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced in the clause.

        Returns
        -------
        set[ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all directly contained expressions in this clause.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides access to all column in this clause.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n*
        times in this clause, it will also be returned *n* times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.

        Returns
        -------
        Iterable[ColumnReference]
            All columns exactly in the order in which they are used
        """
        raise NotImplementedError

    @abc.abstractmethod
    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        """Enables processing of the current clause by a visitor.

        Parameters
        ----------
        visitor : ClauseVisitor[VisitorResult]
            The visitor
        args
            Additional arguments that are passed to the visitor
        kwargs
            Additional keyword arguments that are passed to the visitor
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class Hint(BaseClause):
    """Hint block of a clause.

    Depending on the SQL dialect, these hints will be placed at different points in the query. Furthermore, the precise
    contents (i.e. syntax and semantic) vary from database system to system.

    Hints are differentiated in two parts:

    - preparatory statements can be executed as valid commands on the database system, e.g. optimizer settings, etc.
    - query hints are the actual hints. Typically, these will be inserted as comments at some place in the query.

    These two parts are set as parameters in the `__init__` method and are available as read-only properties
    afterwards.

    Parameters
    ----------
    preparatory_statements : str, optional
        Statements that configure the optimizer and have to be run *before* the actual query is executed. Such settings
        often configure the optimizer for the entire session and can thus influence other queries as well. Defaults to
        an empty string, which indicates that there are no such settings.
    query_hints : str, optional
        Hints that configure the optimizer, often for an individual join. These hints are executed as part of the
        actual query.

    Examples
    --------
    A hint clause for MySQL could look like this:

    .. code-block::sql

        SET optimizer_switch = 'block_nested_loop=off';
        SELECT /*+ HASH_JOIN(R S) */ R.a
        FROM R, S, T
        WHERE R.a = S.b AND S.b = T.c

    This enforces the join between tables *R* and *S* to be executed as a hash join (due to the query hint) and
    disables usage of the block nested-loop join for the entire query (which in this case only affects the join between
    tables *S* and *T*) due to the preparatory ``SET optimizer_switch`` statement.
    """

    def __init__(self, preparatory_statements: str = "", query_hints: str = ""):
        self._preparatory_statements = preparatory_statements
        self._query_hints = query_hints

        hash_val = hash((preparatory_statements, query_hints))
        super().__init__(hash_val)

    __slots__ = ("_preparatory_statements", "_query_hints")
    __match_args__ = ("preparatory_statements", "query_hints")

    @property
    def preparatory_statements(self) -> str:
        """Get the string of preparatory statements. Can be empty.

        Returns
        -------
        str
            The preparatory statements. If these are multiple statements, they are concatenated into a single string
            with appropriate separator characters between them.
        """
        return self._preparatory_statements

    @property
    def query_hints(self) -> str:
        """Get the query hint text. Can be empty.

        Returns
        -------
        str
            The hints. The string has to be understood as-is by the target database system. If multiple hints are used,
            they have to be concatenated into a single string with appropriate separator characters between them.
            Correspondingly, if the hint blocks requires a specific prefix/suffix (e.g. comment syntax), this has to
            be part of the string as well.
        """
        return self._query_hints

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_hint_clause(self, *args, **kwargs)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.preparatory_statements == other.preparatory_statements
            and self.query_hints == other.query_hints
        )

    def __str__(self) -> str:
        if self.preparatory_statements and self.query_hints:
            return self.preparatory_statements + "\n" + self.query_hints
        elif self.preparatory_statements:
            return self.preparatory_statements
        return self.query_hints


class Explain(BaseClause):
    """*EXPLAIN* block of a query.

    *EXPLAIN* queries change the execution mode of a query. Instead of focusing on the actual query result, an
    *EXPLAIN* query produces information about the internal processes of the database system. Typically, this
    includes which execution plan the DBS would choose for the query. Additionally, *EXPLAIN ANALYZE* (as for example
    supported by Postgres) provides the query plan and executes the actual query. The returned plan is then annotated
    by how the optimizer predictions match reality. Furthermore, such *ANALYZE* plans typically also contain some
    runtime statistics such as runtime of certain operators.

    Notice that there is no *EXPLAIN* keyword in the SQL standard, but all major database systems provide this
    functionality. Nevertheless, the precise syntax and semantic of an *EXPLAIN* statement depends on the actual DBS.
    The Explain clause object is modeled after Postgres and needs to adapted accordingly for different systems (see
    `db.HintService`). Especially the *EXPLAIN ANALYZE* variant is not supported by all systems.

    Parameters
    ----------
    analyze : bool, optional
        Whether the query should not only be executed as an *EXPLAIN* query, but rather as an *EXPLAIN ANALYZE*
        query. Defaults to *False* which runs the query as a pure *EXPLAIN* query.
    target_format : Optional[str], optional
        The desired output format of the query plan, if this is supported by the database system. Defaults to *None*
        which normally forces the default output format.

    See Also
    --------
    postbound.db.db.HintService.format_query

    References
    ----------

    .. PostgreSQL *EXPLAIN* command: https://www.postgresql.org/docs/current/sql-explain.html
    """

    @staticmethod
    def explain_analyze(format_type: str = "JSON") -> Explain:
        """Constructs an *EXPLAIN ANALYZE* clause with the specified output format.

        Parameters
        ----------
        format_type : str, optional
            The output format, by default ``JSON``

        Returns
        -------
        Explain
            The explain clause
        """
        return Explain(True, format_type)

    @staticmethod
    def plan(format_type: str = "JSON") -> Explain:
        """Constructs a pure *EXPLAIN* clause (i.e. without *ANALYZE*) with the specified output format.

        Parameters
        ----------
        format_type : str, optional
            The output format, by default ``JSON``

        Returns
        -------
        Explain
            The explain clause
        """
        return Explain(False, format_type)

    def __init__(self, analyze: bool = False, target_format: Optional[str] = None):
        self._analyze = analyze
        self._target_format = target_format if target_format != "" else None

        hash_val = hash((analyze, target_format))
        super().__init__(hash_val)

    __slots__ = ("_analyze", "_target_format")
    __match_args__ = ("analyze", "target_format")

    @property
    def analyze(self) -> bool:
        """Check, whether the query should be executed as *EXPLAIN ANALYZE* rather than just plain *EXPLAIN*.

        Usually, *EXPLAIN ANALYZE* executes the query and gathers extensive runtime statistics (e.g. comparing
        estimated vs. true cardinalities for intermediate nodes).

        Returns
        -------
        bool
            Whether *ANALYZE* mode is enabled
        """
        return self._analyze

    @property
    def target_format(self) -> Optional[str]:
        """Get the target format in which the *EXPLAIN* plan should be provided.

        Returns
        -------
        Optional[str]
            The output format, or *None* if this is not specified. This is never an empty string.
        """
        return self._target_format

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_explain_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.analyze == other.analyze
            and self.target_format == other.target_format
        )

    def __str__(self) -> str:
        explain_prefix = "EXPLAIN"
        explain_body = ""
        if self.analyze and self.target_format:
            explain_body = f" (ANALYZE, FORMAT {self.target_format})"
        elif self.analyze:
            explain_body = " ANALYZE"
        elif self.target_format:
            explain_body = f" (FORMAT {self.target_format})"
        return explain_prefix + explain_body


class WithQuery:
    """A single common table expression that can be referenced in the actual query.

    Each *WITH* clause can consist of multiple auxiliary common table expressions. This class models exactly one
    such query. It consists of the query as well as the name under which the temporary table can be referenced
    in the actual SQL query.

    Parameters
    ----------
    query : SqlQuery
        The query that should be used to construct the temporary common table.
    target_name : str | TableReference
        The name under which the table should be made available. If a table reference is provided, its identifier will be used.
    materialized : Optional[bool], optional
        Whether the query should be materialized or not. If this is not supported or not known, this can be set to *None*
        (the default). Since materialization is not part of the SQL standard, we do not include it in the WITH querie's
        identity.


    Raises
    ------
    ValueError
        If the `target_name` is empty
    """

    def __init__(
        self,
        query: SqlQuery,
        target_name: str | TableReference,
        *,
        materialized: Optional[bool] = None,
    ) -> None:
        if not target_name:
            raise ValueError("Target name is required")

        # TODO: model output column names -- e.g. WITH t(n) AS (SELECT 1)

        self._query = query
        self._subquery_expression = SubqueryExpression(query)
        self._target_name = (
            target_name if isinstance(target_name, str) else target_name.identifier()
        )
        self._materialized = materialized
        self._hash_val = hash((query, target_name))

    __slots__ = (
        "_query",
        "_subquery_expression",
        "_target_name",
        "_materialized",
        "_hash_val",
    )

    @property
    def query(self) -> SqlQuery:
        """The query that is used to construct the temporary table

        Returns
        -------
        SqlQuery
            The query
        """
        return self._query

    @property
    def subquery(self) -> SubqueryExpression:
        """Provides the query that constructsd the temporary table as a subquery object.

        Returns
        -------
        SubqueryExpression
            The subquery
        """
        return self._subquery_expression

    @property
    def target_name(self) -> str:
        """The table name under which the temporary table can be referenced in the actual SQL query

        Returns
        -------
        str
            The name. Will never be empty.
        """
        return self._target_name

    @property
    def target_table(self) -> TableReference:
        """The table under which the temporary CTE table can be referenced in the actual SQL query

        The only difference to `target_name` is the type of this property: it provides a proper (virtual) table
        reference object

        Returns
        -------
        TableReference
            The table. Will always be a virtual table.
        """
        return TableReference.create_virtual(self.target_name)

    @property
    def materialized(self) -> Optional[bool]:
        """Get whether this is materialized WITH query or not.

        If materialization is unknown or not supported, **None** can be used. Therefore, this property should always be checked
        against **None** before checking the actual truth value.
        Since materialization is not part of the SQL standard, we do not include it in the WITH querie's identity.

        Returns
        -------
        Optional[bool]
            The materialization status
        """
        return self._materialized

    def output_columns(self) -> Sequence[ColumnReference]:
        """Provides the columns that form the result relation of this CTE.

        The columns are named according to the following rules:

        - If the column has an alias, this name is used
        - If the column is a simple column reference, the column name is used
        - Otherwise, a generic name is used, e.g. "column_1", "column_2", etc.

        Returns
        -------
        Sequence[ColumnReference]
            The columns. Their order matches the order in which they appear in the *SELECT* clause of the query.
        """
        return [col.bind_to(self.target_table) for col in self._query.output_columns()]

    def tables(self) -> set[TableReference]:
        return self._query.tables()

    def columns(self) -> set[ColumnReference]:
        return self._query.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self._subquery_expression]

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._query.itercolumns()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._target_name == other._target_name
            and self._query == other._query
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        query_str = self._query.stringify(trailing_delimiter=False)
        if self._materialized is None:
            mat_info = ""
        else:
            mat_info = "MATERIALIZED " if self._materialized else "NOT MATERIALIZED "
        return f"{quote(self._target_name)} AS {mat_info}({query_str})"


class ValuesWithQuery(WithQuery):
    """Models a common table expression that is based on a **VALUES** clause, e.g. ``WITH t(a, b) AS (VALUES (1, 2), (3, 4))``.

    Parameters
    ----------
    values : ValuesList
        The values that should be used to construct the CTE.
    target_name : str | TableReference, optional
        The name under which the table should be made available. If a table reference is provided, its identifier will be used.
    columns : Optional[Iterable[str | ColumnReference]], optional
        The columns that should be used to construct the CTE. If no columns are provided, all columns are anonymous.
        If columns are provided, they have to match the number of columns in the values list.
    materialized : Optional[bool], optional
        Whether the query should be materialized or not. If this is not supported or not known, this can be set to *None*
        (the default). Since materialization is not part of the SQL standard, we do not include it in the WITH querie's
        identity.
    """

    def __init__(
        self,
        values: ValuesList,
        *,
        target_name: str | TableReference = "",
        columns: Optional[Iterable[str | ColumnReference]] = None,
        materialized: Optional[bool] = None,
    ) -> None:
        self._values = list(values)
        self._materialized = materialized

        if isinstance(target_name, TableReference):
            self._table = target_name
        else:
            self._table = (
                TableReference.create_virtual(target_name) if target_name else None
            )

        parsed_columns: list[ColumnReference] = []
        for col in columns if columns else []:
            if isinstance(col, ColumnReference):
                parsed_columns.append(col.bind_to(self._table))
                continue
            parsed_columns.append(ColumnReference(col, self._table))
        self._columns = tuple(parsed_columns)

        table_source = ValuesTableSource(
            values, alias=self._table, columns=self._columns
        )
        self._query = SqlQuery(
            select_clause=Select.star(),
            from_clause=From([table_source]),
        )
        super().__init__(self._query, self._table, materialized=materialized)

    __slots__ = ("_values", "_materialized", "_table", "_columns", "_query")

    @property
    def rows(self) -> ValuesList:
        """Get the values that are used to construct the CTE.

        Returns
        -------
        ValuesList
            The values
        """
        return self._values

    @property
    def cols(self) -> Sequence[ColumnReference]:
        """Get the columns that are used to construct the common table expression.

        Returns
        -------
        Sequence[ColumnReference]
            The columns
        """
        return self._columns

    def tables(self) -> set[TableReference]:
        return set()

    def columns(self) -> set[ColumnReference]:
        if not self._columns:
            return set()
        return {ColumnReference(col, self.target_table) for col in self._columns}

    def output_columns(self) -> Sequence[ColumnReference]:
        """Provides the columns that form the result relation of this CTE.

        Returns
        -------
        Sequence[ColumnExpression]
            The columns. If no columns were provided, generic column names are used.
        """
        if self._columns:
            return self._columns
        return [
            ColumnReference(f"column_{i + 1}", self.target_table)
            for i in range(len(self._values[0]))
        ]

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(row for row in self._values)

    def itercolumns(self) -> list[ColumnReference]:
        return [ColumnReference(col, self.target_table) for col in self._columns]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._values == other._values
            and self._columns == other._columns
            and self._table == other._table
        )

    __hash__ = WithQuery.__hash__

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        vals: list[str] = []
        for row in self._values:
            current_str = ", ".join(str(val) for val in row)
            vals.append(f"({current_str})")
        complete_vals_str = ", ".join(vals)

        if self._materialized is None:
            mat_info = ""
        else:
            mat_info = "MATERIALIZED " if self._materialized else "NOT MATERIALIZED "

        cols = ", ".join(quote(col.name) for col in self._columns)
        cols_str = f"({cols})" if cols else ""

        return f"{quote(self._target_name)}{cols_str} AS {mat_info}(VALUES {complete_vals_str})"


class CommonTableExpression(BaseClause):
    """The *WITH* clause of a query, consisting of at least one CTE query.

    Parameters
    ----------
    with_queries : Iterable[WithQuery]
        The common table expressions that form the WITH clause.
    recursive : bool, optional
        Whether the WITH clause is recursive or not. Defaults to *False*.

    Raises
    ------
    ValueError
        If `with_queries` does not contain any CTE

    Warnings
    --------
    The `tables()` method provides all tables that are referenced as part of the CTEs as well as their aliases.
    The `referenced_tables()` method does not include the CTE aliases. Likewise, the `aliases()` method provides all the
    aliases, but not the tables that are referenced within the CTEs.
    """

    @staticmethod
    def create_for(query: SqlQuery, target_name: str) -> CommonTableExpression:
        """Shortcut method to create a CTE clause with a single query."""
        return CommonTableExpression([WithQuery(query, target_name)])

    def __init__(self, with_queries: Iterable[WithQuery], *, recursive: bool = False):
        self._with_queries = tuple(with_queries)
        if not self._with_queries:
            raise ValueError("With queries cannnot be empty")
        self._recursive = recursive
        hash_val = hash((self._with_queries, self._recursive))
        super().__init__(hash_val)

    __slots__ = ("_with_queries", "_recursive")
    __match_args__ = ("queries",)

    @property
    def queries(self) -> Sequence[WithQuery]:
        """Get CTEs that form the *WITH* clause

        Returns
        -------
        Sequence[WithQuery]
            The CTEs in the order in which they were originally specified.
        """
        return self._with_queries

    @property
    def recursive(self) -> bool:
        """Check whether the WITH clause is recursive or not.

        Returns
        -------
        bool
            Whether the WITH clause is recursive
        """
        return self._recursive

    def tables(self) -> set[TableReference]:
        all_tables: set[TableReference] = set()
        for cte in self._with_queries:
            all_tables |= cte.tables() | {cte.target_table}
        return all_tables

    def referenced_tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in the CTEs. This does not include the CTE aliases.

        Returns
        -------
        set[TableReference]
            The tables.
        """
        return util.set_union(cte.tables() for cte in self._with_queries)

    def aliases(self) -> set[TableReference]:
        """Provides all aliases that are used in the CTEs.

        Returns
        -------
        set[TableReference]
            The aliases.
        """
        return {cte.target_table for cte in self._with_queries}

    def columns(self) -> set[ColumnReference]:
        return util.set_union(with_query.columns() for with_query in self._with_queries)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(
            with_query.iterexpressions() for with_query in self._with_queries
        )

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(
            with_query.itercolumns() for with_query in self._with_queries
        )

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_cte_clause(self, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[WithQuery]:
        return iter(self.queries)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._with_queries == other._with_queries
            and self._recursive == other._recursive
        )

    def __str__(self) -> str:
        query_str = ", ".join(str(with_query) for with_query in self._with_queries)
        recursive_str = " RECURSIVE" if self._recursive else ""
        return f"WITH{recursive_str} {query_str}"


class BaseProjection:
    """The `BaseProjection` forms the fundamental building block of a *SELECT* clause.

    Each *SELECT* clause is composed of at least one base projection. Each projection can be an arbitrary
    `SqlExpression` (rules and restrictions of the SQL standard are not enforced here). In addition, each projection
    can receive a target name as in ``SELECT foo AS f FROM bar``.

    Parameters
    ----------
    expression : SqlExpression
        The expression that is used to calculate the column value. In the simplest case, this can just be a
        `ColumnExpression`, which provides the column values directly.
    target_name : str, optional
        An optional name under which the column should be accessible. Defaults to an empty string, which indicates that
        the original column value or a system-specific modification of that value should be used. The latter case
        mostly applies to columns which are modified in some way, e.g. by a mathematical expression or a function call.
        Depending on the specific database system, the default column name could just be the function name, or the
        function name along with all its parameters.

    """

    @staticmethod
    def count_star(target_name: str = "") -> BaseProjection:
        """Shortcut method to create a ``COUNT(*)`` projection.

        Parameters
        ----------
        target_name : str, optional
            An optional name under which the column should be accessible.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(
            FunctionExpression("count", [StarExpression()]),
            target_name=target_name,
        )

    @staticmethod
    def star() -> BaseProjection:
        """Shortcut method to create a ``*`` (as in ``SELECT * FROM R``) projection.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(StarExpression())

    @staticmethod
    def create_count(
        column: ColumnReference | SqlExpression,
        *,
        target_name: str = "",
        distinct: bool = False,
    ) -> BaseProjection:
        """Shorthand to create a projection with a COUNT() aggregation."""
        fn = FunctionExpression.create_count(column, distinct=distinct)
        return BaseProjection(fn, target_name)

    @staticmethod
    def create_min(
        column: ColumnReference | SqlExpression, *, target_name: str = ""
    ) -> BaseProjection:
        """Shorthand to create a projection with a MIN() aggregation."""
        fn = FunctionExpression.create_min(column)
        return BaseProjection(fn, target_name)

    @staticmethod
    def create_max(
        column: ColumnReference | SqlExpression, *, target_name: str = ""
    ) -> BaseProjection:
        """Shorthand to create a projection with a MAX() aggregation."""
        fn = FunctionExpression.create_max(column)
        return BaseProjection(fn, target_name)

    @staticmethod
    def create_sum(
        column: ColumnReference | SqlExpression, *, target_name: str = ""
    ) -> BaseProjection:
        """Shorthand to create a projection with a SUM() aggregation."""
        fn = FunctionExpression.create_sum(column)
        return BaseProjection(fn, target_name)

    @staticmethod
    def create_window(
        fn: FunctionExpression | str,
        *,
        partitioning: Optional[Sequence[SqlExpression | ColumnReference]] = None,
        ordering: Optional[OrderBy] = None,
        filter_condition: Optional[AbstractPredicate] = None,
        target_name: str = "",
    ) -> BaseProjection:
        """Shorthand to create a projection with a window function."""
        window = WindowExpression(
            fn,
            partitioning=partitioning,
            ordering=ordering,
            filter_condition=filter_condition,
        )
        return BaseProjection(window, target_name)

    @staticmethod
    def create_function(
        fn: str,
        *,
        args: Optional[Sequence[SqlExpression]] = None,
        kwargs: Optional[Mapping[str, SqlExpression]] = None,
        distinct: bool = False,
        filter_where: Optional[AbstractPredicate] = None,
        target_name: str = "",
    ) -> BaseProjection:
        """Shorthand to create a project for an (arbitrary) function.

        All parameters (except for the target name) are passed to the corresponding
        `FunctionExpression`, see its documentation for details.
        """
        expr = FunctionExpression(
            fn, args, keyword_args=kwargs, distinct=distinct, filter_where=filter_where
        )
        return BaseProjection(expr, target_name)

    @staticmethod
    def column(col: ColumnReference, target_name: str = "") -> BaseProjection:
        """Shortcut method to create a projection for a specific column.

        Parameters
        ----------
        col : ColumnReference
            The column that should be projected
        target_name : str, optional
            An optional name under which the column should be available.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(ColumnExpression(col), target_name)

    def __init__(self, expression: SqlExpression, target_name: str = ""):
        if not expression:
            raise ValueError("Expression must be set")
        self._expression = expression
        self._target_name = target_name
        self._hash_val = hash((expression, target_name))

    __slots__ = ("_expression", "_target_name", "_hash_val")

    @property
    def expression(self) -> SqlExpression:
        """Get the expression that forms the column.

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._expression

    @property
    def target_name(self) -> str:
        """Get the alias under which the column should be accessible.

        Can be empty to indicate the absence of a target name.

        Returns
        -------
        str
            The name
        """
        return self._target_name

    def is_star(self) -> bool:
        """Checks, whether this projection is a star projection."""
        return isinstance(self._expression, StarExpression)

    def columns(self) -> set[ColumnReference]:
        return self.expression.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.expression.itercolumns()

    def tables(self) -> set[TableReference]:
        return self.expression.tables()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.expression == other.expression
            and self.target_name == other.target_name
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self.target_name:
            return str(self.expression)
        return f"{self.expression} AS {quote(self.target_name)}"


DistinctType = Literal["all", "none", "on"]
"""The different modes a *SELECT* clause can use for duplicate elimination.

- *all*: All columns are used for duplicate elimination
- *none*: No duplicate elimination is performed
- *on*: A specific subset of columns is used for duplicate elimination
"""


class Select(BaseClause):
    """The *SELECT* clause of a query.

    This is the only required part of a query. Everything else is optional and can be left out. (Notice that PostBOUND
    is focused on SPJ-queries, hence there are no *INSERT*, *UPDATE*, or *DELETE* queries)

    A *SELECT* clause simply consists of a number of individual projections (see `BaseProjection`), its `targets`.

    Parameters
    ----------
    targets : BaseProjection | Sequence[BaseProjection]
        The individual projection(s) that form the *SELECT* clause
    distinct : Iterable[SqlExpression] | bool, optional
        Whether a duplicate elimination should be performed. By default, this is *False* indicating no duplicate elimination.
        If *True*, rows are eliminated based on all columns. Alternatively, a *DISTINCT ON* clause can be created by specifying
        the columns that should be used for duplicate elimination.

    Raises
    ------
    ValueError
        If the `targets` are empty.
    """

    @staticmethod
    def count_star(*, target_name: str = "") -> Select:
        """Shortcut method to create a ``SELECT COUNT(*)`` clause.

        Parameters
        ----------
        target_name : str, optional
            An optional name under which the column should be accessible
            (e.g., ``COUNT(*) AS freq``).

        Returns
        -------
        Select
            The clause
        """
        return Select(BaseProjection.count_star(target_name))

    @staticmethod
    def star(*, distinct: Iterable[SqlExpression] | bool = False) -> Select:
        """Shortcut to create a ``SELECT *`` clause.

        Parameters
        ----------
        distinct : Iterable[SqlExpression] | bool
            Whether a duplicate elimination should be performed. By default, this is *False* indicating no duplicate
            elimination. If *True*, rows are eliminated based on all columns. Alternatively, a *DISTINCT ON* clause can be
            created by specifying the columns that should be used for duplicate elimination.

        Returns
        -------
        Select
            The clause
        """
        return Select(BaseProjection.star(), distinct=distinct)

    @staticmethod
    def create_for(
        columns: ColumnReference
        | SqlExpression
        | Iterable[ColumnReference | SqlExpression],
        *,
        distinct: Iterable[SqlExpression] | bool = False,
    ) -> Select:
        """Full factory method to accompany `star` and `count_star` factory methods.

        This is basically the same as calling the `__init__` method directly.

        Parameters
        ----------
        columns : ColumnReference | Iterable[ColumnReference]
            The columns that should form the projection
        distinct : Iterable[SqlExpression] | bool, optional
            Whether a duplicate elimination should be performed. By default, this is *False* indicating no duplicate
            elimination. If *True*, rows are eliminated based on all columns. Alternatively, a *DISTINCT ON* clause can be
            created by specifying the columns that should be used for duplicate elimination.

        Returns
        -------
        Select
            The clause
        """
        columns = util.enlist(columns)
        target_columns = [
            BaseProjection.column(column)
            if isinstance(column, ColumnReference)
            else BaseProjection(column)
            for column in columns
        ]
        return Select(target_columns, distinct=distinct)

    @staticmethod
    def create_count(
        column: ColumnReference | SqlExpression | None = None,
        *,
        target_name: str = "",
    ) -> Select:
        """Shorthand to create a SELECT clause with a single COUNT() expression.

        If no column is given, defaults to a COUNT(*) expression.
        """
        if column is None:
            return Select.count_star()
        return Select(BaseProjection.create_count(column, target_name=target_name))

    @staticmethod
    def create_min(
        column: ColumnReference | SqlExpression, *, target_name: str = ""
    ) -> Select:
        """Shorthand to create a SELECT clause with a single MIN() expression."""
        return Select(BaseProjection.create_min(column, target_name=target_name))

    @staticmethod
    def create_max(
        column: ColumnReference | SqlExpression, *, target_name: str = ""
    ) -> Select:
        """Shorthand to create a SELECT clause with a single MAX() expression."""
        return Select(BaseProjection.create_max(column, target_name=target_name))

    @staticmethod
    def create_sum(
        column: ColumnReference | SqlExpression, *, target_name: str = ""
    ) -> Select:
        """Shorthand to create a SELECT clause with a single SUM() expression."""
        return Select(BaseProjection.create_sum(column, target_name=target_name))

    def __init__(
        self,
        targets: BaseProjection
        | SqlExpression
        | Sequence[BaseProjection | SqlExpression],
        *,
        distinct: Iterable[SqlExpression] | bool = False,
    ) -> None:
        if not targets:
            raise ValueError("At least one target must be specified")

        match targets:
            case BaseProjection():
                self._targets = (targets,)
            case SqlExpression():
                self._targets = (BaseProjection(targets),)
            case Sequence():
                self._targets = tuple(
                    t if isinstance(t, BaseProjection) else BaseProjection(t)
                    for t in targets
                )
            case _:
                raise ValueError(f"Unexpected targets value: {targets}")

        match distinct:
            case True:
                self._distinct_type = "all"
                self._distinct_cols = ()
            case False:
                self._distinct_type = "none"
                self._distinct_cols = ()
            case _:
                self._distinct_type = "on" if distinct else "none"
                self._distinct_cols = tuple(distinct)

        hash_val = hash((self._distinct_type, self._distinct_cols, self._targets))
        super().__init__(hash_val)

    __slots__ = ("_targets", "_distinct_type", "_distinct_cols")
    __match_args__ = ("targets", "distinct", "distinct_on")

    @property
    def targets(self) -> Sequence[BaseProjection]:
        """Get all projections.

        Returns
        -------
        Sequence[BaseProjection]
            The projections in the order in which they were originally specified
        """
        return self._targets

    @property
    def distinct(self) -> DistinctType:
        return self._distinct_type

    @property
    def distinct_on(self) -> Sequence[SqlExpression]:
        return self._distinct_cols

    def is_star(self) -> bool:
        """Checks, whether the clause is simply *SELECT \\**."""
        return len(self._targets) == 1 and self._targets[0] == BaseProjection.star()

    def is_distinct(self) -> bool:
        """Checks, whether this is a *SELECT DISTINCT* clause (including a *DISTINCT ON*)."""
        return self._distinct_type != "none"

    def distinct_specifier(self) -> Iterable[SqlExpression] | bool:
        """Provides a precise description of the distinct qualifier.

        Returns
        -------
        Iterable[SqlExpression] | bool
            Output should be interpreted as follows:

            - *True*: plain *DISTINCT* over all columns
            - *False*: no *DISTINCT* qualifier
            - *Iterable[SqlExpression]*: *DISTINCT ON* clause over the given expressions
        """
        match self._distinct_type:
            case "all":
                return True
            case "none":
                return False
            case "on":
                return self._distinct_cols
            case _:
                raise RuntimeError("Invalid distinct type, something is severly broken")

    def star_expressions(self) -> Iterable[BaseProjection]:
        """Provides all * and R.* expressions.

        Returns
        -------
        Iterable[BaseProjection]
            The star expressions. Can be empty if no star expressions are used.
        """
        return [target for target in self.targets if target.is_star()]

    def columns(self) -> set[ColumnReference]:
        return util.set_union(target.columns() for target in self.targets)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(target.itercolumns() for target in self.targets)

    def tables(self) -> set[TableReference]:
        return util.set_union(target.tables() for target in self.targets)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [target.expression for target in self.targets]

    def output_names(self) -> dict[str, ColumnReference]:
        """Output names map the alias of each column to the actual column.

        For example, consider a query ``SELECT R.a AS foo, R.b AS bar FROM R``. Calling `output_names` on this query
        provides the dictionary ``{'foo': R.a, 'bar': R.b}``.

        Currently, this method only works for 1:1 mappings and other aliases are ignored. For example, consider a query
        ``SELECT my_udf(R.a, R.b) AS c FROM R``. Here, a user-defined function is used to combine the values of *R.a*
        and *R.b* to form an output column *c*. Such a projection is ignored by `output_names`.

        Returns
        -------
        dict[str, ColumnReference]
            A mapping from the column target name to the original column.
        """
        output = {}
        for projection in self.targets:
            if not projection.target_name:
                continue
            source_columns = projection.expression.columns()
            if len(source_columns) != 1:
                continue
            output[projection.target_name] = util.simplify(source_columns)
        return output

    def local_columns(self) -> set[str]:
        """Get all column names that are defined by the *SELECT* clause and not provided by the underlying tables.

        For example, consider a query ``SELECT R.a, R.b AS foo FROM R``. This query defines the local column *foo*,
        while *R.a* is provided by the underlying table *R*.
        """
        return {target.target_name for target in self.targets if target.target_name}

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_select_clause(self, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self) -> Iterator[BaseProjection]:
        return iter(self.targets)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self._distinct_type == other._distinct_type
            and self._targets == other._targets
            and self._distinct_cols == other._distinct_cols
        )

    def __str__(self) -> str:
        if self.is_distinct():
            distinct_cols = ", ".join(str(col) for col in self.distinct_on)
            select_str = (
                f"SELECT DISTINCT ON ({distinct_cols})"
                if distinct_cols
                else "SELECT DISTINCT"
            )
        else:
            select_str = "SELECT"
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"


class TableSource(abc.ABC):
    """A table source models a relation that can be scanned by the database system, filtered, joined, ...

    This is what is commonly reffered to as a *table* or a *relation* and forms the basic item of a *FROM* clause. In
    an SQL query the items of the *FROM* clause can originate from a number of different concepts. In the simplest
    case, this is just a physical table (e.g. ``SELECT * FROM R, S, T WHERE ...``), but other queries might reference
    subqueries or common table expressions in the *FROM* clause (e.g.
    ``SELECT * FROM R, (SELECT * FROM S, T WHERE ...) WHERE ...``). This class models the similarities between these
    concepts. Specific sub-classes implement them for the concrete kind of source (e.g. `DirectTableSource` or
    `SubqueryTableSource`).
    """

    @abc.abstractmethod
    def tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in the source.

        For plain table sources this will just be the actual table. For more complicated structures, such as subquery
        sources, this will include all tables of the subquery as well.

        Returns
        -------
        set[TableReference]
            The tables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all column sthat are referenced in the source.

        For plain table sources this will be empty, but for more complicate structures such as subquery source, this
        will include all columns that are referenced in the subquery.

        Returns
        -------
        set[ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all directly contained expressions in the source.

        For plain table sources this will be empty, but for subquery sources, etc. all expressions are returned. Nested
        expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression` interface for
        details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides access to all column in the source.

        For plain table sources this will be empty, but for subquery sources, etc. all expressions are returned. In
        contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n*
        times in this source, it will also be returned *n* times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.

        Returns
        -------
        Iterable[ColumnReference]
            All columns exactly in the order in which they are used
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predicates(self) -> QueryPredicates | None:
        """Provides all predicates that are contained in the source.

        For plain table sources this will be *None*, but for subquery sources, etc. all predicates are returned.

        Returns
        -------
        QueryPredicates | None
            The predicates or *None* if the source does not allow predicates or simply does not contain any.
        """
        raise NotImplementedError


class DirectTableSource(TableSource):
    """Models a plain table that is directly referenced in a *FROM* clause, e.g. *R* in ``SELECT * FROM R, S``.

    Parameters
    ----------
    table : TableReference
        The table that is sourced
    """

    @staticmethod
    def create_for(table_name: str, alias: str = "") -> DirectTableSource:
        """Shorthand to create a direct table source from a table name and an optional alias."""
        table = TableReference(table_name, alias=alias)
        return DirectTableSource(table)

    def __init__(self, table: TableReference) -> None:
        self._table = table

    __slots__ = ("_table",)
    __match_args__ = ("table",)

    @property
    def table(self) -> TableReference:
        """Get the table that is sourced.

        This can be a virtual table (e.g. for CTEs), but will most commonly be an actual table.

        Returns
        -------
        TableReference
            The table.
        """
        return self._table

    def identifier(self) -> str:
        """Get the table's identifier."""
        return self._table.identifier()

    def tables(self) -> set[TableReference]:
        return {self._table}

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def predicates(self) -> QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return hash(self._table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __repr__(self) -> str:
        return str(self._table)

    def __str__(self) -> str:
        return str(self._table)


class SubqueryTableSource(TableSource):
    """Models subquery that is referenced as a virtual table in the *FROM* clause.

    Consider the example query ``SELECT * FROM R, (SELECT * FROM S, T WHERE S.a = T.b) AS s_t WHERE R.c = s_t.a``.
    In this query, the subquery *s_t* would be represented as a subquery table source.

    Parameters
    ----------
    query : SqlQuery | SubqueryExpression
        The query that is sourced as a subquery
    target_name : str | TableReference, optional
        The name under which the subquery should be made available. Can empty for an anonymous subquery.
    lateral : bool, optional
        Whether the subquery should be executed as a lateral join. Defaults to *False*.

    Raises
    ------
    ValueError
        If the `target_name` is empty
    """

    def __init__(
        self,
        query: SqlQuery | SubqueryExpression,
        target_name: str | TableReference = "",
        *,
        lateral: bool = False,
    ) -> None:
        self._subquery_expression = (
            query
            if isinstance(query, SubqueryExpression)
            else SubqueryExpression(query)
        )
        self._target_name = (
            target_name if isinstance(target_name, str) else target_name.identifier()
        )
        self._lateral = lateral
        self._hash_val = hash(
            (self._subquery_expression, self._target_name, self._lateral)
        )

    __slots__ = (
        "_subquery_expression",
        "_target_name",
        "_lateral",
        "_hash_val",
    )
    __match_args__ = ("query", "target_name", "lateral")

    @property
    def query(self) -> SqlQuery:
        """Get the query that is sourced as a virtual table.

        Returns
        -------
        SqlQuery
            The query
        """
        return self._subquery_expression.query

    @property
    def target_name(self) -> str:
        """Get the name under which the virtual table can be accessed in the actual query.

        Returns
        -------
        str
            The name. Can be empty for an anonymous subquery.
        """
        return self._target_name

    @property
    def target_table(self) -> Optional[TableReference]:
        """Get the name under which the virtual table can be accessed in the actual query.

        The only difference to `target_name` this return type: this property provides the name as a proper table
        reference, rather than a string.

        Returns
        -------
        Optional[TableReference]
            The table. This will always be a virtual table. Can be **None** for an anonymous subquery.
        """
        return (
            TableReference.create_virtual(self._target_name)
            if self.target_name
            else None
        )

    @property
    def expression(self) -> SubqueryExpression:
        """Get the query that is used to construct the virtual table, as a subquery expression.

        Returns
        -------
        SubqueryExpression
            The subquery.
        """
        return self._subquery_expression

    @property
    def lateral(self) -> bool:
        """Get, whether this is a lateral join.

        Returns
        -------
        bool
            Whether this is a lateral join
        """
        return self._lateral

    def output_columns(self) -> Sequence[ColumnReference]:
        """Provides the columns that form the result relation of this subquery.

        The columns are named according to the following rules:

        - If the column has an alias, this name is used
        - If the column is a simple column reference, the column name is used
        - Otherwise, a generic name is used, e.g. "column_1", "column_2", etc.

        Returns
        -------
        Sequence[ColumnReference]
            The columns. Their order matches the order in which they appear in the *SELECT* clause of the query.
        """
        if not self.target_name:
            return self._subquery_expression.query.output_columns()
        return [
            col.bind_to(self.target_table)
            for col in self._subquery_expression.query.output_columns()
        ]

    def tables(self) -> set[TableReference]:
        return self._subquery_expression.tables() | {self.target_table}

    def columns(self) -> set[ColumnReference]:
        return self._subquery_expression.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self._subquery_expression]

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._subquery_expression.itercolumns()

    def predicates(self) -> QueryPredicates | None:
        return self._subquery_expression.query.predicates()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._subquery_expression == other._subquery_expression
            and self._target_name == other._target_name
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        lateral_str = "LATERAL " if self._lateral else ""
        query_str = self._subquery_expression.query.stringify(trailing_delimiter=False)
        target_str = f" AS {quote(self._target_name)}" if self._target_name else ""
        return f"{lateral_str}({query_str}){target_str}"


ValuesList = Iterable[tuple[StaticValueExpression, ...]]
"""Represents a nested list of rows. As an invariant, all tuples in the list should have the same length."""


class ValuesTableSource(TableSource):
    """Represents a virtual table that is constructed from a literal list of values.

    Parameters
    ----------
    values : ValuesList
        The available table rows.
    alias : str | TableReference, optional
        The name under which the virtual table can be accessed in the actual query. If this is empty, an anonymous table is
        created.
    columns : Optional[Iterable[str | ColumnReference]], optional
        The names of the columns that are available in the virtual table. The length of this list must match the length
        of the tuples in the `values` list. Alternatively, an empty list can be provided, in which case the columns will
        be named automatically.
    """

    def __init__(
        self,
        values: ValuesList,
        *,
        alias: str | TableReference = "",
        columns: Optional[Iterable[str | ColumnReference]] = None,
    ) -> None:
        self._values = tuple(values)

        if isinstance(alias, TableReference):
            self._table = alias
        else:
            self._table = TableReference.create_virtual(alias) if alias else None

        parsed_columns: list[ColumnReference] = []
        for col in columns if columns else []:
            if isinstance(col, ColumnReference):
                parsed_columns.append(col.bind_to(self._table))
                continue
            parsed_columns.append(ColumnReference(col, self._table))
        self._columns = tuple(parsed_columns)

        self._hash_val = hash((self._table, self._columns, self._values))

    __slots__ = ("_values", "_table", "_columns", "_hash_val")
    __match_args__ = ("rows", "alias", "cols")

    @property
    def rows(self) -> ValuesList:
        """Get the rows that are available in the virtual table.

        Returns
        -------
        ValuesList
            The rows
        """
        return self._values

    @property
    def table(self) -> Optional[TableReference]:
        """Get the name under which the virtual table can be accessed in the actual query.

        Returns
        -------
        Optional[TableReference]
            The table. Can be **None** for an anonymous table.
        """
        return self._table

    @property
    def alias(self) -> str:
        """Get the name under which the virtual table can be accessed in the actual query.

        Returns
        -------
        str
            The name. Can be empty for an anonymous table.
        """
        return self._table.alias if self._table else ""

    @property
    def cols(self) -> Sequence[ColumnReference]:
        """Get the columns that are available in the virtual table.

        Returns
        -------
        Sequence[ColumnReference]
            The columns. Can be empty if the columns are not named explicitly.
        """
        return self._columns

    def tables(self) -> set[TableReference]:
        return {self._table}

    def columns(self) -> set[ColumnReference]:
        return set(self._columns)

    def output_columns(self) -> Sequence[ColumnReference]:
        """Provides the columns that form the result relation.

        Returns
        -------
        Sequence[ColumnReference]
            The columns. If column names have been explicitly provided, these are used. Otherwise, generic names are
            used, e.g. "column_1", "column_2", etc.
        """
        if self._columns:
            return self._columns
        return [
            ColumnReference(f"column_{i}", self._table)
            for i in range(len(self._values[0]))
        ]

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(row for row in self._values)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._columns

    def predicates(self) -> QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._table == other._table
            and self._columns == other._columns
            and self._values == other._values
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        vals = []
        for row in self._values:
            current_str = ", ".join(str(val) for val in row)
            vals.append(f"({current_str})")

        complete_vals_str = ", ".join(vals)
        cols = ", ".join(quote(col.name) for col in self._columns)
        cols_str = f" ({cols})" if cols else ""
        tab_str = (
            f" AS {quote(self._table.identifier())}{cols_str}" if self._table else ""
        )

        return f"(VALUES {complete_vals_str}){tab_str}"


class FunctionTableSource(TableSource):
    """Models a table that is constructed from a function call, e.g. as in *SELECT * FROM my_udf(42)*.

    Parameters
    ----------
    function : FunctionExpression
        The function that computes the temporary relation.
    alias : str | TableReference, optional
        The name under which the virtual table can be accessed in the actual query. If this is empty, an anonymous table is
        created.
    """

    def __init__(
        self, function: FunctionExpression, *, alias: str | TableReference = ""
    ) -> None:
        self._function = function

        if isinstance(alias, TableReference):
            self._alias = alias.make_virtual()
        else:
            self._alias = TableReference.create_virtual(alias) if alias else None

        self._hash_val = hash((self._function, self._alias))

    __slots__ = ("_function", "_alias", "_hash_val")
    __match_args__ = ("function", "alias")

    @property
    def function(self) -> FunctionExpression:
        """Get the function that computes the temporary relation."""
        return self._function

    @property
    def target_name(self) -> str:
        """Get the name under which the virtual table can be accessed in the actual query.

        For anonymous tables, this is an empty string.
        """
        return self._alias.identifier() if self._alias else ""

    @property
    def target_table(self) -> Optional[TableReference]:
        """Get the virtual table that contains the tuples.

        This will always be a virtual table, or *None* for anonymous tables.

        This property differs from `target_name` only by its return type.
        """
        return self._alias

    def alias(self) -> str:
        """Get the name under which the virtual table can be accessed in the actual query."""
        return self._alias.identifier() if self._alias else ""

    def tables(self) -> set[TableReference]:
        return self._function.tables() | {self._alias} if self._alias else set()

    def columns(self) -> set[ColumnReference]:
        return self._function.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self._function]

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._function.itercolumns()

    def predicates(self) -> QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._function == other._function
            and self._alias == other._alias
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self._alias:
            return str(self._function)
        return f"{self._function} AS {quote(self._alias.identifier())}"


class JoinType(enum.Enum):
    """Indicates the type of a join using the explicit *JOIN* syntax, e.g. *OUTER JOIN* or *NATURAL JOIN*.

    The names of the individual values should be pretty self-explanatory and correspond entirely to the names in the
    SQL standard.
    """

    InnerJoin = "JOIN"
    OuterJoin = "FULL OUTER JOIN"
    LeftJoin = "LEFT OUTER JOIN"
    RightJoin = "RIGHT OUTER JOIN"
    CrossJoin = "CROSS JOIN"

    NaturalInnerJoin = "NATURAL JOIN"
    NaturalOuterJoin = "NATURAL OUTER JOIN"
    NaturalLeftJoin = "NATURAL LEFT JOIN"
    NaturalRightJoin = "NATURAL RIGHT JOIN"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.value


AutoJoins = {
    JoinType.CrossJoin,
    JoinType.NaturalInnerJoin,
    JoinType.NaturalOuterJoin,
    JoinType.NaturalLeftJoin,
    JoinType.NaturalRightJoin,
}
"""Automatic joins are those joins that use the *JOIN* syntax, but do not require a predicate to work.

Examples include *CROSS JOIN* and *NATURAL JOIN*.
"""


class JoinTableSource(TableSource):
    """Models a table that is referenced in a *FROM* clause using the explicit *JOIN* syntax.

    Such a table source consists of three parts:

    1. the left-hand side table source
    2. the right-hand side table source being joined to the left-hand side
    3. an (optional) join condition that specifies how the two tables are joined, if the specific join requires such a
       predicate

    Parameters
    ----------
    left : TableSource
        The left-hand side of the join
    right : TableSource
        The right-hand side of the join
    join_condition : Optional[AbstractPredicate], optional
        The predicate that is used to join the specified table with the other tables of the *FROM* clause. For most
        joins this is a required argument in order to create a valid SQL query (e.g. *LEFT JOIN* or *INNER JOIN*),
        but there are some joins without a condition (e.g. *CROSS JOIN* and *NATURAL JOIN*).
    join_type : JoinType, optional
        The specific join that should be performed. Defaults to `JoinType.InnerJoin`.
    """

    def __init__(
        self,
        left: TableSource,
        right: TableSource,
        *,
        join_condition: Optional[AbstractPredicate] = None,
        join_type: JoinType = JoinType.InnerJoin,
    ) -> None:
        if join_condition is None and join_type not in AutoJoins:
            raise ValueError(
                "Join condition is required for this join type: " + str(join_type)
            )

        self._left = left
        self._right = right
        self._join_condition = join_condition
        self._join_type = join_type if join_condition else JoinType.CrossJoin
        self._hash_val = hash(
            (self._left, self._right, self._join_condition, self._join_type)
        )

    __slots__ = (
        "_left",
        "_right",
        "_join_condition",
        "_join_type",
        "_hash_val",
    )
    __match_args__ = ("left", "right", "join_condition", "join_type")

    @property
    def left(self) -> TableSource:
        """Get the left-hand side of the join.

        Returns
        -------
        TableSource
            The join partner. Can be anything from a plain base table, to a subquery, to another join.
        """
        return self._left

    @property
    def right(self) -> TableSource:
        """Get the right-hand side of the join.

        Returns
        -------
        TableSource
            The join partner. Can be anything from a plain base table, to a subquery, to another join.
        """
        return self._right

    @property
    def source(self) -> TableSource:
        """Get the actual table being joined. This can be a proper table or a subquery.

        .. deprecated:: 0.10.0
            Use `left` instead. This is an artifact of the old mosql-based query representation

        Returns
        -------
        TableSource
            The table
        """
        return self._left

    @property
    def joined_table(self) -> JoinTableSource:
        """Get the nested join statements contained in this join.

        .. deprecated:: 0.10.0
            Use `right` instead. This is an artifact of the old mosql-based query representation

        A nested join is a *JOIN* statement within a *JOIN* statement, as in
        ``SELECT * FROM R JOIN (S JOIN T ON a = b) ON a = c``.

        Returns
        -------
        JoinTableSource
            The nested joins, can be empty if there are no such joins.
        """
        return self._right

    @property
    def join_condition(self) -> Optional[AbstractPredicate]:
        """Get the predicate that is used to determine matching tuples from the table.

        This can be *None* if the specific `join_type` does not require or allow a join condition (e.g.
        *NATURAL JOIN*).

        Returns
        -------
        Optional[AbstractPredicate]
            The condition if it is specified, *None* otherwise.
        """
        return self._join_condition

    @property
    def join_type(self) -> JoinType:
        """Get the kind of join that should be performed.

        Returns
        -------
        JoinType
            The join type
        """
        return self._join_type

    def base_table(self) -> TableReference:
        """Provide the table that is farthest to the left in the join chain.

        For subqueries or **VALUES** clauses, this will return the alias of the expression, i.e. the name of the virtual table
        that is created for the subquery or **VALUES** clause.

        Returns
        -------
        TableReference
            The table
        """
        match self._left:
            case DirectTableSource():
                return self._left.table
            case JoinTableSource():
                return self._left.base_table()
            case SubqueryTableSource():
                return self._left.target_table
            case ValuesTableSource():
                return self._left.table
            case _:
                raise TypeError("Unknown table source type: " + str(type(self._left)))

    def tables(self) -> set[TableReference]:
        return self._left.tables() | self._right.tables()

    def columns(self) -> set[ColumnReference]:
        condition_columns = (
            self._join_condition.columns() if self._join_condition else set()
        )
        return self._left.columns() | self._right.columns() | condition_columns

    def iterexpressions(self) -> Iterable[SqlExpression]:
        left_expressions = list(self._left.iterexpressions())
        right_expressions = list(self._right.iterexpressions())
        condition_expressions = (
            list(self._join_condition.iterexpressions()) if self._join_condition else []
        )
        return left_expressions + right_expressions + condition_expressions

    def itercolumns(self) -> Iterable[ColumnReference]:
        left_columns = list(self._left.itercolumns())
        right_columns = list(self._right.itercolumns())
        condition_columns = (
            list(self._join_condition.itercolumns()) if self._join_condition else []
        )
        return left_columns + right_columns + condition_columns

    def predicates(self) -> Optional[QueryPredicates]:
        all_predicates: list[AbstractPredicate] = []

        left_predicates = self._left.predicates()
        if left_predicates:
            all_predicates.append(left_predicates.root)
        right_predicates = self._right.predicates()
        if right_predicates:
            all_predicates.append(right_predicates.root)
        if self._join_condition:
            all_predicates.append(self._join_condition)

        return (
            QueryPredicates(CompoundPredicate.create_and(all_predicates))
            if all_predicates
            else None
        )

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._left == other._left
            and self._right == other._right
            and self._join_condition == other._join_condition
            and self._join_type == other._join_type
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.join_type in AutoJoins:
            return f"{self.source} {self.join_type} {self.joined_table}"
        return f"{self.source} {self.join_type} {self.joined_table} ON {self.join_condition}"


TableType = TypeVar("TableType", bound=TableSource)


class From(BaseClause, Generic[TableType]):
    """The *FROM* clause models which tables should be selected and potentially how they are combined.

    A *FROM* clause permits arbitrary source items and does not enforce a specific structure or semantic on them.
    This puts the user in charge to generate a valid and meaningful structure. For example, the model allows for the
    first item to be a `JoinTableSource`, even though this is not valid SQL. Likewise, no duplicate checks are
    performed.

    To represent *FROM* clauses with a bit more structure, the `ImplicitFromClause` and `ExplicitFromClause`
    subclasses exist and should generally be preffered over direct usage of the raw `From` clause class.

    Parameters
    ----------
    items : TableSource | Iterable[TableSource]
        The tables that should be sourced in the *FROM* clause

    Raises
    ------
    ValueError
        If no items are specified
    """

    @staticmethod
    def create_for(
        items: TableReference | Iterable[TableReference],
    ) -> ImplicitFromClause:
        """Shorthand method to create a *FROM* clause for a set of table references."""
        return ImplicitFromClause.create_for(items)

    def __init__(self, items: TableSource | Iterable[TableSource]) -> None:
        items = util.enlist(items)
        if not items:
            raise ValueError("At least one source is required")
        self._items: tuple[TableSource] = tuple(items)
        super().__init__(hash(self._items))

    __slots__ = ("_items",)
    __match_args__ = ("items",)

    @property
    def items(self) -> Sequence[TableType]:
        """Get the tables that are sourced in the *FROM* clause

        Returns
        -------
        Sequence[TableSource]
            The sources in exactly the sequence in which they were specified
        """
        return self._items

    def tables(self) -> set[TableReference]:
        return util.set_union(src.tables() for src in self._items)

    def columns(self) -> set[ColumnReference]:
        return util.set_union(src.columns() for src in self._items)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(src.iterexpressions() for src in self._items)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(src.itercolumns() for src in self._items)

    def predicates(self) -> Optional[QueryPredicates]:
        source_predicates = [src.predicates() for src in self._items]
        if not any(source_predicates):
            return None
        actual_predicates = [
            src_pred.root for src_pred in source_predicates if src_pred
        ]
        merged_predicate = CompoundPredicate.create_and(actual_predicates)
        return QueryPredicates(merged_predicate)

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_from_clause(self, *args, **kwargs)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[TableSource]:
        return iter(self._items)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._items == other._items

    def __str__(self) -> str:
        items_str = ", ".join(str(entry) for entry in self._items)
        return f"FROM {items_str}"


class ImplicitFromClause(From[DirectTableSource]):
    """Represents a special case of *FROM* clause that only allows for pure tables to be selected.

    Specifically, this means that subqueries or explicit joins using the *JOIN ON* syntax are not allowed. Just
    plain old ``SELECT ... FROM R, S, T WHERE ...`` queries.

    As a special case, all *FROM* clauses that consist of a single (non-subquery) table can be represented as
    implicit clauses.

    Parameters
    ----------
    tables : DirectTableSource | Iterable[DirectTableSource]
        The tables that should be selected
    """

    @staticmethod
    def create_for(
        tables: TableReference | Iterable[TableReference],
    ) -> ImplicitFromClause:
        """Shorthand method to create a *FROM* clause for a set of table references.

        This saves the user from creating the `DirectTableSource` instances before instantiating a implicit *FROM*
        clause.

        Parameters
        ----------
        tables : TableReference | Iterable[TableReference]
            The tables that should be sourced

        Returns
        -------
        ImplicitFromClause
            The *FROM* clause
        """
        tables = util.enlist(tables)
        return ImplicitFromClause([DirectTableSource(tab) for tab in tables])

    def __init__(self, tables: DirectTableSource | Iterable[DirectTableSource]):
        super().__init__(tables)

    __match_args__ = ("items",)

    def itertables(self) -> Sequence[TableReference]:
        """Provides all tables in the *FROM* clause exactly in the sequence in which they were specified.

        This utility saves the user from unwrapping all the `DirectTableSource` objects by herself.

        Returns
        -------
        Sequence[TableReference]
            The tables.
        """
        return [src.table for src in self.items]


class ExplicitFromClause(From[JoinTableSource]):
    """Represents a special kind of *FROM* clause that requires all tables to be joined using the *JOIN ON* syntax.

    Parameters
    ----------
    joins : JoinTableSource | Iterable[JoinTableSource]
        The joins that should be performed
    """

    def __init__(self, joins: JoinTableSource | Iterable[JoinTableSource]) -> None:
        if isinstance(joins, Iterable) and len(joins) != 1:
            raise ValueError("Explicit FROM clauses can only contain a single join")
        super().__init__(joins)

    __match_args__ = ("root",)

    @property
    def root(self) -> JoinTableSource:
        """Get the root join of the *FROM* clause.

        Returns
        -------
        JoinTableSource
            The root join
        """
        return self.items[0]

    def base_table(self) -> TableReference:
        """Get the table that is farthest to the left in the join chain.

        For subqueries or **VALUES** clauses, this will return the alias of the expression, i.e. the name of the virtual table
        that is created for the subquery or **VALUES** clause.

        Returns
        -------
        TableReference
            The table
        """
        return self.root.base_table()

    def iterpredicates(self) -> Iterable[AbstractPredicate]:
        """Provides all join conditions that are contained in the *FROM* clause.

        Returns
        -------
        Iterable[AbstractPredicate]
            The join conditions.
        """
        return util.flatten(join.predicates() for join in self.items)


class Where(BaseClause):
    """The *WHERE* clause specifies conditions that result rows must satisfy.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.

    Parameters
    ----------
    predicate : AbstractPredicate
        The root predicate that specifies all conditions
    """

    def __init__(self, predicate: AbstractPredicate) -> None:
        if not predicate:
            raise ValueError("Predicate must be set")
        self._predicate = predicate
        super().__init__(hash(predicate))

    __slots__ = ("_predicate",)
    __match_args__ = ("predicate",)

    @property
    def predicate(self) -> AbstractPredicate:
        """Get the root predicate that contains all filters and joins in the *WHERE* clause.

        Returns
        -------
        AbstractPredicate
            The condition
        """
        return self._predicate

    def tables(self) -> set[TableReference]:
        return self._predicate.tables()

    def columns(self) -> set[ColumnReference]:
        return self.predicate.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self.predicate.iterexpressions()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.predicate.itercolumns()

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_where_clause(self, *args, **kwargs)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.predicate == other.predicate

    def __str__(self) -> str:
        return f"WHERE {self.predicate}"


class GroupBy(BaseClause):
    """The *GROUP BY* clause combines rows that match a grouping criterion to enable aggregation on these groups.

    Despite their names, all grouped columns can be arbitrary `SqlExpression` instances, rules and restrictions of the SQL
    standard are not enforced by PostBOUND.

    Parameters
    ----------
    group_columns : Sequence[SqlExpression]
        The expressions that should be used to perform the grouping
    distinct : bool, optional
        Whether the grouping should perform duplicate elimination, by default *False*

    Raises
    ------
    ValueError
        If `group_columns` is empty.
    """

    @staticmethod
    def create_for(
        column: ColumnReference | Iterable[ColumnReference], *more_cols
    ) -> GroupBy:
        """Shortcut method to create a *GROUP BY* clause from column references."""
        column = [column] if isinstance(column, ColumnReference) else list(column)
        column.extend(more_cols)

        return GroupBy([ColumnExpression(col) for col in column])

    def __init__(
        self, group_columns: Sequence[SqlExpression], distinct: bool = False
    ) -> None:
        if not group_columns:
            raise ValueError("At least one group column must be specified")
        self._group_columns = tuple(group_columns)
        self._distinct = distinct

        hash_val = hash((self._group_columns, self._distinct))
        super().__init__(hash_val)

    __slots__ = ("_group_columns", "_distinct")
    __match_args__ = ("group_columns", "distinct")

    @property
    def group_columns(self) -> Sequence[SqlExpression]:
        """Get all expressions that should be used to determine the grouping.

        Returns
        -------
        Sequence[SqlExpression]
            The grouping expressions in exactly the sequence in which they were specified.
        """
        return self._group_columns

    @property
    def distinct(self) -> bool:
        """Get whether the grouping should eliminate duplicates.

        Returns
        -------
        bool
            Whether duplicate removal is performed.
        """
        return self._distinct

    def columns(self) -> set[ColumnReference]:
        return util.set_union(column.columns() for column in self.group_columns)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self.group_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(column.itercolumns() for column in self.group_columns)

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_groupby_clause(self, *args, **kwargs)

    def __len__(self) -> int:
        return len(self._group_columns)

    def __iter__(self) -> Iterator[SqlExpression]:
        return iter(self._group_columns)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.group_columns == other.group_columns
            and self.distinct == other.distinct
        )

    def __str__(self) -> str:
        columns_str = ", ".join(str(col) for col in self.group_columns)
        distinct_str = " DISTINCT" if self.distinct else ""
        return f"GROUP BY{distinct_str} {columns_str}"


class Having(BaseClause):
    """The *HAVING* clause enables filtering of the groups constructed by a *GROUP BY* clause.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.

    The structure of this clause is similar to the `Where` clause, but its scope is different (even though PostBOUND
    does no semantic validation to enforce this): predicates of the *HAVING* clause are only checked on entire groups
    of values and have to be valid their, instead of on individual tuples.

    Parameters
    ----------
    condition : AbstractPredicate
        The root predicate that contains all actual conditions
    """

    def __init__(self, condition: AbstractPredicate) -> None:
        if not condition:
            raise ValueError("Condition must be set")
        self._condition = condition
        super().__init__(hash(condition))

    __slots__ = ("_condition",)
    __match_args__ = ("condition",)

    @property
    def condition(self) -> AbstractPredicate:
        """Get the root predicate that is used to form the *HAVING* clause.

        Returns
        -------
        AbstractPredicate
            The condition
        """
        return self._condition

    def columns(self) -> set[ColumnReference]:
        return self.condition.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self.condition.iterexpressions()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.condition.itercolumns()

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_having_clause(self, *args, **kwargs)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.condition == other.condition

    def __str__(self) -> str:
        return f"HAVING {self.condition}"


class OrderByExpression:
    """The `OrderByExpression` is the fundamental building block for an *ORDER BY* clause.

    Each expression consists of the actual column (which might be an arbitrary `SqlExpression`, rules and restrictions
    by the SQL standard are not enforced here) as well as information regarding the ordering of the column. Setting
    this information to `None` falls back to the default interpretation by the target database system.

    Parameters
    ----------
    column : SqlExpression
        The column that should be used for ordering
    ascending : Optional[bool], optional
        Whether the column values should be sorted in ascending order. Defaults to *None*, which indicates that the
        system-default ordering should be used.
    nulls_first : Optional[bool], optional
        Whether *NULL* values should be placed at beginning or at the end of the sorted list. Defaults to *None*,
        which indicates that the system-default behaviour should be used.
    """

    @staticmethod
    def create_for(
        column: ColumnReference,
        *,
        ascending: Optional[bool] = None,
        nulls_first: Optional[bool] = None,
    ) -> OrderByExpression:
        """Shorthand method to create an `OrderByExpression` for a specific column reference."""
        return OrderByExpression(
            ColumnExpression(column),
            ascending=ascending,
            nulls_first=nulls_first,
        )

    def __init__(
        self,
        column: SqlExpression,
        ascending: Optional[bool] = None,
        nulls_first: Optional[bool] = None,
    ) -> None:
        if not column:
            raise ValueError("Column must be specified")
        self._column = column
        self._ascending = ascending
        self._nulls_first = nulls_first
        self._hash_val = hash((self._column, self._ascending, self._nulls_first))

    __slots__ = ("_column", "_ascending", "_nulls_first", "_hash_val")

    @property
    def column(self) -> SqlExpression:
        """Get the expression used to specify the current grouping.

        In the simplest case this can just be a `ColumnExpression` which sorts directly by the column values. More
        complicated constructs like mathematical expressions over the column values are also possible.

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    @property
    def ascending(self) -> Optional[bool]:
        """Get the desired ordering of the output rows.

        Returns
        -------
        Optional[bool]
            Whether to sort in ascending order. *None* indicates that the default behaviour of the system should be
            used.
        """
        return self._ascending

    @property
    def nulls_first(self) -> Optional[bool]:
        """Get where to place *NULL* values in the result set.

        Returns
        -------
        Optional[bool]
            Whether to put *NULL* values at the beginning of the result set (or at the end). *None* indicates that
            the default behaviour of the system should be used.
        """
        return self._nulls_first

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.column == other.column
            and self.ascending == other.ascending
            and self.nulls_first == other.nulls_first
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        ascending_str = (
            "" if self.ascending is None else (" ASC" if self.ascending else " DESC")
        )
        nulls_first = (
            ""
            if self.nulls_first is None
            else (" NULLS FIRST " if self.nulls_first else " NULLS LAST")
        )
        return f"{self.column}{ascending_str}{nulls_first}"


class OrderBy(BaseClause):
    """The *ORDER BY* clause specifies how result rows should be sorted.

    This clause has a similar structure like a `Select` clause and simply consists of an arbitrary number of
    `OrderByExpression` objects.

    Parameters
    ----------
    expressions : Iterable[OrderByExpression] | OrderByExpression
        The terms that should be used to determine the ordering. At least one expression is required

    Raises
    ------
    ValueError
        If no `expressions` are provided
    """

    @staticmethod
    def create_for(
        *columns: ColumnReference | Iterable[ColumnReference],
        ascending: Optional[bool] = None,
        nulls_first: Optional[bool] = None,
    ) -> OrderBy:
        """Shorthand method to create an *ORDER BY* clause for a set of column references.

        Additional ordering parameters are assigned to all columns.
        """
        columns = util.enlist(columns)
        columns = util.flatten(columns)
        return OrderBy(
            [
                OrderByExpression.create_for(
                    col, ascending=ascending, nulls_first=nulls_first
                )
                for col in columns
            ]
        )

    def __init__(
        self, expressions: Iterable[OrderByExpression] | OrderByExpression
    ) -> None:
        if not expressions:
            raise ValueError("At least one ORDER BY expression required")
        self._expressions = tuple(util.enlist(expressions))
        super().__init__(hash(self._expressions))

    __slots__ = ("_expressions",)
    __match_args__ = ("expressions",)

    @property
    def expressions(self) -> Sequence[OrderByExpression]:
        """Get the expressions that form this *ORDER BY* clause.

        Returns
        -------
        Sequence[OrderByExpression]
            The individual terms that make up the ordering in exactly the sequence in which they were specified (which
            is the only valid sequence since all other orders could change the ordering of the result set).
        """
        return self._expressions

    def columns(self) -> set[ColumnReference]:
        return util.set_union(
            expression.column.columns() for expression in self.expressions
        )

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [expression.column for expression in self.expressions]

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(
            expression.itercolumns() for expression in self.iterexpressions()
        )

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_orderby_clause(self)

    def __len__(self) -> int:
        return len(self._expressions)

    def __iter__(self) -> Iterator[OrderByExpression]:
        return iter(self._expressions)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.expressions == other.expressions

    def __str__(self) -> str:
        return "ORDER BY " + ", ".join(
            str(order_expr) for order_expr in self.expressions
        )


FetchDirection = Literal["first", "next", "prior", "last"]
"""Which values should be selected in a *LIMIT* / *FETCH* clause."""


class Limit(BaseClause):
    """The *FETCH FIRST* or *LIMIT* clause restricts the number of output rows returned by the database system.

    Each clause can specify an offset (which is probably only meaningful if there is also an *ORDER BY* clause)
    and the actual limit. Notice that although many database systems use a non-standard syntax for this clause, our
    implementation is modelled after the actual SQL standard version (i.e. it produces a *FETCH ...* string output).

    Parameters
    ----------
    limit : Optional[int], optional
        The maximum number of tuples to put in the result set. Defaults to *None* which indicates that all tuples
        should be returned.
    offset : Optional[int], optional
        The number of tuples that should be skipped from the beginning of the result set. If no `OrderBy` clause is
        defined, this makes the result set's contents non-deterministic (at least in theory). Defaults to *None*
        which indicates that no tuples should be skipped.

    Raises
    ------
    ValueError
        If neither a `limit`, nor an `offset` are specified
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        fetch_direction: FetchDirection = "first",
    ) -> None:
        if limit is None and offset is None:
            raise ValueError("Limit and offset cannot be both unspecified")
        self._limit = limit
        self._fetch_dir = fetch_direction
        self._offset = offset

        hash_val = hash((self._limit, self._offset, self._fetch_dir))
        super().__init__(hash_val)

    __slots__ = ("_limit", "_offset", "_fetch_dir")
    __match_args__ = ("limit", "offset")

    @property
    def limit(self) -> Optional[int]:
        """Get the maximum number of rows in the result set.

        Returns
        -------
        Optional[int]
            The limit or *None*, if all rows should be returned.
        """
        return self._limit

    @property
    def offset(self) -> Optional[int]:
        """Get the offset within the result set (i.e. number of first rows to skip).

        Returns
        -------
        Optional[int]
            The offset or *None* if no rows should be skipped.
        """
        return self._offset

    @property
    def fetch_direction(self) -> FetchDirection:
        """Get the direction of the limit (e.g. *first* or *prior*)."""
        return self._fetch_dir

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_limit_clause(self, *args, **kwargs)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.limit == other.limit
            and self.offset == other.offset
        )

    def __str__(self) -> str:
        offset_str = f"OFFSET {self.offset} ROWS" if self.offset is not None else ""
        fetch_direction = self.fetch_direction.upper()
        limit_str = (
            f"FETCH {fetch_direction} {self.limit} ROWS ONLY"
            if self.limit is not None
            else ""
        )
        if offset_str and limit_str:
            return offset_str + " " + limit_str
        elif offset_str:
            return offset_str
        elif limit_str:
            return limit_str
        return ""


class UnionClause(BaseClause):
    """The *UNION* or *UNION ALL* clause of a query.

    Parameters
    ----------
    left_query: SelectStatement
        The left input to the UNION operation. Since UNIONs are commutative, the assignment of left and right does not really
        matter.
    right_query: SelectStatement
        The right input to the UNION operation. Since UNIONs are commutative, the assignment of left and right does not really
        matter.
    union_all : bool, optional
        Whether the *UNION* operation should eliminate duplicates or not. Defaults to *False* which indicates that
        duplicates should be eliminated.
    """

    def __init__(
        self,
        left_query: SelectStatement,
        right_query: SelectStatement,
        *,
        union_all: bool = False,
    ) -> None:
        self._lhs = left_query
        self._rhs = right_query
        self._union_all = union_all
        hash_val = hash((self._lhs, self._rhs, self._union_all))
        super().__init__(hash_val)

    __slots__ = ("_lhs", "_rhs", "_union_all")
    __match_args__ = ("left_query", "right_query", "union_all")

    @property
    def left_query(self) -> SelectStatement:
        """Get the left query that is part of the *UNION* operation.

        Returns
        -------
        SelectStatement
            The left query. Since UNIONs are commutative, the assignment of left and right does not really matter.

        See Also
        --------
        input_queries() : Get both input queries
        """
        return self._lhs

    @property
    def right_query(self) -> SelectStatement:
        """Get the right query that is part of the *UNION* operation.

        Returns
        -------
        SelectStatement
            The right query. Since UNIONs are commutative, the assignment of left and right does not really matter.

        See Also
        --------
        input_queries() : Get both input queries
        """
        return self._rhs

    @property
    def union_all(self) -> bool:
        """Get whether this is a *UNION* or *UNION ALL* clause.

        Returns
        -------
        bool
            *True* if duplicates are eliminated, *False* otherwise.
        """
        return self._union_all

    def is_union_all(self) -> bool:
        """Whether this is a *UNION* or *UNION ALL* clause.

        Returns
        -------
        bool
            *True* if duplicates are eliminated, *False* otherwise.
        """
        return self._union_all

    def input_queries(self) -> set[SelectStatement]:
        """Get the two input queries that are part of the *UNION* operation.

        Returns
        -------
        set[SelectStatement]
            The left and right queries. Since UNIONs are commutative, the assignment of left and right does not really
            matter.
        """
        return {self._lhs, self._rhs}

    def tables(self) -> set[TableReference]:
        return self._lhs.tables() | self._rhs.tables()

    def columns(self) -> set[ColumnReference]:
        return self._lhs.columns() | self._rhs.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return list(self._lhs.iterexpressions()) + list(self._rhs.iterexpressions())

    def itercolumns(self) -> Iterable[ColumnReference]:
        return list(self._lhs.itercolumns()) + list(self._rhs.itercolumns())

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_union_clause(self, *args, **kwargs)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._lhs == other._lhs
            and self._rhs == other._rhs
            and self._union_all == other._union_all
        )

    def __str__(self) -> str:
        lhs_str = self._lhs.stringify(trailing_delimiter=False)
        rhs_str = self._rhs.stringify(trailing_delimiter=True)
        union_str = "UNION ALL" if self._union_all else "UNION"
        return f"{lhs_str} {union_str} {rhs_str}"


class ExceptClause(BaseClause):
    """The *EXCEPT* clause of a query.

    Parameters
    ----------
    left_query: SelectStatement
        The left query that is part of the *EXCEPT* operation. This is the result set from which tuples are removed.
    right_query: SelectStatement
        The right query that is part of the *EXCEPT* operation. This is the result set of the tuples that should be removed.
    """

    def __init__(
        self, left_query: SelectStatement, right_query: SelectStatement
    ) -> None:
        self._lhs = left_query
        self._rhs = right_query
        super().__init__(hash((self._lhs, self._rhs)))

    __slots__ = ("_lhs", "_rhs")
    __match_args__ = ("left_query", "right_query")

    @property
    def left_query(self) -> SelectStatement:
        """Get the left query that is part of the *EXCEPT* operation.

        The left query provides the result set from which tuples are removed.

        Returns
        -------
        SelectStatement
            The left query.
        """
        return self._lhs

    @property
    def right_query(self) -> SelectStatement:
        """Get the right query that is part of the *EXCEPT* operation.

        The right query provides the result set of the tuples that should be removed.

        Returns
        -------
        SelectStatement
            The right query.
        """
        return self._rhs

    def tables(self) -> set[TableReference]:
        return self._lhs.tables() | self._rhs.tables()

    def columns(self) -> set[ColumnReference]:
        return self._lhs.columns() | self._rhs.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return list(self._lhs.iterexpressions()) + list(self._rhs.iterexpressions())

    def itercolumns(self) -> Iterable[ColumnReference]:
        return list(self._lhs.itercolumns()) + list(self._rhs.itercolumns())

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_except_clause(self, *args, **kwargs)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._partner_query == other._partner_query
        )

    def __str__(self) -> str:
        lhs_str = self._lhs.stringify(trailing_delimiter=False)
        rhs_str = self._rhs.stringify(trailing_delimiter=True)
        return f"{lhs_str} EXCEPT {rhs_str}"


class IntersectClause(BaseClause):
    """The *INTERSECT* clause of a query.

    Parameters
    ----------
    left_query: SelectStatement
        The left query that is part of the *INTERSECT* operation. Since set intersection is commutative, the assignment
        of left and right does not really matter.
    right_query: SelectStatement
        The right query that is part of the *INTERSECT*. Since set intersection is commutative, the assignment of left
        and right does not really matter.
    """

    def __init__(
        self, left_query: SelectStatement, right_query: SelectStatement
    ) -> None:
        self._lhs = left_query
        self._rhs = right_query
        super().__init__(hash((self._lhs, self._rhs)))

    __slots__ = ("_lhs", "_rhs")
    __match_args__ = ("left_query", "right_query")

    @property
    def left_query(self) -> SelectStatement:
        """Get the left query that is part of the *INTERSECT* operation.

        Returns
        -------
        SelectStatement
            The left query. Since set intersection is commutative, the assignment of left and right does not really matter.

        See Also
        --------
        input_queries() : Get both input queries
        """
        return self._lhs

    @property
    def right_query(self) -> SelectStatement:
        """Get the right query that is part of the *INTERSECT* operation.

        Returns
        -------
        SelectStatement
            The right query. Since set intersection is commutative, the assignment of left and right does not really matter.

        See Also
        --------
        input_queries() : Get both input queries
        """
        return self._rhs

    def input_queries(self) -> set[SelectStatement]:
        """Get the two input queries that are part of the *INTERSECT* operation.

        Returns
        -------
        set[SelectStatement]
            The left and right queries. Since set intersection is commutative, the assignment of left and right does not
            really matter.
        """
        return {self._lhs, self._rhs}

    def tables(self) -> set[TableReference]:
        return self._lhs.tables() | self._rhs.tables()

    def columns(self) -> set[ColumnReference]:
        return self._lhs.columns() | self._rhs.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return list(self._lhs.iterexpressions()) + list(self._rhs.iterexpressions())

    def itercolumns(self) -> Iterable[ColumnReference]:
        return list(self._lhs.itercolumns()) + list(self._rhs.itercolumns())

    def accept_visitor(
        self, visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> VisitorResult:
        return visitor.visit_intersect_clause(self, *args, **kwargs)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._partner_query == other._partner_query
        )

    def __str__(self) -> str:
        lhs_str = self._lhs.stringify(trailing_delimiter=False)
        rhs_str = self._rhs.stringify(trailing_delimiter=True)
        return f"{lhs_str} INTERSECT {rhs_str}"


SetOperationClause = Union[UnionClause, ExceptClause, IntersectClause]
"""Supertype for all possible set operation clauses (**UNION**, **UNION ALL**, **INTERSECT**, **EXCEPT**)."""


class ClauseVisitor(abc.ABC, Generic[VisitorResult]):
    """Basic visitor to operate on arbitrary clause lists.

    See Also
    --------
    BaseClause

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_hint_clause(self, clause: Hint, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_explain_clause(self, clause: Explain, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_cte_clause(
        self, clause: CommonTableExpression, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_select_clause(self, clause: Select, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_from_clause(self, clause: From, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_where_clause(self, clause: Where, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_groupby_clause(self, clause: GroupBy, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_having_clause(self, clause: Having, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_orderby_clause(self, clause: OrderBy, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_limit_clause(self, clause: Limit, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_union_clause(self, clause: UnionClause, *args, **kwargs) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_except_clause(
        self, clause: ExceptClause, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_intersect_clause(
        self, clause: IntersectClause, *args, **kwargs
    ) -> VisitorResult:
        raise NotImplementedError


def _stringify_clause(clause: BaseClause) -> str:
    """Handler method to provide a refined string for a specific given clause, to be used by the `SqlQuery` ``__str__`` method.

    This method is slightly smarter than calling ``__str__`` directly, because it inserts newlines at sensible places in a
    query, e.g. after the hint block.

    Parameters
    ----------
    clause : BaseClause
        The clause to build the string representation

    Returns
    -------
    str
        The string representation of the clause
    """
    if isinstance(clause, Hint):
        return str(clause) + "\n"
    return str(clause) + " "


def collect_subqueries_in_expression(
    expression: SqlExpression,
) -> set[SqlQuery]:
    """Handler method to provide all the subqueries that are contained in a specific expression.

    Parameters
    ----------
    expression : SqlExpression
        The expression to analyze

    Returns
    -------
    set[SqlQuery]
        The subqueries from the `expression`

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(expression, SubqueryExpression):
        return {expression.query}
    return util.set_union(
        collect_subqueries_in_expression(child_expr)
        for child_expr in expression.iterchildren()
    )


def _collect_subqueries_in_table_source(
    table_source: TableSource,
) -> set[SqlQuery]:
    """Handler method to provide all subqueries that are contained in a specific table.

    This does not collect the subqueries in a recursive manner: once a subquery has been found, the collection stops.
    Therefore, subqueries that are contained in subqueries are not collected.

    Parameters
    ----------
    table_source : TableSource
        The table to analyze

    Returns
    -------
    set[SqlQuery]
        The subqueries that were found

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(table_source, SubqueryTableSource):
        return {table_source.query}
    elif isinstance(table_source, JoinTableSource):
        source_subqueries = _collect_subqueries_in_table_source(table_source.source)
        nested_subqueries = util.set_union(
            _collect_subqueries_in_table_source(nested_join)
            for nested_join in table_source.joined_table
        )
        condition_subqueries = (
            util.set_union(
                collect_subqueries_in_expression(cond_expr)
                for cond_expr in table_source.join_condition.iterexpressions()
            )
            if table_source.join_condition
            else set()
        )
        return source_subqueries | nested_subqueries | condition_subqueries
    else:
        return set()


def _collect_subqueries(clause: BaseClause) -> set[SqlQuery]:
    """Handler method to provide all the subqueries that are contained in a specific clause.

    Following the definitions of `SqlQuery.subqueries`, this completely ignores CTEs. Therefore, subqueries that are defined
    within CTEs are not detected.

    Parameters
    ----------
    clause : BaseClause
        The clause to check

    Returns
    -------
    set[SqlQuery]
        All subqueries that have been found in the clause.

    Raises
    ------
    ValueError
        If the given clause is unknown. This indicates that this method is missing a handler for a specific clause type that
        was added later on.

    See Also
    --------
    SqlQuery.subqueries
    """
    if (
        isinstance(clause, Hint)
        or isinstance(clause, Limit)
        or isinstance(clause, Explain)
    ):
        return set()

    if isinstance(clause, CommonTableExpression):
        return set()
    elif isinstance(clause, Select):
        return util.set_union(
            collect_subqueries_in_expression(target.expression)
            for target in clause.targets
        )
    elif isinstance(clause, ImplicitFromClause):
        return set()
    elif isinstance(clause, From):
        return util.set_union(
            _collect_subqueries_in_table_source(src) for src in clause.items
        )
    elif isinstance(clause, Where):
        where_predicate = clause.predicate
        return util.set_union(
            collect_subqueries_in_expression(expression)
            for expression in where_predicate.iterexpressions()
        )
    elif isinstance(clause, GroupBy):
        return util.set_union(
            collect_subqueries_in_expression(column) for column in clause.group_columns
        )
    elif isinstance(clause, Having):
        having_predicate = clause.condition
        return util.set_union(
            collect_subqueries_in_expression(expression)
            for expression in having_predicate.iterexpressions()
        )
    elif isinstance(clause, OrderBy):
        return util.set_union(
            collect_subqueries_in_expression(expression.column)
            for expression in clause.expressions
        )
    elif isinstance(clause, UnionClause):
        return clause.left_query.subqueries() | clause.right_query.subqueries()
    elif isinstance(clause, ExceptClause):
        return clause.left_query.subqueries() | clause.right_query.subqueries()
    elif isinstance(clause, IntersectClause):
        return clause.left_query.subqueries() | clause.right_query.subqueries()
    else:
        raise ValueError(f"Unknown clause type: {clause}")


def _collect_bound_tables_from_source(
    table_source: TableSource,
) -> set[TableReference]:
    """Handler method to provide all tables that are "produced" by a table source.

    "Produced" tables are tables that are either directly referenced in the *FROM* clause (e.g. ``FROM R``), referenced as
    part of joins (e.g. ``FROM R JOIN S ON ...``), or part of the *FROM* clauses of subqueries. In contrast, an unbound table
    is one that has to be provided by "context", such as the dependent table in a dependent subquery.

    Parameters
    ----------
    table_source : TableSource
        The table to check

    Returns
    -------
    set[TableReference]
        The "produced" tables.

    See Also
    --------
    SqlQuery.bound_tables
    """
    if isinstance(table_source, DirectTableSource):
        return {table_source.table}
    elif isinstance(table_source, SubqueryTableSource):
        return _collect_bound_tables(table_source.query.from_clause)
    elif isinstance(table_source, JoinTableSource):
        direct_tables = _collect_bound_tables_from_source(table_source.source)
        nested_tables = util.set_union(
            _collect_bound_tables_from_source(nested_join)
            for nested_join in table_source.joined_table
        )
        return direct_tables | nested_tables


def _collect_bound_tables(from_clause: From) -> set[TableReference]:
    """Handler method to provide all tables that are "produced" in the given clause.

    "Produced" tables are tables that are either directly referenced in the *FROM* clause (e.g. ``FROM R``), referenced as
    part of joins (e.g. ``FROM R JOIN S ON ...``), or part of the *FROM* clauses of subqueries. In contrast, an unbound table
    is one that has to be provided by "context", such as the dependent table in a dependent subquery.

    Parameters
    ----------
    from_clause : From
        The clause to check

    Returns
    -------
    set[TableReference]
        The "produced" tables

    See Also
    --------
    SqlQuery.bound_tables
    """
    if isinstance(from_clause, ImplicitFromClause):
        return from_clause.tables()
    else:
        return util.set_union(
            _collect_bound_tables_from_source(src) for src in from_clause.items
        )


FromClauseType = TypeVar("FromClauseType", bound=From)


def _create_ast(item: Any, *, indentation: int = 0) -> str:
    """Helper method to generate a pretty representation of the logical internal structure of a query."""
    prefix = " " * indentation
    item_str = type(item).__name__
    match item:
        # Predicates
        case CompoundPredicate() if not item.is_negation():
            children = [
                _create_ast(c, indentation=indentation + 2) for c in item.children
            ]
            child_str = "\n".join(children)
            item_str = f"{item_str} [{item.operation.value}]"
            return f"{prefix}+-{item_str}\n{child_str}"
        case CompoundPredicate() if item.is_negation():
            child = _create_ast(item.children, indentation=indentation + 2)
            item_str = f"{item_str} [NOT]"
            return f"{prefix}+-{item_str}\n{child}"
        case UnaryPredicate():
            child = _create_ast(item.column, indentation=indentation + 2)
            item_str = (
                f"{item_str} [{item.operation.value}]" if item.operation else item_str
            )
            return f"{prefix}+-{item_str}\n{child}"
        case BinaryPredicate():
            lhs = _create_ast(item.first_argument, indentation=indentation + 2)
            rhs = _create_ast(item.second_argument, indentation=indentation + 2)
            item_str = f"{item_str} [{item.operation.value}]"
            return f"{prefix}+-{item_str}\n{lhs}\n{rhs}"
        case AbstractPredicate():
            expressions = [
                _create_ast(e, indentation=indentation + 2)
                for e in item.iterexpressions()
            ]
            expression_str = "\n".join(expressions)
            return f"{prefix}+-{item_str}\n{expression_str}"

        # Expressions
        case ColumnExpression():
            return f"{prefix}+ {item_str} [{item.column}]"
        case CastExpression():
            child = _create_ast(item.casted_expression, indentation=indentation + 2)
            target_type = (
                f"{item.target_type}[]" if item.array_type else item.target_type
            )
            return f"{prefix}+-{item_str} [{target_type}]\n{child}"
        case QuantifierExpression():
            child = _create_ast(item.expression, indentation=indentation + 2)
            return f"{prefix}+-{item_str} [{item.quantifier.value}]\n{child}"
        case FunctionExpression():
            arguments = [
                _create_ast(arg, indentation=indentation + 2) for arg in item.arguments
            ]
            argument_str = "\n".join(arguments)
            item_str = f"{item_str} [{item.function}]"
            return f"{prefix}+-{item_str}\n{argument_str}"
        case SqlExpression():
            # NB: This has to be the last expression/predicate handler since they all inherit from SqlExpression
            expressions = [
                _create_ast(e, indentation=indentation + 2) for e in item.iterchildren()
            ]
            expression_str = "\n".join(expressions)
            return (
                f"{prefix}+-{item_str}\n{expression_str}"
                if expressions
                else f"{prefix}+-{item_str} [{item}]"
            )

        # Clauses
        case Where() | Having():
            predicate_str = _create_ast(item.predicate, indentation=indentation + 2)
            return f"{prefix}+-{item_str}\n{predicate_str}"
        case DirectTableSource():
            return f"{prefix}+-{item_str} [{item.table}]"
        case JoinTableSource():
            left_str = _create_ast(item.left, indentation=indentation + 2)
            right_str = _create_ast(item.right, indentation=indentation + 2)
            return f"{prefix}+-{item_str}\n{left_str}\n{right_str}"
        case SubqueryTableSource():
            subquery_str = _create_ast(item.query, indentation=indentation + 2)
            if item.target_name:
                item_str = f"{item_str} [{item.target_name}]"
            return f"{prefix}+-{item_str}\n{subquery_str}"
        case ValuesTableSource():
            return f"{prefix}+-{item_str}"
        case From():
            tables = [_create_ast(t, indentation=indentation + 2) for t in item.items]
            table_str = "\n".join(tables)
            return f"{prefix}+-{item_str}\n{table_str}"
        case CommonTableExpression():
            ctes = [_create_ast(c, indentation=indentation + 2) for c in item.queries]
            cte_str = "\n".join(ctes)
            return f"{prefix}+-{item_str}\n{cte_str}"
        case BaseClause():
            expressions = [
                _create_ast(e, indentation=indentation + 2)
                for e in item.iterexpressions()
            ]
            expression_str = "\n".join(expressions)
            return f"{prefix}+-{item_str}\n{expression_str}"
        case ValuesWithQuery():
            return f"{prefix}+-{item_str}"
        case WithQuery():
            child_expression = _create_ast(item.query, indentation=indentation + 2)
            item_str = f"{item_str} [{item.target_name}]"
            return f"{prefix}+-{item_str}\n{child_expression}"
        case SetQuery():
            subqueries = [
                _create_ast(q, indentation=indentation + 2)
                for q in (item.left_query, item.right_query)
            ]
            subquery_str = "\n".join(subqueries)
            return f"{prefix}+-{item_str}\n{subquery_str}"
        case SqlQuery():
            clauses = [
                _create_ast(c, indentation=indentation + 2) for c in item.clauses()
            ]
            clause_str = "\n".join(clauses)
            return f"{prefix}+-{item_str}\n{clause_str}"
        case _:
            raise ValueError(f"Unknown item type '{type(item)}': {item}")


class SqlQuery:
    """Represents a plain *SELECT* query, providing direct access to the different clauses in the query.

    At a basic level, PostBOUND differentiates between two types of queries:

    - implicit SQL queries specify all referenced tables in the *FROM* clause and the join predicates in the *WHERE*
      clause, e.g. ``SELECT * FROM R, S WHERE R.a = S.b AND R.c = 42``. This is the traditional way of writing SQL queries.
    - explicit SQL queries use the *JOIN ON* syntax to reference tables, e.g.
      ``SELECT * FROM R JOIN S ON R.a = S.b WHERE R.c = 42``. This is the more "modern" way of writing SQL queries.

    There is also a third possibility of mixing the implicit and explicit syntax. For each of these cases, designated
    subclasses exist. They all provide the same functionality and only differ in the (sub-)types of their *FROM* clauses.
    Therefore, these classes can be considered as "marker" types to indicate that at a certain point of a computation, a
    specific kind of query is required. The `SqlQuery` class acts as a superclass that specifies the general behaviour of all
    query instances and can act as the most general type of query.

    To represent other types of SQL statements (e.g. DML statements), different classes have to be used. Notably, this also
    applies to set queries, i.e. queries containing *UNION*, *INTERSECT*, or *EXCEPT* clauses. These are represented by
    the `SetQuery` class. The reason for this distinction is a pragmatic one: most research in query optimization is currently
    concerned with single **SELECT** queries and the interface for a set query has to be quite different from that of an
    ordinary query. For example, there is no obvious way how to represent the predicates of a query with an **EXCEPT** clause.
    Therefore, optimizers that provide support for set queries have to explicitly state this in their interface.
    At the same time, pretty much all of PostBOUND's code that uses queries operates on features that are common to both
    `SqlQuery` as well as `SetQuery` objects. Therefore, set queries can be passed even though the interface only specifies
    `SqlQuery` objects. This is just because set queries are a late addition to PostBOUND and simply do not have the time to
    re-visit all other method definitions to update the their signatures.

    If you want to explicitly communicate that some method accepts both plain SQL queries as well as set queries, you can use
    the `SelectStatement` super type.

    The clauses of each query can be accessed via properties. If a clause is optional, the absence of the clause is indicated
    through a *None* value. All additional behaviour of the queries is provided by the different methods. These are mostly
    focused on an easy introspection of the query's structure.

    Notice that PostBOUND does not enforce any semantics on the queries (e.g. regarding data types, access to values, the
    cardinality of subquery results, or the connection between different clauses). This has to be done by the user, or by the
    actual database system.

    Limitations
    -----------

    While the query abstraction is quite powerful, it is cannot represent the full range of SQL statements. Noteworthy
    limitations include:

    - no DDL or DML statements. The query abstraction is really only focused on *queries*, i.e. *SELECT*
      statements.
    - no recursive CTEs. While plain CTEs are supported, recursive CTEs are not. While this would be an easy addition, there
      simply was no need for it so far. If you need recursive CTEs, PRs are always welcome!
    - no support for GROUPING SETS, including CUBE() and ROLLUP(). Conceptually speaking, these would not be hard to add, but
      there simply was no need for them so far. If you need them, PRs are always welcome!

    Parameters
    ----------
    select_clause : Select
        The *SELECT* part of the query. This is the only required part of a query. Notice however, that some database systems
        do not allow queries without a *FROM* clause.
    from_clause : Optional[From], optional
        The *FROM* part of the query, by default *None*
    where_clause : Optional[Where], optional
        The *WHERE* part of the query, by default *None*
    groupby_clause : Optional[GroupBy], optional
        The *GROUP BY* part of the query, by default *None*
    having_clause : Optional[Having], optional
        The *HAVING* part of the query, by default *None*.
    orderby_clause : Optional[OrderBy], optional
        The *ORDER BY* part of the query, by default *None*
    limit_clause : Optional[Limit], optional
        The *LIMIT* and *OFFSET* part of the query. In standard SQL, this is designated using the *FETCH FIRST* syntax.
        Defaults to *None*.
    cte_clause : Optional[CommonTableExpression], optional
        The *WITH* part of the query, by default *None*
    hints : Optional[Hint], optional
        The hint block of the query. Hints are not part of standard SQL and follow a completely system-specific syntax. Even
        their placement in within the query varies from system to system and from extension to extension. Defaults to *None*.
    explain : Optional[Explain], optional
        The *EXPLAIN* part of the query. Like hints, this is not part of standard SQL. However, most systems provide
        *EXPLAIN* functionality. The specific features and syntax are quite similar, but still system specific. Defaults to
        *None*.

    Warnings
    --------
    See the `Limitations` section for unsupported SQL features.
    """

    def __init__(
        self,
        *,
        select_clause: Select,
        from_clause: Optional[From] = None,
        where_clause: Optional[Where] = None,
        groupby_clause: Optional[GroupBy] = None,
        having_clause: Optional[Having] = None,
        orderby_clause: Optional[OrderBy] = None,
        limit_clause: Optional[Limit] = None,
        cte_clause: Optional[CommonTableExpression] = None,
        hints: Optional[Hint] = None,
        explain: Optional[Explain] = None,
    ) -> None:
        self._cte_clause = cte_clause
        self._select_clause = select_clause
        self._from_clause = from_clause
        self._where_clause = where_clause
        self._groupby_clause = groupby_clause
        self._having_clause = having_clause
        self._orderby_clause = orderby_clause
        self._limit_clause = limit_clause
        self._hints = hints
        self._explain = explain

        self._query_predicates: QueryPredicates | None = None

        self._hash_val = hash(
            (
                self._hints,
                self._explain,
                self._cte_clause,
                self._select_clause,
                self._from_clause,
                self._where_clause,
                self._groupby_clause,
                self._having_clause,
                self._orderby_clause,
                self._limit_clause,
            )
        )

    __slots__ = (
        "_cte_clause",
        "_select_clause",
        "_from_clause",
        "_where_clause",
        "_groupby_clause",
        "_having_clause",
        "_orderby_clause",
        "_limit_clause",
        "_hints",
        "_explain",
        "_query_predicates",
        "_hash_val",
    )

    @property
    def cte_clause(self) -> Optional[CommonTableExpression]:
        """Get the *WITH* clause of the query.

        Returns
        -------
        Optional[CommonTableExpression]
            The *WITH* clause if it was specified, or *None* otherwise.
        """
        return self._cte_clause

    @property
    def select_clause(self) -> Select:
        """Get the *SELECT* clause of the query. Will always be set.

        Returns
        -------
        Select
            The *SELECT* clause
        """
        return self._select_clause

    @property
    def from_clause(self) -> Optional[From]:
        """Get the *FROM* clause of the query.

        Returns
        -------
        Optional[From]
            The *FROM* clause if it was specified, or *None* otherwise.
        """
        return self._from_clause

    @property
    def where_clause(self) -> Optional[Where]:
        """Get the *WHERE* clause of the query.

        Returns
        -------
        Optional[Where]
            The *WHERE* clause if it was specified, or *None* otherwise.
        """
        return self._where_clause

    @property
    def groupby_clause(self) -> Optional[GroupBy]:
        """Get the *GROUP BY* clause of the query.

        Returns
        -------
        Optional[GroupBy]
            The *GROUP BY* clause if it was specified, or *None* otherwise.
        """
        return self._groupby_clause

    @property
    def having_clause(self) -> Optional[Having]:
        """Get the *HAVING* clause of the query.

        Returns
        -------
        Optional[Having]
            The *HAVING* clause if it was specified, or *None* otherwise.
        """
        return self._having_clause

    @property
    def orderby_clause(self) -> Optional[OrderBy]:
        """Get the *ORDER BY* clause of the query.

        Returns
        -------
        Optional[OrderBy]
            The *ORDER BY* clause if it was specified, or *None* otherwise.
        """
        return self._orderby_clause

    @property
    def limit_clause(self) -> Optional[Limit]:
        """Get the combined *LIMIT* and *OFFSET* clauses of the query.

        According to the SQL standard, these clauses should use the *FETCH FIRST* syntax. However, many systems use
        *OFFSET* and *LIMIT* instead.

        Returns
        -------
        Optional[Limit]
            The *FETCH FIRST* clause if it was specified, or *None* otherwise.
        """
        return self._limit_clause

    @property
    def hints(self) -> Optional[Hint]:
        """Get the hint block of the query.

        The hints can specify preparatory statements that have to be executed before the actual query is run in addition to the
        hints themselves.

        Returns
        -------
        Optional[Hint]
            The hint block if it was specified, or *None* otherwise.
        """
        return self._hints

    @property
    def explain(self) -> Optional[Explain]:
        """Get the *EXPLAIN* block of the query.

        Returns
        -------
        Optional[Explain]
            The *EXPLAIN* settings if specified, or *None* otherwise.
        """
        return self._explain

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        """Checks, whether this query has an implicit *FROM* clause.

        The implicit *FROM* clause only consists of the source tables that should be scanned for the query. No subqueries or
        joins are contained in the clause. All join predicates must be part of the *WHERE* clause.

        Returns
        -------
        bool
            Whether the query is implicit

        See Also
        --------
        ImplicitSqlQuery
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_explicit(self) -> bool:
        """Checks, whether this query has an explicit *FROM* clause.

        The explicit *FROM* clause exclusively makes use of the *JOIN ON* syntax to denote both the tables that should be
        scanned, and the predicates that should be used to join the tables together. Therefore, the *WHERE* clause should
        only consist of filter predicates on the base tables. However, this is not enforced and the contents of the *ON*
        conditions as well as the *WHERE* clause can be arbitrary predicates.

        Returns
        -------
        bool
            Whether the query is explicit

        See Also
        --------
        ExplicitSqlQuery
        """
        raise NotImplementedError

    def is_explain(self) -> bool:
        """Checks, whether this query is an *EXPLAIN* query rather than a normal SQL query.

        An *EXPLAIN* query is not executed like a normal ``SELECT ...`` query. Instead of actually calculating a result set,
        the database system only provides a query plan. This plan is the execution plan that would be used, had the query been
        entered as a normal SQL query.

        Returns
        -------
        bool
            Whether this query should be explained, rather than executed.
        """
        return self.explain is not None

    @functools.cache
    def tables(self) -> set[TableReference]:
        """Provides all tables that are referenced at any point in the query.

        This includes tables from all clauses. Virtual tables will be included and tables that are only scanned within
        subqueries are included as well. Notice however, that some database systems might not support subqueries to be put
        at arbitrary positions in the query (e.g. *GROUP BY* clause).

        Returns
        -------
        set[TableReference]
            All tables that are referenced in the query.
        """
        relevant_clauses: list[BaseClause] = [
            self._select_clause,
            self._from_clause,
            self._where_clause,
            self._groupby_clause,
            self._having_clause,
            self._orderby_clause,
            self._limit_clause,
        ]

        tabs = set()
        tabs |= self.cte_clause.referenced_tables() if self.cte_clause else set()
        for clause in relevant_clauses:
            if clause is None:
                continue
            tabs |= clause.tables()

        return tabs

    def output_columns(self) -> Sequence[ColumnReference]:
        """Provides the columns that form the result relation of this query.

        Columns are ordered according to their appearance in the *SELECT* clause and will not have a bound table associated
        with them. This is because the query result is "anonymous" and does not have a relation name associated with it.
        The columns are named according to the following rules:

        - If the expression has an alias, this name is used
        - If the expression is a simple column reference, the column name is used
        - Otherwise, a generic name is used, e.g. "column_1", "column_2", etc.

        Returns
        -------
        Sequence[ColumnReference]
            The columns. Their order matches the order in which they appear in the *SELECT* clause of the query.
        """
        cols: list[ColumnReference] = []
        anon_idx = 1

        for projection in self._select_clause:
            if projection.target_name:
                cols.append(ColumnReference(projection.target_name))
                continue

            match projection.expression:
                case ColumnExpression(column):
                    cols.append(column.as_unbound())
                case StarExpression():
                    warnings.warn(
                        "Cannot compute the output columns for SELECT * queries. Please use the database schema "
                        "to infer the columns. The result of this method is likely not what you want.",
                        stacklevel=2,
                    )
                case _:
                    cols.append(ColumnReference(f"column_{anon_idx}"))
                    anon_idx += 1

        return cols

    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced at any point in the query.

        This includes columns from all clauses and does not account for renamed columns from subqueries. For example, consider
        the query ``SELECT R.a, my_sq.b FROM R JOIN (SELECT b FROM S) my_sq ON R.a < my_sq.b``. `columns` would return the
        following set: ``{R.a, S.b, my_sq.b}``, even though ``my_sq.b`` can be considered as just an alias for ``S.b``.

        Returns
        -------
        set[ColumnReference]
            All columns that are referenced in the query.
        """
        return util.set_union(clause.columns() for clause in self.clauses())

    def columns_of(self, table: TableReference) -> set[ColumnReference]:
        """Provides all columns of a specific table that are referenced at any point in the query.

        Parameters
        ----------
        table : TableReference
            The table to which the columns should belong

        Returns
        -------
        set[ColumnReference]
            All columns of the specified table that are referenced in the query.

        See Also
        --------
        columns : Get all columns that are referenced in the query
        """
        return {col for col in self.columns() if col.belongs_to(table)}

    def predicates(self) -> QueryPredicates:
        """Provides all predicates in this query.

        *All* predicates really means *all* predicates: this includes predicates that appear in the *FROM* clause, the
        *WHERE* clause, as well as any predicates from CTEs.

        Returns
        -------
        QueryPredicates
            A predicates wrapper around the conjunction of all individual predicates.
        """
        if self._query_predicates is not None:
            return self._query_predicates

        current_predicate = QueryPredicates.empty_predicate()

        if self.cte_clause:
            for with_query in self.cte_clause.queries:
                current_predicate = current_predicate.and_(
                    with_query.query.predicates()
                )

        if self.where_clause:
            current_predicate = current_predicate.and_(self.where_clause.predicate)

        from_predicates = self.from_clause.predicates()
        if from_predicates:
            current_predicate = current_predicate.and_(from_predicates)

        self._query_predicates = current_predicate
        return current_predicate

    def filters(self) -> Collection[AbstractPredicate]:
        """Alias for `predicates().filters()`.

        See Also
        --------
        QueryPredicates.filters
        """
        return self.predicates().filters()

    def joins(self) -> Collection[AbstractPredicate]:
        """Alias for `predicates().joins()`.

        See Also
        --------
        QueryPredicates.joins
        """
        return self.predicates().joins()

    def join_graph(self, *, merge_aliases: bool = False) -> nx.Graph:
        """Alias for `predicates().join_graph()`.

        See Also
        --------
        QueryPredicates.join_graph
        """
        return self.predicates().join_graph(merge_aliases=merge_aliases)

    def filters_for(self, table: TableReference) -> Optional[AbstractPredicate]:
        """Alias for `predicates().filters_for(table)`.

        See Also
        --------
        QueryPredicates.filters_for
        """
        return self.predicates().filters_for(table)

    def joins_for(self, table: TableReference) -> Collection[AbstractPredicate]:
        """Alias for `predicates().joins_for(table)`.

        See Also
        --------
        QueryPredicates.joins_for
        """
        return self.predicates().joins_for(table)

    def joins_between(
        self,
        table1: TableReference | Iterable[TableReference],
        table2: TableReference | Iterable[TableReference],
    ) -> Optional[AbstractPredicate]:
        """Alias for `predicates().joins_between(table1, table2)`.

        See Also
        --------
        QueryPredicates.joins_between
        """
        return self.predicates().joins_between(table1, table2)

    def joins_tables(
        self,
        tables: TableReference | Iterable[TableReference],
        *more_tables: TableReference,
    ) -> bool:
        """Alias for `predicates().joins_tables(tables, *more_tables)`.

        See Also
        --------
        QueryPredicates.joins_tables
        """
        return self.predicates().joins_tables(tables, *more_tables)

    def subqueries(self) -> Collection[SqlQuery]:
        """Provides all subqueries that are referenced in this query.

        Notice that CTEs are ignored by this method, since they can be accessed directly via the `cte_clause` property.

        Returns
        -------
        Collection[SqlQuery]
            All subqueries that appear in any of the "inner" clauses of the query
        """
        # the implementation of subqueries() on SetQuery relies on this being a set, both methods should be changed together
        return util.set_union(_collect_subqueries(clause) for clause in self.clauses())

    def clauses(
        self, *, skip: Optional[Type | Iterable[Type]] = NoneType
    ) -> Sequence[BaseClause]:
        """Provides all the clauses that are defined (i.e. not *None*) in this query.

        Parameters
        ----------
        skip : Optional[Type | Iterable[Type]], optional
            The clause types that should be skipped in the output. This can be a single type or an iterable of types.

        Returns
        -------
        Sequence[BaseClause]
            The clauses. The current order of the clauses is as follows: hints, explain, cte, select, from, where, group by,
            having, order by, limit. Notice however, that this order is not strictly standardized and may change in the future.
            All clauses that are not specified on the query will be skipped.
        """
        all_clauses = [
            self.hints,
            self.explain,
            self.cte_clause,
            self.select_clause,
            self.from_clause,
            self.where_clause,
            self.groupby_clause,
            self.having_clause,
            self.orderby_clause,
            self.limit_clause,
        ]
        return [
            clause
            for clause in all_clauses
            if clause is not None and not isinstance(clause, skip)
        ]

    def bound_tables(self) -> set[TableReference]:
        """Provides all tables that can be assigned to a physical or virtual table reference in this query.

        Bound tables are those tables, that are selected in the *FROM* clause of the query, or a subquery. Conversely,
        unbound tables are those that have to be "injected" by an outer query, as is the case for dependent subqueries.

        For example, the query ``SELECT * FROM R, S WHERE R.a = S.b`` has two bound tables: *R* and *S*.
        On the other hand, the query ``SELECT * FROM R WHERE R.a = S.b`` has only bound *R*, whereas *S* has to be bound in
        a surrounding query.

        Returns
        -------
        set[TableReference]
            All tables that are bound (i.e. listed in the *FROM* clause or a CTE) of the query.
        """
        subquery_produced_tables = util.set_union(
            subquery.bound_tables() for subquery in self.subqueries()
        )
        cte_produced_tables = self.cte_clause.tables() if self.cte_clause else set()
        own_produced_tables = _collect_bound_tables(self.from_clause)
        return own_produced_tables | subquery_produced_tables | cte_produced_tables

    def unbound_tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in this query but not bound.

        While `tables()` provides all tables that are referenced in this query in any way, `bound_tables` restricts
        these tables. This method provides the complementary set to `bound_tables` i.e.
        ``tables = bound_tables ⊕ unbound_tables``.

        Returns
        -------
        set[TableReference]
            The unbound tables that have to be supplied as part of an outer query
        """
        if self.from_clause:
            virtual_subquery_targets = {
                subquery_source.target_table
                for subquery_source in self.from_clause.items
                if isinstance(subquery_source, SubqueryTableSource)
            }
        else:
            virtual_subquery_targets = set()

        if self.cte_clause:
            virtual_cte_targets = {
                with_query.target_table for with_query in self.cte_clause.queries
            }
        else:
            virtual_cte_targets = set()

        return (
            self.tables()
            - self.bound_tables()
            - virtual_subquery_targets
            - virtual_cte_targets
        )

    def is_ordered(self) -> bool:
        """Checks, whether this query produces its result tuples in order.

        Returns
        -------
        bool
            Whether a valid *ORDER BY* clause was specified on the query.
        """
        return self.orderby_clause is not None

    def is_dependent(self) -> bool:
        """Checks, whether all columns that are referenced in this query are provided by the tables from this query.

        In order for this check to work, all columns have to be bound to actual tables, i.e. the `tables` attribute of all
        column references have to be set to a valid object.

        Returns
        -------
        bool
            Whether all columns belong to tables that are bound by this query
        """
        return not (self.tables() <= self.bound_tables())

    def is_scalar(self) -> bool:
        """Checks, whether the query is guaranteed to provide a single scalar value as a result.

        Scalar results can only be calculated by queries with a single projection in the *SELECT* clause and if that projection
        is an aggregate function, e.g. *SELECT min(R.a) FROM R*. However, there are other queries which could also be scalar
        "by chance", e.g. *SELECT R.b FROM R WHERE R.a = 1*  if *R.a* is the primary key of *R*. Notice that such cases are not
        recognized by this method.

        Returns
        -------
        bool
            Whether the query will always return a single scalar value
        """
        if not len(self.select_clause.targets) == 1 or self.select_clause.is_distinct():
            return False
        target: SqlExpression = util.simplify(self.select_clause.targets).expression
        return (
            isinstance(target, FunctionExpression)
            and target.is_aggregate()
            and not self._groupby_clause
        )

    def is_set_query(self) -> bool:
        """Checks, whether this query is a set query.

        A set query is a query that combines the results of two or more queries into a single result set. This can be done
        by combining the tuples from both sets using a *UNION* clause (which removes duplicates), or a *UNION ALL* clause
        (which retains duplicates). Alternatively, only tuples that are present in both sets can be retained using an
        *INTERSECT* clause. Finally, all tuples from the first result set that are not part of the second result set can be
        computed using an *EXCEPT* clause.

        Notice that only one of the set operators can be used at a time, but the input query of one set operation can itself
        use another set operation.

        Returns
        -------
        bool
            Whether this query is a set query
        """
        return False

    def contains_cross_product(self) -> bool:
        """Checks, whether this query has at least one cross product.

        Returns
        -------
        bool
            Whether this query has cross products.
        """
        if not self._from_clause:
            return False
        if self.predicates().empty_predicate():
            return True
        join_graph = self.predicates().join_graph()
        return len(nx.connected_components(join_graph)) > 1

    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all expressions that are directly contained in this query.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        return util.flatten(clause.iterexpressions() for clause in self.clauses())

    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides access to all column in this query.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n* times
        in this query, it will also be returned *n* times by this method. Furthermore, the order in which columns are provided
        by the iterable matches the order in which they appear in this query.

        Returns
        -------
        Iterable[ColumnReference]
            The columns
        """
        return util.flatten(clause.itercolumns() for clause in self.clauses())

    def stringify(self, *, trailing_delimiter: bool = True) -> str:
        """Provides a string representation of this query.

        The only difference to calling `str` directly, is that the `stringify` method provides control over whether a trailing
        delimiter should be appended to the query.

        Parameters
        ----------
        trailing_delimiter : bool, optional
            Whether a delimiter should be appended to the query. Defaults to *True*.

        Returns
        -------
        str
            A string representation of this query
        """
        delim = ";" if trailing_delimiter else ""
        return (
            "".join(_stringify_clause(clause) for clause in self.clauses()).rstrip()
            + delim
        )

    def ast(self) -> str:
        """Provides a human-readable representation of the abstract syntax tree for this query.

        The AST is a textual representation of the query that shows the structure of the query in a tree-like manner.

        Returns
        -------
        str
            The abstract syntax tree of this query
        """
        return _create_ast(self)

    def accept_visitor(
        self, clause_visitor: ClauseVisitor[VisitorResult], *args, **kwargs
    ) -> dict[BaseClause, VisitorResult]:
        """Applies a visitor over all clauses in the current query.

        Notice that since the visitor is applied to all clauses, it returns the results for each of them.

        Parameters
        ----------
        clause_visitor : ClauseVisitor
            The visitor algorithm to use.
        """
        return {
            clause: clause.accept_visitor(clause_visitor, *args, **kwargs)
            for clause in self.clauses()
        }

    def __json__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.clauses() == other.clauses()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.stringify(trailing_delimiter=True)


class ImplicitSqlQuery(SqlQuery):
    """An implicit query restricts the constructs that may appear in the *FROM* clause.

    For implicit queries, the *FROM* clause may only consist of simple table sources. All join conditions have to be put in
    the *WHERE* clause. Notice that this does not restrict the structure of other clauses. For example, the *WHERE* clause
    can still contain subqueries. As a special case, queries without a *FROM* clause are also considered implicit.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    See Also
    --------
    ImplicitFromClause
    ExplicitSqlQuery

    Examples
    --------
    The following queries are considered as implicit queries:

    .. code-block:: sql

        SELECT *
        FROM R, S, T
        WHERE R.a = S.b
            AND S.b = T.c
            AND R.a < 42

    .. code-block:: sql

        SELECT *
        FROM R, S, T
        WHERE R.a = S.b
            AND S.b = T.c
            AND R.a = (SELECT MIN(R.a) FROM R)
    """

    def __init__(
        self,
        *,
        select_clause: Select,
        from_clause: Optional[ImplicitFromClause] = None,
        where_clause: Optional[Where] = None,
        groupby_clause: Optional[GroupBy] = None,
        having_clause: Optional[Having] = None,
        orderby_clause: Optional[OrderBy] = None,
        limit_clause: Optional[Limit] = None,
        cte_clause: Optional[CommonTableExpression] = None,
        explain_clause: Optional[Explain] = None,
        hints: Optional[Hint] = None,
    ) -> None:
        super().__init__(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
            groupby_clause=groupby_clause,
            having_clause=having_clause,
            orderby_clause=orderby_clause,
            limit_clause=limit_clause,
            cte_clause=cte_clause,
            explain=explain_clause,
            hints=hints,
        )

    @property
    def from_clause(self) -> Optional[ImplicitFromClause]:
        return self._from_clause

    def is_implicit(self) -> bool:
        return True

    def is_explicit(self) -> bool:
        return False


class ExplicitSqlQuery(SqlQuery):
    """An explicit query restricts the constructs that may appear in the *FROM* clause.

    For explicit queries, the *FROM* clause must utilize the *JOIN ON* syntax for all tables. The join conditions should
    be put into the *ON* blocks. Notice however, that PostBOUND does not perform any sanity checks here. Therefore, it is
    possible to put mix joins and filters in the *ON* blocks, move all joins to the *WHERE* clause or scatter the join
    conditions between the two clauses. Whether this is good style is up for debate, but at least PostBOUND does allow it. In
    contrast to the implicit query, subqueries are also allowed as table sources.

    Notice that each explicit query must join at least two tables in its *FROM* clause.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    See Also
    --------
    ExplicitFromClause
    ImplicitSqlQuery

    Examples
    --------
    The following queries are considered as explicit queries:

    .. code-block:: sql

        SELECT *
        FROM R
            JOIN S ON R.a = S.b
            JOIN T ON S.b = T.c
        WHERE R.a < 42

    .. code-block:: sql

        SELECT *
        FROM R
            JOIN S ON R.a = S.b AND R.a = (SELECT MIN(R.a) FROM R)
            JOIN T ON S.b = T.c
    """

    def __init__(
        self,
        *,
        select_clause: Select,
        from_clause: Optional[ExplicitFromClause] = None,
        where_clause: Optional[Where] = None,
        groupby_clause: Optional[GroupBy] = None,
        having_clause: Optional[Having] = None,
        orderby_clause: Optional[OrderBy] = None,
        limit_clause: Optional[Limit] = None,
        cte_clause: Optional[CommonTableExpression] = None,
        explain_clause: Optional[Explain] = None,
        hints: Optional[Hint] = None,
    ) -> None:
        super().__init__(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
            groupby_clause=groupby_clause,
            having_clause=having_clause,
            orderby_clause=orderby_clause,
            limit_clause=limit_clause,
            cte_clause=cte_clause,
            explain=explain_clause,
            hints=hints,
        )

    @property
    def from_clause(self) -> Optional[ExplicitFromClause]:
        return self._from_clause

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return True


class MixedSqlQuery(SqlQuery):
    """A mixed query allows for both the explicit as well as the implicit syntax to be used within the same *FROM* clause.

    The mixed query complements `ImplicitSqlQuery` and `ExplicitSqlQuery` by removing the "purity" restriction: the tables that
    appear in the *FROM* clause can be described using either plain references or subqueries and they are free to use the
    *JOIN ON* syntax. The only thing that is not allowed as a *FROM* clause is an instance of `ImplicitFromClause` or an
    instance of `ExplicitFromClause`, since those cases are already covered by their respective query classes.

    Notice however, that we currently do not enforce the `From` clause to not be a valid explicit or implicit clause. All
    checks happen on a type level. If the contents of a general `From` clause just happen to also be a valid
    `ImplicitFromClause`, this is fine.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    Raises
    ------
    ValueError
        If the given `from_clause` is either an implicit *FROM* clause or an explicit one.
    """

    def __init__(
        self,
        *,
        select_clause: Select,
        from_clause: Optional[From] = None,
        where_clause: Optional[Where] = None,
        groupby_clause: Optional[GroupBy] = None,
        having_clause: Optional[Having] = None,
        orderby_clause: Optional[OrderBy] = None,
        limit_clause: Optional[Limit] = None,
        cte_clause: Optional[CommonTableExpression] = None,
        explain_clause: Optional[Explain] = None,
        hints: Optional[Hint] = None,
    ) -> None:
        if isinstance(from_clause, ExplicitFromClause) or isinstance(
            from_clause, ImplicitFromClause
        ):
            raise ValueError(
                "MixedSqlQuery cannot be combined with explicit/implicit FROM clause"
            )
        super().__init__(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
            groupby_clause=groupby_clause,
            having_clause=having_clause,
            orderby_clause=orderby_clause,
            limit_clause=limit_clause,
            cte_clause=cte_clause,
            explain=explain_clause,
            hints=hints,
        )

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return False


class SetQuery:
    """A set query combines the result sets of two queries using one of the set operations.

    Set operations include *UNION*, *UNION ALL*, *INTERSECT*, and *EXCEPT*. We represent set queries as a different
    type than "plain" *SELECT* queries because these allow for a different interface (e.g. providing access to predicates
    or the *SELECT* block). See the documentation of `SqlQuery` for more details on the distinction and the reasoning behind
    it.

    Still, the `SetQuery` provides exactly the same high-level interface. In case a specific method or property is not
    applicable for set queries (e.g. calling ``query.predicates()``), a `QueryTypeError` will be raised. This is motivated by
    entirely pragmatic reasons: oftentimes a client will not care whether it receive a `SqlQuery` or a `SetQuery` because it is
    only interested in the common denominator between the two (e.g. calling ``str`` or retrieving its tables). Therefore, we
    want to be set queries applicable in the same places. At the same time, set queries are a much more recent addition to
    PostBOUND, and we do not want to force the client to update its code base if this is not really necessary.

    Notice that set queries provide some clauses are supported by both plain SQL queries as well as set queries.

    Parameters
    ----------
    left_query : SelectStatement
        The left-hand side of the set operation
    right_query : SelectStatement
        The right-hand side of the set operation
    set_operation : SetOperator
        The actual operation to combine the two result sets.
    cte_clause : Optional[CommonTableExpression], optional
        The **WITH** part of the query, by default **None**
    orderby_clause : Optional[OrderBy], optional
        The **ORDER BY** part of the query, by default **None**
    limit_clause : Optional[Limit], optional
        The **LIMIT** and **OFFSET** part of the query. In standard SQL, this is designated using the *FETCH FIRST* syntax.
        Defaults to **None**.
    hints : Optional[Hint], optional
        The hint block of the query. Hints are not part of standard SQL and follow a completely system-specific syntax. Even
        their placement in within the query varies from system to system and from extension to extension. Defaults to **None**.
    explain_clause : Optional[Explain], optional
        The **EXPLAIN** part of the query. Like hints, this is not part of standard SQL. However, most systems provide
        **EXPLAIN** functionality. The specific features and syntax are quite similar, but still system specific. Defaults to
        **None**.

    See Also
    --------
    SqlQuery
    """

    def __init__(
        self,
        left_query: SelectStatement,
        right_query: SelectStatement,
        *,
        set_operation: SetOperator,
        cte_clause: Optional[CommonTableExpression] = None,
        orderby_clause: Optional[OrderBy] = None,
        limit_clause: Optional[Limit] = None,
        hints: Optional[Hint] = None,
        explain_clause: Optional[Explain] = None,
    ) -> None:
        if left_query.is_explain():
            warnings.warn(
                "Left query is an EXPLAIN query. Ignoring the EXPLAIN clause.",
                stacklevel=2,
            )
            left_query = build_query(left_query.clauses(skip=Explain))
        if right_query.is_explain():
            warnings.warn(
                "Right query is an EXPLAIN query. Ignoring the EXPLAIN clause.",
                stacklevel=2,
            )
            right_query = build_query(right_query.clauses(skip=Explain))

        self._lhs = left_query
        self._rhs = right_query
        self._op = set_operation
        self._cte = cte_clause
        self._orderby = orderby_clause
        self._limit = limit_clause
        self._hints = hints
        self._explain = explain_clause
        self._hash_val = hash(
            (
                self._lhs,
                self._rhs,
                self._op,
                self._cte,
                self._limit,
                self._hints,
                self._explain,
            )
        )

    __slots__ = (
        "_lhs",
        "_rhs",
        "_op",
        "_cte",
        "_orderby",
        "_limit",
        "_hints",
        "_explain",
        "_hash_val",
    )
    __match_args__ = ("set_operation", "left_query", "right_query")

    @property
    def set_operation(self) -> SetOperator:
        """Get the set operation that is used to combine the two queries.

        Returns
        -------
        SetOperator
            The set operation
        """
        return self._op

    @property
    def left_query(self) -> SelectStatement:
        """Get the left-hand side of the set operation.

        Returns
        -------
        SelectStatement
            The left-hand side of the set operation
        """
        return self._lhs

    @property
    def right_query(self) -> SelectStatement:
        """Get the right-hand side of the set operation.

        Returns
        -------
        SelectStatement
            The right-hand side of the set operation
        """
        return self._rhs

    @property
    def set_clause(self) -> SetOperationClause:
        """Get the set clause of the query."""

        match self._op:
            case SetOperator.Union:
                return UnionClause(self._lhs, self._rhs, union_all=False)
            case SetOperator.UnionAll:
                return UnionClause(self._lhs, self._rhs, union_all=True)
            case SetOperator.Intersect:
                return IntersectClause(self._lhs, self._rhs)
            case SetOperator.Except:
                return ExceptClause(self._lhs, self._rhs)
            case _:
                raise RuntimeError(
                    "Unknown set operation. This is likely a bug in the PostBOUND query abstraction. "
                    "Please consider filing a bug report."
                )

    @property
    def cte_clause(self) -> Optional[CommonTableExpression]:
        """Get the **WITH** clause of the query.

        Returns
        -------
        Optional[CommonTableExpression]
            The **WITH** clause if it was specified, or **None** otherwise.
        """
        return self._cte

    @property
    def orderby_clause(self) -> Optional[OrderBy]:
        """Get the **ORDER BY** clause of the query.

        Returns
        -------
        Optional[OrderBy]
            The **ORDER BY** clause if it was specified, or **None** otherwise.
        """
        return self._orderby

    @property
    def limit_clause(self) -> Optional[Limit]:
        """Get the combined **LIMIT** and **OFFSET** clauses of the query.

        According to the SQL standard, these clauses should use the **FETCH FIRST** syntax. However, many systems use
        **OFFSET** and **LIMIT** instead.

        Returns
        -------
        Optional[Limit]
            The **FETCH FIRST** clause if it was specified, or **None** otherwise.
        """
        return self._limit

    @property
    def hints(self) -> Optional[Hint]:
        """Get the hint block of the query.

        The hints can specify preparatory statements that have to be executed before the actual query is run in addition to the
        hints themselves.

        Returns
        -------
        Optional[Hint]
            The hint block if it was specified, or **None** otherwise.
        """
        return self._hints

    @property
    def explain(self) -> Optional[Explain]:
        """Get the **EXPLAIN** block of the query.

        Returns
        -------
        Optional[Explain]
            The **EXPLAIN** settings if specified, or **None** otherwise.
        """
        return self._explain

    @property
    def select_clause(self) -> Select:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are trying to access the SELECT clause on a set query. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    @property
    def from_clause(self) -> Optional[From]:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are trying to access the FROM clause on a set query. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    @property
    def where_clause(self) -> Optional[Where]:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are trying to access the WHERE clause on a set query. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    @property
    def groupby_clause(self) -> Optional[GroupBy]:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are trying to access the GROUP BY clause on a set query. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    @property
    def having_clause(self) -> Optional[Having]:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are trying to access the HAVING clause on a set query. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    def is_implicit(self) -> bool:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are accessing a set query. "
            "Set queries are neither explicit nor implicit. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    def is_explicit(self) -> bool:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are accessing a set query. "
            "Set queries are neither explicit nor implicit. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    def is_explain(self) -> bool:
        """Checks, whether this query is an **EXPLAIN** query rather than a normal SQL query.

        An **EXPLAIN** query is not executed like a normal **SELECT** query. Instead of actually calculating a result set,
        the database system only provides a query plan. This plan is the execution plan that would be used, had the query been
        entered as a normal SQL query.

        Returns
        -------
        bool
            Whether this query should be explained, rather than executed.
        """
        return self._explain is not None

    def tables(self) -> set[TableReference]:
        """Provides all tables that are referenced at any point in the query.

        This includes tables from all clauses. Virtual tables will be included and tables that are only scanned within
        subqueries are included as well. Notice however, that some database systems might not support subqueries to be put
        at arbitrary positions in the query (e.g. **GROUP BY** clause).

        Returns
        -------
        set[TableReference]
            All tables that are referenced in the query.
        """
        return self._lhs.tables() | self._rhs.tables()

    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced at any point in the query.

        This includes columns from all clauses and does not account for renamed columns from subqueries. For example, consider
        the query ``SELECT R.a, my_sq.b FROM R JOIN (SELECT b FROM S) my_sq ON R.a < my_sq.b``. `columns` would return the
        following set: ``{R.a, S.b, my_sq.b}``, even though ``my_sq.b`` can be considered as just an alias for ``S.b``.

        Returns
        -------
        set[ColumnReference]
            All columns that are referenced in the query.
        """
        return self._lhs.columns() | self._rhs.columns()

    def output_columns(self) -> Sequence[ColumnReference]:
        """Provides the columns that form the result relation of this query.

        Columns are ordered according to their appearance in the *SELECT* clause and will not have a bound table associated
        with them. This is because the query result is "anonymous" and does not have a relation name associated with it.
        The columns are named according to the following rules:

        - If the expression has an alias, this name is used
        - If the expression is a simple column reference, the column name is used
        - Otherwise, a generic name is used, e.g. "column_1", "column_2", etc.

        Additionally, to resolve naming conflicts between the left-hand side and the right-hand side of the set operation,
        names from the left-hand side overwrite names from the right-hand side. Names from the right-hand side are only used if
        the left-hand side does not provide a name. If both sides do not specify a name, a new generic name is used.

        Returns
        -------
        Sequence[ColumnReference]
            The columns. Their order matches the order in which they appear in the *SELECT* clause of the query.
        """
        cols: list[ColumnReference] = []
        anon_idx = 1

        lhs_cols, rhs_cols = (
            self._lhs.output_columns(),
            self._rhs.output_columns(),
        )
        if len(lhs_cols) != len(rhs_cols):
            raise ValueError(
                "The left and right queries of a set operation must have the same number of columns"
            )

        for i in range(len(lhs_cols)):
            lhs_col = lhs_cols[i]
            if not lhs_col.name.startswith("column_"):
                cols.append(lhs_col)
                continue

            rhs_col = rhs_cols[i]
            if not rhs_col.name.startswith("column_"):
                cols.append(rhs_col)
                continue

            cols.append(ColumnReference(f"column_{anon_idx}"))
            anon_idx += 1

        return cols

    def predicates(self) -> QueryPredicates:
        """Placeholder method to ensure compatibility with the `SqlQuery` interface. Raises a `QueryTypeError`."""
        raise QueryTypeError(
            "You are trying to access the predicates on a set query. "
            "Set queries do not have predicates by themselves, since they combine other queries. "
            "Make sure to check the actual query type before accessing specific clauses."
        )

    def subqueries(self) -> Collection[SqlQuery]:
        """Provides all subqueries that are referenced in this query.

        Notice that CTEs are ignored by this method, since they can be accessed directly via the `cte_clause` property.

        Returns
        -------
        Collection[SqlQuery]
            All subqueries that appear in any of the "inner" clauses of the query
        """
        # as an implementation detail we know for a fact that SqlQuery always returns a set
        return self._lhs.subqueries() | self._rhs.subqueries()

    def clauses(
        self, *, skip: Optional[Type | Iterable[Type]] = NoneType
    ) -> Sequence[BaseClause]:
        """Provides all the clauses that are defined (i.e. not *None*) in this query.

        Parameters
        ----------
        skip : Optional[Type | Iterable[Type]], optional
            The clause types that should be skipped in the output. This can be a single type or an iterable of types.

        Returns
        -------
        Sequence[BaseClause]
            The clauses. The current order of the clauses is as follows: hints, explain, cte, set operation, orderby, limit.
            Notice however, that this order is not strictly standardized and may change in the future.
            All clauses that are not specified on the query will be skipped.
        """
        clauses: list[BaseClause] = []

        if self._hints:
            clauses.append(self._hints)
        if self._explain:
            clauses.append(self._explain)
        if self._cte:
            clauses.append(self._cte)

        match self._op:
            case SetOperator.Union:
                clauses.append(UnionClause(self._lhs, self._rhs, union_all=False))
            case SetOperator.UnionAll:
                clauses.append(UnionClause(self._lhs, self._rhs, union_all=True))
            case SetOperator.Intersect:
                clauses.append(IntersectClause(self._lhs, self._rhs))
            case SetOperator.Except:
                clauses.append(ExceptClause(self._lhs, self._rhs))

        if self._orderby:
            clauses.append(self._orderby)
        if self._limit:
            clauses.append(self._limit)

        return [c for c in clauses if not isinstance(c, skip)]

    def bound_tables(self) -> set[TableReference]:
        """Provides all tables that can be assigned to a physical or virtual table reference in this query.

        Bound tables are those tables, that are selected in the *FROM* clause of the query, or a subquery. Conversely,
        unbound tables are those that have to be "injected" by an outer query, as is the case for dependent subqueries.

        For example, the query ``SELECT * FROM R, S WHERE R.a = S.b`` has two bound tables: *R* and *S*.
        On the other hand, the query ``SELECT * FROM R WHERE R.a = S.b`` has only bound *R*, whereas *S* has to be bound in
        a surrounding query.

        Returns
        -------
        set[TableReference]
            All tables that are bound (i.e. listed in any *FROM* clause or a CTE) of the query.
        """
        return self._lhs.bound_tables() | self._rhs.bound_tables()

    def unbound_tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in this query but not bound.

        While `tables()` provides all tables that are referenced in this query in any way, `bound_tables` restricts
        these tables. This method provides the complementary set to `bound_tables` i.e.
        ``tables = bound_tables ⊕ unbound_tables``.

        Returns
        -------
        set[TableReference]
            The unbound tables that have to be supplied as part of an outer query
        """
        return self._lhs.unbound_tables() | self._rhs.unbound_tables()

    def is_ordered(self) -> bool:
        """Checks, whether this query produces its result tuples in order.

        Returns
        -------
        bool
            Whether a valid *ORDER BY* clause was specified on the query.
        """
        return self._orderby is not None

    def is_dependent(self) -> bool:
        """Checks, whether all columns that are referenced in this query are provided by the tables from this query.

        In order for this check to work, all columns have to be bound to actual tables, i.e. the `tables` attribute of all
        column references have to be set to a valid object.

        Returns
        -------
        bool
            Whether all columns belong to tables that are bound by this query
        """
        # we cannot apply the same check as in SqlQuery, since a unbound table in one inner query might be unbound in the other
        # TODO: the above situation would likely lead to an invalid SQL query anyway, so it would be nice to assert that this
        # is not the case during intialization of the SetQuery
        return self._lhs.is_dependent() or self._rhs.is_dependent()

    def is_scalar(self) -> bool:
        """Checks, whether the query is guaranteed to provide a single scalar value as a result.

        Scalar results can only be calculated by queries with a single projection in the *SELECT* clause and if that projection
        is an aggregate function, e.g. *SELECT min(R.a) FROM R*. However, there are other queries which could also be scalar
        "by chance", e.g. *SELECT R.b FROM R WHERE R.a = 1*  if *R.a* is the primary key of *R*. Notice that such cases are not
        recognized by this method.

        Returns
        -------
        bool
            Whether the query will always return a single scalar value
        """
        # set queries cannot guarantee a scalar result
        return False

    def is_set_query(self) -> bool:
        """Checks, whether this query is a set query.

        A set query is a query that combines the results of two or more queries into a single result set. This can be done
        by combining the tuples from both sets using a *UNION* clause (which removes duplicates), or a *UNION ALL* clause
        (which retains duplicates). Alternatively, only tuples that are present in both sets can be retained using an
        *INTERSECT* clause. Finally, all tuples from the first result set that are not part of the second result set can be
        computed using an *EXCEPT* clause.

        Notice that only one of the set operators can be used at a time, but the input query of one set operation can itself
        use another set operation.

        Returns
        -------
        bool
            Whether this query is a set query
        """
        return True

    def contains_cross_product(self) -> bool:
        """Checks, whether this query has at least one cross product.

        Returns
        -------
        bool
            Whether this query has cross products.
        """
        return self._lhs.contains_cross_product() or self._rhs.contains_cross_product()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all expressions that are directly contained in this query.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        return util.flatten(clause.iterexpressions() for clause in self.clauses())

    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides access to all column in this query.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n* times
        in this query, it will also be returned *n* times by this method. Furthermore, the order in which columns are provided
        by the iterable matches the order in which they appear in this query.

        Returns
        -------
        Iterable[ColumnReference]
            The columns
        """
        return util.flatten(clause.itercolumns() for clause in self.clauses())

    def stringify(self, *, trailing_delimiter: bool = True) -> str:
        """Provides a string representation of this query.

        The only difference to calling `str` directly, is that the `stringify` method provides control over whether a trailing
        delimiter should be appended to the query.

        Parameters
        ----------
        trailing_delimiter : bool, optional
            Whether a delimiter should be appended to the query. Defaults to *True*.

        Returns
        -------
        str
            A string representation of this query
        """
        delim = ";" if trailing_delimiter else ""
        return (
            "".join(
                _stringify_clause(clause).rstrip("; ") for clause in self.clauses()
            ).rstrip()
            + delim
        )

    def ast(self) -> str:
        """Provides a human-readable representation of the abstract syntax tree for this query.

        The AST is a textual representation of the query that shows the structure of the query in a tree-like manner.

        Returns
        -------
        str
            The abstract syntax tree of this query
        """
        return _create_ast(self)

    def accept_visitor(
        self, clause_visitor: ClauseVisitor, *args, **kwargs
    ) -> dict[BaseClause, VisitorResult]:
        """Applies a visitor over all clauses in the current query.

        Notice that since the visitor is applied to all clauses, it returns the results for each of them.

        Parameters
        ----------
        clause_visitor : ClauseVisitor
            The visitor algorithm to use.
        """
        return {
            clause: clause.accept_visitor(clause_visitor, *args, **kwargs)
            for clause in self.clauses()
        }

    def __json__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self._lhs == other._lhs
            and self._rhs == other._rhs
            and self._op == other._op
            and self._cte == other._cte
            and self._orderby == other._orderby
            and self._limit == other._limit
            and self._hints == other._hints
            and self._explain == other._explain
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.stringify(trailing_delimiter=True)


SelectStatement = SqlQuery | SetQuery
"""Super type that might be any valid SQL query, i.e. plain **SELECT** queries or set queries, but no DML, DDL, etc.

See Also
--------
SqlQuery
SetQuery
"""

SqlStatement = SelectStatement
"""Super type that might be any valid SQL statement (including queries, DML statements, DDL statements, etc.).

For now, this is equivalent with a `SelectStatement`, but we might add support for additional statements in the future.

See Also
--------
SelectStatement
"""


class QueryTypeError(RuntimeError):
    """Error to indicate that a different type of query was expected (e.g. an `ExplicitSqlQuery` instead of a `SetQuery`)."""

    def __init__(self, *args) -> None:
        super().__init__(*args)


def build_query(query_clauses: Iterable[BaseClause]) -> SqlQuery:
    """Constructs an SQL query based on specific clauses.

    No validation is performed. If clauses appear multiple times, later clauses overwrite former ones. The specific
    type of query (i.e. implicit, explicit or mixed) is inferred from the clauses (i.e. occurrence of an implicit *FROM*
    clause enforces an `ImplicitSqlQuery` and vice-versa). The overwriting rules apply here as well: a later `From` clause
    overwrites a former one and can change the type of the produced query.

    This method can also be used to contruct `SetQuery` objects by passing one of the set clauses (`UnionClause`,
    `IntersectClause` or `ExceptClause`). In this case, the user must ensure that no clauses that are illegal in the context of
    a set operation are supplied (e.g. `Select` or `From`). Otherwise, an error is raised.

    Parameters
    ----------
    query_clauses : Iterable[BaseClause]
        The clauses that should be used to construct the query. If any of the clauses are **None**, they will simply be
        skipped.

    Returns
    -------
    SqlQuery
        A query consisting of the specified clauses

    Raises
    ------
    ValueError
        If `query_clauses` does not contain a `Select` clause
    ValueError
        If any of the clause types is unknown. This indicates that this method is missing a handler for a specific clause type
        that was added later on.
    """
    build_implicit_query, build_explicit_query, build_set_query = (
        True,
        True,
        False,
    )

    cte_clause = None
    select_clause, from_clause, where_clause = None, None, None
    groupby_clause, having_clause = None, None
    orderby_clause, limit_clause = None, None
    union_clause, intersect_clause, except_clause = None, None, None
    explain_clause, hints_clause = None, None
    for clause in query_clauses:
        if not clause:
            continue

        if isinstance(clause, CommonTableExpression):
            cte_clause = clause
        elif isinstance(clause, Select):
            select_clause = clause
        elif isinstance(clause, ImplicitFromClause):
            from_clause = clause
            build_implicit_query, build_explicit_query = True, False
        elif isinstance(clause, ExplicitFromClause):
            from_clause = clause
            build_implicit_query, build_explicit_query = False, True
        elif isinstance(clause, From):
            from_clause = clause
            build_implicit_query, build_explicit_query = False, False
        elif isinstance(clause, Where):
            where_clause = clause
        elif isinstance(clause, GroupBy):
            groupby_clause = clause
        elif isinstance(clause, Having):
            having_clause = clause
        elif isinstance(clause, OrderBy):
            orderby_clause = clause
        elif isinstance(clause, Limit):
            limit_clause = clause
        elif isinstance(clause, UnionClause):
            build_set_query = True
            union_clause = clause
        elif isinstance(clause, ExceptClause):
            build_set_query = True
            except_clause = clause
        elif isinstance(clause, IntersectClause):
            build_set_query = True
            intersect_clause = clause
        elif isinstance(clause, Explain):
            explain_clause = clause
        elif isinstance(clause, Hint):
            hints_clause = clause
        else:
            raise ValueError("Unknown clause type: " + str(clause))

    if build_set_query:
        if union_clause is not None:
            setop = (
                SetOperator.UnionAll if union_clause.union_all else SetOperator.Union
            )
            lhs, rhs = union_clause.left_query, union_clause.right_query
        elif except_clause is not None:
            setop = SetOperator.Except
            lhs, rhs = except_clause.left_query, except_clause.right_query
        elif intersect_clause is not None:
            setop = SetOperator.Intersect
            lhs, rhs = intersect_clause.left_query, intersect_clause.right_query
        else:
            raise ValueError("Unknown set operation")

        misplaced_clauses = [
            select_clause,
            from_clause,
            where_clause,
            groupby_clause,
            having_clause,
        ]
        if any(clause for clause in misplaced_clauses if clause is not None):
            raise ValueError(
                "Set operation specified but illegal clauses found. "
                "Set clauses do not support SELECT, FROM, WHERE, GROUP BY or HAVING clauses."
            )

        return SetQuery(
            lhs,
            rhs,
            set_operation=setop,
            cte_clause=cte_clause,
            orderby_clause=orderby_clause,
            limit_clause=limit_clause,
            hints=hints_clause,
            explain_clause=explain_clause,
        )

    if select_clause is None:
        raise ValueError("No SELECT clause detected")

    if build_implicit_query:
        return ImplicitSqlQuery(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
            groupby_clause=groupby_clause,
            having_clause=having_clause,
            orderby_clause=orderby_clause,
            limit_clause=limit_clause,
            cte_clause=cte_clause,
            hints=hints_clause,
            explain_clause=explain_clause,
        )
    elif build_explicit_query:
        return ExplicitSqlQuery(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
            groupby_clause=groupby_clause,
            having_clause=having_clause,
            orderby_clause=orderby_clause,
            limit_clause=limit_clause,
            cte_clause=cte_clause,
            hints=hints_clause,
            explain_clause=explain_clause,
        )
    else:
        return MixedSqlQuery(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
            groupby_clause=groupby_clause,
            having_clause=having_clause,
            orderby_clause=orderby_clause,
            limit_clause=limit_clause,
            cte_clause=cte_clause,
            hints=hints_clause,
            explain_clause=explain_clause,
        )


def as_query(clauses: BaseClause | Iterable[BaseClause], *args) -> SqlQuery:
    """Transforms the given clauses into a full query.

    This is just a slightly more convenient alias for `build_query` and the same rules apply:
    if clauses of the same type appear multiple times (e.g., multiple `Select` clauses), later
    occurrences overwrite earlier ones. The type of query is inferred from the clauses, e.g., if an
    `ImplicitFromClause` is present, an `ImplicitSqlQuery` will be created.
    """
    all_clauses = list(util.enlist(clauses))
    all_clauses.extend(args)
    return build_query(all_clauses)

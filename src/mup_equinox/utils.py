from typing import Any, Callable
import jax.tree as jt 
import equinox as eqx
import functools

class TreePathError(RuntimeError):
    path: tuple


def flexible_path_metadata_tree_map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
    check_type: bool = False,
    check_ndims: bool = False,
) -> Any:
    """Apply ``f`` to several pytrees while tolerating path-metadata differences.

    This helper mirrors :func:`jax.tree.map` but relaxes the requirement that
    custom-nodal paths match exactly. It is designed for Equinox modules where
    node metadata (for example, width hints, layer sizes, or other static
    attributes) may vary between otherwise compatible trees. Paths across the
    input trees must still share the same *types* (e.g. ``GetAttrKey`` vs
    ``SequenceKey``), yet their attached metadata is allowed to differ. When you
    need stronger guarantees, enable ``check_type`` to insist on matching leaf
    Python types and ``check_ndims`` to require equal array ranks.

    Args:
        f: Callable receiving one leaf per tree.
        tree, *rest: Pytrees with an equivalent structural skeleton; width-driven
            attribute differences are allowed.
        is_leaf: Optional predicate forwarded to :func:`jax.tree.flatten_with_path`.
        check_type: When ``True``, enforce that leaf objects share the same Python type.
        check_ndims: When ``True``, enforce that array leaves have the same number of dimensions.

    Returns:
        A pytree mirroring ``tree`` with the results of ``f``.

    Raises:
        TreePathError: If the path element types diverge.
        TypeError: When ``check_type`` is enabled and leaf types differ.
        ValueError: When ``check_ndims`` is enabled and array ranks differ.
    """

    def _check_path_type(path_leaf, *rest_paths_leaves):
        path, _ = path_leaf
        for t, rest_path_leaf in enumerate(rest_paths_leaves):
            rest_path, _ = rest_path_leaf
            for p, r in zip(path, rest_path):
                if type(p) is not type(r):
                    raise TreePathError(
                        f"Path mismatch: {path} vs {rest_path} for tree {t + 1}"
                    )

    def _check_type(path_leaf, *rest_paths_leaves):
        path, leaf = path_leaf
        for t, rest_path_leaf in enumerate(rest_paths_leaves):
            _, rest_leaf = rest_path_leaf
            if type(leaf) is not type(rest_leaf):
                raise TypeError(
                    f"Type mismatch: {type(leaf)} vs {type(rest_leaf)} at path {path} for tree {t + 1}"
                )

    def _check_ndims(path_leaf, *rest_paths_leaves):
        path, leaf = path_leaf
        for t, rest_path_leaf in enumerate(rest_paths_leaves):
            _, rest_leaf = rest_path_leaf
            if eqx.is_array_like(leaf) and eqx.is_array_like(rest_leaf):
                if len(leaf.shape) != len(rest_leaf.shape):
                    raise ValueError(
                        f"#Dim mismatch: {leaf.shape} vs {rest_leaf.shape} at path {path} for tree {t + 1}"
                    )

    @functools.wraps(f)
    def _f(*xs):
        try:
            _check_path_type(*xs)
            _check_type(*xs) if check_type else None
            _check_ndims(*xs) if check_ndims else None
            return f(*[x[1] for x in xs])  # pass only the leaves to f
        except TreePathError as e:
            path = xs[0][0]
            combo_path = path + e.path
            exc = TreePathError(f"Error at leaf with path {combo_path}")
            exc.path = combo_path
            raise exc from e
        except Exception as e:
            path = xs[0][0]
            exc = TreePathError(f"Error at leaf with path {path}")
            exc.path = path
            raise exc from e

    tree_paths_leaves, treedef = jt.flatten_with_path(
        tree, is_leaf=is_leaf
    )  # list of (path, leaf)
    rest_tree_paths_leaves = [
        jt.flatten_with_path(r, is_leaf=is_leaf)[0] for r in rest
    ]  # list of list of (path, leaf)
    all_tree_paths_leaves = [
        tree_paths_leaves
    ] + rest_tree_paths_leaves  # list of list of (path, leaf)
    return treedef.unflatten(_f(*xs) for xs in zip(*all_tree_paths_leaves))
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel


class ExecutionTreeNode(BaseModel):
    """
    Represents a function execution node in the function execution tree.
    """

    name: str
    module: str
    line: int
    time: float
    children: list["ExecutionTreeNode"]

    def to_df(self) -> pd.DataFrame:
        """
        Converts the function execution node subtree to a pandas DataFrame.
        """
        next_items: list[ExecutionTreeNode] = [self]
        data: list[dict[str, Any]] = []
        while next_items:
            new_next_items: list[ExecutionTreeNode] = []
            for item in next_items:
                new_next_items.extend(item.children)
                data.append(
                    {
                        "name": item.name,
                        "module": item.module,
                        "line": item.line,
                        "time": item.time,
                    }
                )
            next_items = new_next_items
        return pd.DataFrame(data)


class ExecutionTree(BaseModel):
    """
    Represents the execution tree.
    """

    root: ExecutionTreeNode

    @staticmethod
    def from_pyinstrument_output(output: dict[str, Any], root_func_name: str, try_cnt: int = 1) -> "ExecutionTree":
        """
        Converts the pyinstrument output to an execution tree.

        param output: pyinstrument output.
        param root_func_name: name of the root function.
        param try_cnt: number of validation runs. All time values are divided by this number.
        """
        root_func = ExecutionTree._get_root_func(output, root_func_name)
        if "time" not in root_func:
            raise ValueError("time   not found in root_func")
        time = root_func["time"] / try_cnt
        root = ExecutionTreeNode(
            name=root_func["function"],
            time=time,
            children=[],
            module=root_func["file_path_short"],
            line=root_func["line_no"],
        )
        ExecutionTree._build_execution_tree(root, root_func, try_cnt)
        return ExecutionTree(root=root)

    def to_df(self) -> pd.DataFrame:
        """
        Converts the execution tree to a pandas DataFrame.
        """
        df = self.root.to_df()
        df = df.groupby(["name", "module", "line"], as_index=False).sum()
        df = df.sort_values(by="time", ascending=False)
        return df

    @staticmethod
    def _get_root_func(profile_data: dict[str, Any], root_func_name: str) -> dict[str, Any]:
        """
        Finds the member in a tree with root_func_name.
        """
        root_frame = profile_data["root_frame"]
        if "children" not in root_frame:
            raise ValueError("children not found in root_frame")
        children = root_frame["children"]
        for child in children:
            if "function" not in child:
                continue
            if child["function"] == root_func_name:
                return cast(dict[str, Any], child)
        raise ValueError(f"{root_func_name} not found in profile data")

    @staticmethod
    def _build_execution_tree(parent: ExecutionTreeNode, parent_data: dict[str, Any], try_cnt: int) -> None:
        """
        Recursively builds a tree of function execution nodes from the profile data.
        """
        if "children" not in parent_data:
            return
        children = parent_data["children"]
        for child in children:
            if "function" not in child:
                continue
            child_func = ExecutionTreeNode(
                name=child["function"],
                time=child["time"] / try_cnt,
                children=[],
                module=child["file_path_short"],
                line=child["line_no"],
            )
            parent.children.append(child_func)
            ExecutionTree._build_execution_tree(child_func, child, try_cnt)

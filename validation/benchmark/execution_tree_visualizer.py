from execution_tree import ExecutionTreeNode
from rich import print
from rich.tree import Tree


class ExecutionTreeVisualizer:
    """
    Visualizes execution tree.
    """

    @staticmethod
    def print(*, root: ExecutionTreeNode, max_level: int) -> None:
        """
        Prints profiling result in a readable format.

        param root: root of execution  tree.
        param max_level: max level of the function execution node tree to print.
        """
        print("[green]\n\nPROFILING RESULT\n[/green]")
        print(f"[green]{'-'*100}[/green]")
        level = 0
        color_time_points = ExecutionTreeVisualizer._get_time_points_for_colors(root=root)
        color = ExecutionTreeVisualizer._get_time_color(root.time, color_time_points)
        tree = Tree(
            f"[yellow]{root.name}()[/yellow] - "
            f"[{color}]{root.time:.4f}s   (100.0%)[/{color}]   "
            f"[grey50]{root.module}:{root.line}[/grey50]"
        )
        total_time = root.time
        for child in root.children:
            ExecutionTreeVisualizer._print_tree(tree, child, level + 1, max_level, color_time_points, total_time)
        print(tree)

    @staticmethod
    def _print_tree(
        tree: Tree,
        node: ExecutionTreeNode,
        level: int,
        max_level: int,
        color_time_points: tuple[float, float],
        total_time: float,
    ) -> None:
        """
        Recursive function that is used to print function execution node tree with times.
        """
        if level < max_level:
            color = ExecutionTreeVisualizer._get_time_color(node.time, color_time_points)
            branch = tree.add(
                f"[yellow]{node.name}()[/yellow] - [{color}]{node.time:.4f}s   "
                f"({node.time / total_time * 100.0:.1f}%)[/{color}]   "
                f"[grey50]{node.module}:{node.line}[/grey50]"
            )
            for child in node.children:
                ExecutionTreeVisualizer._print_tree(branch, child, level + 1, max_level, color_time_points, total_time)

    @staticmethod
    def _get_time_points_for_colors(root: ExecutionTreeNode) -> tuple[float, float]:
        """
        Calculates the 1/3 and 2/3 of maximal execution time and returns.
        """
        next_nodes: list[ExecutionTreeNode] = [root]
        level: int = 0
        times: list[float] = [root.time]
        while next_nodes:
            level += 1
            new_next_nodes: list[ExecutionTreeNode] = []
            for item in next_nodes:
                for child in item.children:
                    new_next_nodes.append(child)
                    times.append(child.time)
            next_nodes = new_next_nodes
            if not times:
                continue
        times.sort()
        times_cnt = len(times)
        return times[times_cnt // 3], times[2 * times_cnt // 3]

    @staticmethod
    def _get_time_color(time: float, color_time_points: tuple[float, float]) -> str:
        """
        Calculates the color for the function execution time.
        """
        if time > color_time_points[1]:
            return "red"
        elif time > color_time_points[0]:
            return "green"
        else:
            return "green"

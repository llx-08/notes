"""CuTe Layout 入门：比较 row-major、column-major 与带 padding 的布局。

CuTe DSL 需要从文件读取函数 source/AST；不要把本文件直接逐行粘到 REPL。
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def show_layouts():
    row_major = cute.make_layout((2, 3), stride=(3, 1))
    column_major = cute.make_layout((2, 3), stride=(1, 2))
    padded_row = cute.make_layout((2, 3), stride=(4, 1))

    cute.printf("row-major    = {}", row_major)
    cute.printf("column-major = {}", column_major)
    cute.printf("padded-row   = {}", padded_row)

    # Layout 是 coordinate → offset 的映射函数。
    for row in cutlass.range_constexpr(2):
        for column in cutlass.range_constexpr(3):
            coordinate = (row, column)
            cute.printf(
                "coord=({},{}): row-major={} column-major={} padded={}",
                row,
                column,
                row_major(coordinate),
                column_major(coordinate),
                padded_row(coordinate),
            )


if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()
    show_layouts()

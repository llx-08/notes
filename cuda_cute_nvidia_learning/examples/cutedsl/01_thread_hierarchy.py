"""用 CuTe DSL 打印 grid/block/thread/warp/lane。

运行：
    python3 01_thread_hierarchy.py

这个实验故意使用 device printf；它用于理解执行层次，不用于性能测试。
"""

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.runtime import cuda as cuda_runtime


@cute.kernel
def print_hierarchy(experiment: cutlass.Constexpr):
    block_x, block_y, block_z = cute.arch.block_idx()
    thread_x, thread_y, thread_z = cute.arch.thread_idx()
    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()
    cute.printf(
        "experiment={} block=({},{},{}) thread=({},{},{}) warp={} lane={}",
        experiment,
        block_x,
        block_y,
        block_z,
        thread_x,
        thread_y,
        thread_z,
        warp,
        lane,
    )


@cute.jit
def launch_one_warp():
    print_hierarchy(1).launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.jit
def launch_two_warps():
    print_hierarchy(2).launch(grid=(1, 1, 1), block=(64, 1, 1))


if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()

    print("experiment 1: blockDim.x=32，预计只有 warp 0，lane 0..31")
    launch_one_warp()
    cuda_runtime.stream_sync(0)

    print("experiment 2: blockDim.x=64，预计 warp 0/1 各有 lane 0..31")
    launch_two_warps()
    cuda_runtime.stream_sync(0)

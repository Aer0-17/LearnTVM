import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm", host="llvm")


# 描述向量计算 此过程只是在声明应该如何进行计算，不会发生实际的计算
# n 表示shape
#A B是占位符张量
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

# 创建schedule

# for (int i = 0; i < n; ++i){
#     C[i] = A[i] + B[i];
# }
# for i in n:
#     C[i] = A[i] + B[i]

s = te.create_schedule(C.op)

fadd = tvm.build(s, [A, B ,C], tgt, name="myadd")

# 创建 TVM 编译调度的目标设备
dev = tvm.device(tgt.kind.name, 0)
print(tgt.kind.name)

# 初始化设备中的张量
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

print("-------------------")

import timeit

np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))

log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(fadd, tgt, "naive", log=log)

s[C].parallel(C.op.axis[0])

print(tvm.lower(s, [A, B, C], simple_mode=True))

fadd_parallel = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")
fadd_parallel(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

evaluate_addition(fadd_parallel, tgt, "parallel", log=log)


# 更新 schedule 以使用向量化
# 重新创建 schedule, 因为前面的例子在并行操作中修改了它
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

# 这个因子应该和适合 CPU 的线程数量匹配。
# 这会因架构差异而有所不同，不过好的规则是
# 将这个因子设置为 CPU 可用内核数量。
factor = 4

outer, inner = s[C].split(C.op.axis[0], factor=factor)
s[C].parallel(outer)
s[C].vectorize(inner)

fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")

evaluate_addition(fadd_vector, tgt, "vector", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))


# 比较不同的 schedule
baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )

# 要运行这个代码, 更改为 `run_cuda = True`
# 注意：默认这个示例不在 CI 文档上运行

run_cuda = False
if run_cuda:
    # 将这个 target 改为你 GPU 的正确后端。例如：NVIDIA GPUs：cuda；Radeon GPUS：rocm；opencl：OpenCL
    tgt_gpu = tvm.target.Target(target="opencl", host="llvm")

    # 重新创建 schedule
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    print(type(C))

    s = te.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)

    ################################################################################
    # 最终必须将迭代轴 bx 和 tx 和 GPU 计算网格绑定。
    # 原生 schedule 对 GPU 是无效的, 这些是允许我们生成可在 GPU 上运行的代码的特殊构造

    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    ######################################################################
    # 编译
    # -----------
    # 指定 schedule 后, 可将它编译为 TVM 函数。
    # 默认 TVM 编译为可直接从 Python 端调用的类型擦除函数。
    #
    # 下面将用 tvm.build 来创建函数。
    # build 函数接收 schedule、所需的函数签名（包括输入和输出）以及要编译到的目标语言。
    #
    # fadd 的编译结果是 GPU 设备函数（如果利用了 GPU）以及调用 GPU 函数的主机 wrapper。
    # fadd 是生成的主机 wrapper 函数，它包含对内部生成的设备函数的引用。

    fadd_gpu = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd_opencl")

    ################################################################################
    # 编译后的 TVM 函数提供了一个任何语言都可调用的 C API。
    #
    # 我们在 Python 中提供了最小数组 API 来进行快速测试以及制作原型。
    # 数组 API 基于 `DLPack [https://github.com/dmlc/dlpack](https://github.com/dmlc/dlpack)`_ 标准。
    #
    # - 首先创建 GPU 设备。
    # - 然后 tvm.nd.array 将数据复制到 GPU 上。
    # - `fadd` 运行真实的计算。
    # - `numpy()` 将 GPU 数组复制回 CPU 上（然后验证正确性）。
    #
    # 注意将数据复制进出内存是必要步骤。

    dev = tvm.device(tgt_gpu.kind.name, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd_gpu(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    ################################################################################
    # 检查生成的 GPU 代码
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 可以在 TVM 中检查生成的代码。tvm.build 的结果是一个 TVM 模块。fadd 是包含主机模块的主机 wrapper，对 CUDA（GPU）函数来说它还包含设备模块。
    #
    # 下面的代码从设备模块中取出并打印内容代码。

    if (
        tgt_gpu.kind.name == "cuda"
        or tgt_gpu.kind.name == "rocm"
        or tgt_gpu.kind.name.startswith("opencl")
    ):
        dev_module = fadd_gpu.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())
    else:
        print(fadd_gpu.get_source())

    evaluate_addition(fadd_gpu, tgt_gpu, "gpu", log=log)
    
from tvm.contrib import cc
from tvm.contrib import utils

if run_cuda:
    fadd = fadd_gpu
    tgt = tgt_gpu
else:
    fadd = fadd
    tgt = tgt

temp = utils.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt.kind.name == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt.kind.name == "rocm":
    fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
if tgt.kind.name.startswith("opencl"):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())


fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
if tgt.kind.name == "cuda":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name == "rocm":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name.startswith("opencl"):
    print("tgt: opencl")
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
evaluate_addition(fadd1, tgt, "module", log=log)

fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
evaluate_addition(fadd1, tgt, "pack", log=log)

if tgt.kind.name.startswith("opencl"):
    fadd_cl = tvm.build(s, [A, B, C], tgt, name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    dev = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd_cl(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
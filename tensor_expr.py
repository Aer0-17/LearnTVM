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
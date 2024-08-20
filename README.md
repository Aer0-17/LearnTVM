# LearnTVM
## 1. TVM原理介绍
### 1.1 TVM和模型优化概述
![image](https://github.com/user-attachments/assets/4ad8b85d-aa96-46eb-8ff0-c08dd23c7a11)

1. 导入模型
2. 翻译成TVM的高级模型语言**Relay**。Relay是神经网络的功能语言和中间表示(IR)。Relay应用**图级优化 pass**来优化模型。它支持：
   - 传统的数据流式表示
   - 函数式作用域，let-binding 使其成为一种功能齐全的可微语言
   - 允许用户混用两种编程风格的能力
4. 降级为张量表达式（TE）表示。降级是指将较高级的表示转换为较低级的表示。应用了高级优化之后，Relay 通过运行 FuseOps pass，把模型划分为许多小的子图，并将子图降级为 TE 表示。张量表达式（TE）是一种用于描述张量计算的领域特定语言。 TE 还提供了几个 schedule 原语来指定底层循环优化，例如循环切分、矢量化、并行化、循环展开和融合。为将 Relay 表示转换为 TE 表示，TVM 包含了一个张量算子清单（TOPI），其中包含常用张量算子的预定义模板（例如，conv2d、transpose）。
5. 使用 auto-tuning 模块 AutoTVM 或 AutoScheduler 搜索最佳 schedule。schedule 为 TE 中定义的算子或子图指定底层循环优化。auto-tuning 模块搜索最佳 schedule，并将其与 cost model 和设备上的测量值进行比较。 TVM 中有两个 auto-tuning 模块。
   - AutoTVM：基于模板的 auto-tuning 模块。它运行搜索算法以在用户定义的模板中找到可调 knob 的最佳值。 TOPI 中已经提供了常用算子的模板。
   - AutoScheduler（又名 Ansor）：无模板的 auto-tuning 模块。它不需要预定义的 schedule 模板，而是通过分析计算定义自动生成搜索空间，然后在生成的搜索空间中搜索最佳 schedule。
6. 为模型编译选择最佳配置。调优后，auto-tuning 模块会生成 JSON 格式的调优记录。此步骤为每个子图选择最佳 schedule。
7. 降级为张量中间表示（TIR，TVM 的底层中间表示）。基于调优步骤选择最佳配置后，所有 TE 子图降级为 TIR 并通过底层优化 pass 进行优化。接下来，优化的 TIR 降级为硬件平台的目标编译器。这是生成可部署到生产的优化模型的最终代码生成阶段。 TVM 支持多种不同的编译器后端：
   - LLVM，针对任意微处理器架构，包括标准 x86 和 ARM 处理器、AMDGPU 和 NVPTX 代码生成，以及 LLVM 支持的任何其他平台。
   - 特定编译器，例如 NVCC（NVIDIA 的编译器）。
   - 嵌入式和特定 target，通过 TVM 的 自定义代码生成（Bring Your Own Codegen, BYOC）框架实现。
8. 编译成机器码。compiler-specific 的生成代码最终可降级为机器码。 TVM 可将模型编译为可链接对象模块，然后轻量级 TVM runtime 可以用 C 语言的 API 来动态加载模型，也可以为 Python 和 Rust 等其他语言提供入口点。或将 runtime 和模型放在同一个 package 里时，TVM 可以对其构建捆绑部署。


- **schedule（调度）** 是指一组描述如何在特定硬件上执行计算操作的规则和策略。它定义了计算图的运算顺序、并行化方式、数据存储位置、内存访问模式等，最终生成优化后的代码，用于在目标硬件上高效地运行模型。
- **shape** 通常指的是张量（tensor）的形状，即其维度大小。形状定义了张量在每一维度上的大小。例如，一个形状为 (3, 224, 224) 的张量表示有3个224x224的矩阵，常见于RGB图像的输入数据。理解张量的形状对于TVM在编译、优化模型时非常重要，因为形状信息直接影响计算图的生成和优化。

## 2. split
在 TVM 中，`split` 是一个非常有用的调度原语，可以将一个计算轴分成两个子轴，这样做有多个好处，尤其在优化计算性能时非常重要。下面是 `split` 的主要好处和应用场景：

1. **块化 (Tiling)**
   `split` 可以用于块化计算，将大规模的计算分成更小的块，这在缓存优化和内存访问模式优化中尤为重要。例如，处理大规模矩阵乘法时，通过块化可以使得每个块能够更好地适配缓存，从而减少缓存不命中和内存带宽的消耗。

   ```python
   # 原轴为 i，j
   i, j = s[C].op.axis

   # 将 i 轴分割成两个新轴
   io, ii = s[C].split(i, factor=32)
   ```

   这会将原始的 `i` 轴拆分为两个子轴 `io` 和 `ii`，其中 `ii` 的长度为 32，而 `io` 则表示划分后的块数。

2. **并行化**
   `split` 也可以结合 `parallel` 使用，将计算分块后并行执行，从而更好地利用多核处理器的计算能力。

   ```python
   # 对 i 轴进行拆分
   io, ii = s[C].split(i, nparts=8)

   # 对外部的 io 轴进行并行化
   s[C].parallel(io)
   ```

   在这个例子中，我们将 `i` 轴分为 8 个部分，并使每个部分可以并行执行。这有助于加速大规模计算的执行，尤其是在多核 CPU 或 GPU 上。

3. **向量化**
   通过 `split`，可以将计算分割成适合向量化操作的小块，从而利用 SIMD（单指令多数据）指令提高计算效率。

   ```python
   # 对 i 轴进行分割
   io, ii = s[C].split(i, factor=8)

   # 对内部的 ii 轴进行向量化
   s[C].vectorize(ii)
   ```

   这个例子将 `i` 轴分割成大小为 8 的子轴，并对这些小块进行向量化。这可以充分利用处理器的向量指令集，提高计算效率。

4. **减少缓存冲突**
在某些情况下，`split` 可以用于控制计算访问模式，减少缓存行之间的冲突。例如，将计算分割为更小的块可以确保每个块在不同的缓存行中存储，从而减少冲突。

5. **灵活性**
`split` 提供了调度的灵活性，可以根据具体的硬件架构、问题规模、数据布局等因素来调整计算方式。例如，针对不同的硬件平台，合适的块大小可能不同，通过 `split` 可以快速调整和实验，找到最优的调度策略。

6. **总结**
`split` 是 TVM 中一个强大且灵活的工具，用于优化和控制计算的执行方式。它可以通过块化、并行化、向量化、减少缓存冲突等方式来显著提升计算性能。在实践中，`split` 通常结合其他调度原语（如 `reorder`、`fuse`、`vectorize` 等）一起使用，达到最佳的优化效果。
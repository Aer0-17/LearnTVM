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
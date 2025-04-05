# Capacitated Vehicle Routing Problem (CVRP) Genetic Algorithm

## 项目简介

本项目实现了一个用于求解容量约束车辆路径问题（CVRP）的自适应遗传算法。该算法通过模拟自然选择过程，优化车辆路径以最小化总行驶距离，同时满足车辆容量限制。

## 代码结构

- `src/main_genetic.cpp`：主程序入口，负责初始化问题、设置参数并运行遗传算法。
- `src/genetic_solver.cpp`：遗传算法的实现，包括选择、交叉、变异和局部搜索等操作。
- `include/genetic_solver.h`：遗传算法求解器的头文件，定义了算法参数和主要方法。
- `include/types.h`：定义了问题和解的基本数据结构。
- `src/loader.cpp`：负责从文件加载问题数据。

## 配置要求

- **编译器**：需要支持C++17标准的编译器（如g++ 7.3或更高版本）。
- **操作系统**：Windows系统（使用了Windows特定的文件操作命令）。

## 使用说明

1. **编译项目**
   - 使用以下命令编译项目：
     ```bash
     g++ -std=c++17 -Wall -Wextra -g src/main_genetic.cpp src/loader.cpp src/genetic_solver.cpp -I include -o cvrp_genetic.exe
     ```

2. **运行程序**
   - 运行生成的可执行文件：
     ```bash
     .\cvrp_genetic.exe
     ```
   - 程序会提示输入运行次数。

3. **修改问题文件**
   - 在 `src/main_genetic.cpp` 中修改 `problem_filename` 变量以指定不同的问题文件。
   - 默认问题文件路径为：`data/Vrp-Set-A/A/A-n60-k9.vrp`

4. **输出结果**
   - 结果将保存在 `results` 文件夹中，每次运行的结果文件名格式为：`问题名-(运行次数).txt`
   - 统计信息将保存在 `问题名_statistics.txt` 文件中。

## 进化过程概述

1. **初始化阶段**：生成初始种群并评估适应度。
2. **主循环阶段**：
   - 计算种群多样性
   - 更新自适应参数
   - 应用精英策略
   - 生成新种群（选择、交叉、变异、局部搜索）
   - 更新最佳个体

## 主要操作细节

- **选择**：使用锦标赛选择，保留优秀个体。
- **交叉**：包括顺序交叉和路径导向交叉，组合父代特征。
- **变异**：通过交换、插入和反转操作维持多样性。
- **局部搜索**：使用2-opt算法优化路径。

## 评估次数计算

- 初始种群评估：`population_size`
- 每代评估：`population_size + 0.1 × population_size × k`
- 总评估次数：`P + G × P × (1 + 0.1k)`
  - P = population_size
  - G = max_generations
  - k = 平均局部搜索尝试次数

## 参数优化建议

- **种群规模**：`min(200, max(60, 2 × problem_size))`
- **交叉率**：动态调整，初始0.9，终止0.6
- **变异率**：与多样性相关，初始0.3，终止0.1
- **局部搜索频率**：自适应调整
- **精英保留数量**：`max(1, population_size × 0.03)`
- **超载容忍度**：随代数递减

通过这些调整，可以提高算法的效率和解的质量。
# CVRP遗传算法求解器

这是一个使用自适应遗传算法求解车辆路径问题(CVRP)的实现。

## 项目结构

- `src/` - 源代码文件夹
  - `main_genetic.cpp` - 主程序入口
  - `genetic_solver.cpp` - 遗传算法求解器实现
  - `loader.cpp` - 问题加载器实现
- `include/` - 头文件文件夹
  - `types.h` - 基本数据类型定义
  - `genetic_solver.h` - 遗传算法求解器头文件
  - `loader.h` - 问题加载器头文件
- `data/` - 包含CVRP测试实例的数据文件
- `results/` - 存放解决方案输出的文件夹
- `genetic_solver.exe` - 编译好的可执行文件
- `simple_run.bat` - 简单运行脚本

## 如何使用

### 使用预编译的可执行文件

1. 直接双击运行 `simple_run.bat` 脚本，它会使用默认参数运行遗传算法求解A-n32-k5问题实例
2. 或者手动运行以下命令：

```
genetic_solver.exe [问题文件] [输出文件] [种群大小] [最大代数] [交叉率] [变异率]
```

例如：

```
genetic_solver.exe data\Vrp-Set-A\A\A-n32-k5.vrp results\A-n32-k5.sol 100 500 0.8 0.2
```

### 重新编译源代码

使用G++编译器：

```
g++ -std=c++17 -I./include -o genetic_solver.exe src/main_genetic.cpp src/genetic_solver.cpp src/loader.cpp
```

## 问题文件格式

程序可以读取标准CVRP问题实例文件，如Augerat et al.的测试集，可以在`data/Vrp-Set-A/A/`目录中找到。 
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib> // 用于 system() 函数
#include <chrono>  // 用于计时

#include "../include/types.h"
#include "../include/loader.h"
#include "../include/genetic_solver.h"

using namespace std;

// 设置UTF-8输出
void setUTF8Console()
{
#ifdef _WIN32
    // 设置控制台代码页为UTF-8 (仅Windows)
    system("chcp 65001 > nul");
#endif
}

// 辅助函数：获取当前工作目录(简化版)
string getCurrentDir()
{
    return ".";
}

// 辅助函数：检查文件是否存在
bool fileExists(const string &filename)
{
    ifstream file(filename);
    return file.good();
}

// 常见的CVRP样例文件
const vector<string> SAMPLE_FILES = {
    "A-n32-k5.vrp", "A-n33-k5.vrp", "A-n33-k6.vrp", "A-n34-k5.vrp",
    "A-n36-k5.vrp", "A-n37-k5.vrp", "A-n37-k6.vrp", "A-n38-k5.vrp",
    "A-n39-k5.vrp", "A-n39-k6.vrp", "A-n44-k6.vrp", "A-n45-k6.vrp"};

void printUsage()
{
    cout << "使用方法:" << endl;
    cout << "  ./cvrp_genetic [问题文件名] [输出文件] [种群大小] [最大代数] [交叉率] [变异率]" << endl;
    cout << "例如:" << endl;
    cout << "  ./cvrp_genetic A-n32-k5.vrp output.sol 100 500 0.8 0.2" << endl;
    cout << "  如果不指定问题文件名，会列出常见的数据文件" << endl;
}

// 列出常见的CVRP问题文件
void listDataFiles()
{
    cout << "常见的CVRP问题数据文件:" << endl;

    for (const auto &filename : SAMPLE_FILES)
    {
        cout << "  " << filename << endl;
    }
}

// 打印问题信息
void printProblemInfo(const CVRPProblem &problem)
{
    cout << "问题名称: " << problem.name << endl;
    cout << "车辆容量: " << problem.vehicle_capacity << endl;
    cout << "客户数量: " << problem.customers.size() << endl;
    cout << "仓库位置: (" << problem.depot.x << ", " << problem.depot.y << ")" << endl;
}

// 打印解决方案信息
void printSolutionInfo(const CVRPSolution &solution)
{
    cout << "解决方案信息:" << endl;
    cout << "总距离: " << solution.total_distance << endl;
    cout << "使用车辆数: " << solution.routes.size() << endl;

    // 简化路径输出，只显示路径总结信息
    for (const auto &route : solution.routes)
    {
        cout << "车辆 " << route.vehicle_id 
             << " [距离: " << route.total_distance 
             << ", 载重: " << route.total_demand 
             << ", 客户数: " << route.points.size() - 2 
             << "]";
             
        // 只显示起点和终点
        if (!route.points.empty()) {
            cout << " 路径: " << route.points.front() << " -> ... -> " << route.points.back();
        }
        cout << endl;
    }
}

int main(int argc, char *argv[])
{
    // 设置UTF-8控制台输出
    setUTF8Console();

    // 输出当前运行位置
    cout << "当前工作目录: " << getCurrentDir() << endl;

    // 默认遗传算法参数
    int population_size = 100;
    int max_generations = 500;
    double crossover_rate = 0.8;
    double mutation_rate = 0.2;
    int tournament_size = 3;
    int elite_count = 2;
    bool use_local_search = true;
    double overload_tolerance = 0.05;
    int display_interval = 50; // 默认显示间隔为50代

    // 如果没有命令行参数，使用默认参数
    string problem_filename = "data/Vrp-Set-A/A/A-n32-k5.vrp"; // 默认问题文件
    string output_file = "output.sol";        // 默认输出文件

    // 检查命令行参数
    if (argc >= 2)
    {
        problem_filename = argv[1];
    }
    else
    {
        cout << "使用默认问题文件: " << problem_filename << endl;
    }

    if (argc >= 3)
    {
        output_file = argv[2];
    }

    // 解析可选的遗传算法参数
    if (argc >= 4)
        population_size = stoi(argv[3]);
    if (argc >= 5)
        max_generations = stoi(argv[4]);
    if (argc >= 6)
        crossover_rate = stod(argv[5]);
    if (argc >= 7)
        mutation_rate = stod(argv[6]);
    if (argc >= 8)
        tournament_size = stoi(argv[7]);
    if (argc >= 9)
        elite_count = stoi(argv[8]);
    if (argc >= 10)
        use_local_search = (stoi(argv[9]) != 0);
    if (argc >= 11)
        overload_tolerance = stod(argv[10]);
    if (argc >= 12)
        display_interval = stoi(argv[11]);
    
    // 确保显示间隔有效
    if (display_interval <= 0)
        display_interval = 50;

    // 直接尝试使用提供的文件路径
    string full_problem_path = problem_filename;

    cout << "加载问题: " << full_problem_path << endl;

    // 检查文件是否存在
    if (!fileExists(full_problem_path))
    {
        cout << "错误: 问题文件不存在!" << endl;
        cout << "请确保文件位于正确的位置。" << endl;
        listDataFiles();

        cout << "\n按Enter键退出..." << endl;
        cin.get();
        return 1;
    }

    // 加载问题
    cout << "正在从 " << full_problem_path << " 加载问题..." << endl;
    CVRPProblem problem = ProblemLoader::loadProblem(full_problem_path);

    // 检查问题是否成功加载
    if (problem.customers.empty())
    {
        cout << "问题加载失败或无客户点！" << endl;

        cout << "\n按Enter键退出..." << endl;
        cin.get();
        return 1;
    }

    // 打印问题信息
    printProblemInfo(problem);

    // 创建求解器并设置基本参数
    GeneticSolver solver(population_size, max_generations, crossover_rate, mutation_rate, tournament_size);
    
    // 设置高级参数
    double penalty_factor = 1.0;       // 超载惩罚因子
    
    // 使用更积极的初始交叉变异率
    double init_crossover_rate = 0.9;  // 初始交叉率
    double final_crossover_rate = 0.6; // 最终交叉率
    double init_mutation_rate = 0.3;   // 初始变异率
    double final_mutation_rate = 0.1;  // 最终变异率
    double penalty_increase = 0.01;    // 每代增加的惩罚率
    
    // 设置高级参数
    solver.setAdvancedParameters(
        overload_tolerance,
        penalty_factor,
        elite_count,
        use_local_search,
        init_crossover_rate,
        final_crossover_rate,
        init_mutation_rate,
        final_mutation_rate,
        penalty_increase
    );

    // 设置显示间隔
    solver.setDisplayInterval(display_interval);

    cout << "使用求解器: " << solver.getName() << endl;
    cout << "遗传算法参数:" << endl;
    cout << "  种群大小: " << population_size << endl;
    cout << "  最大代数: " << max_generations << endl;
    cout << "  交叉率: " << crossover_rate << " (初始: " << init_crossover_rate << ", 最终: " << final_crossover_rate << ")" << endl;
    cout << "  变异率: " << mutation_rate << " (初始: " << init_mutation_rate << ", 最终: " << final_mutation_rate << ")" << endl;
    cout << "  锦标赛大小: " << tournament_size << endl;
    cout << "高级参数:" << endl;
    cout << "  允许超载: " << (overload_tolerance * 100) << "%" << endl;
    cout << "  精英数量: " << elite_count << endl;
    cout << "  局部搜索: " << (use_local_search ? "启用" : "禁用") << endl;
    cout << "  显示间隔: 每" << display_interval << "代" << endl;
    cout << endl;
    
    // 开始求解提示
    cout << "开始求解..." << endl;
    cout << "按 Ctrl+C 可以中止求解过程" << endl;
    cout << "------------------------------------------" << endl;

    // 开始计时
    auto start_time = chrono::high_resolution_clock::now();
    
    // 求解问题
    try {
        cout << "调用GeneticSolver::solve()..." << endl;
        CVRPSolution solution = solver.solve(problem);
        
        // 结束计时并计算求解时间
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;
        
        // 输出解决方案信息
        cout << "------------------------------------------" << endl;
        cout << "求解完成！用时: " << elapsed.count() << " 秒" << endl;
        printSolutionInfo(solution);
        
        // 保存解决方案到文件
        cout << "\n正在保存解决方案到: " << output_file << endl;
        if (ProblemLoader::saveSolution(solution, output_file)) {
            cout << "解决方案已成功保存！" << endl;
        } else {
            cout << "保存解决方案时出错！" << endl;
            return 1;
        }
    }
    catch (const exception& e) {
        cout << "求解过程中发生异常: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cout << "求解过程中发生未知异常!" << endl;
        return 1;
    }

    cout << "\nCVRP问题已成功求解。" << endl;
    return 0;
}
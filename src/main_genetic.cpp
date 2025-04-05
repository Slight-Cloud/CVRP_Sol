#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>   // 用于 system() 函数
#include <chrono>    // 用于计时
#include <algorithm> // 用于 min_element 和 max_element
#include <cmath>     // 用于 pow 和 sqrt
#include <direct.h>  // Windows系统下使用_mkdir
#include <io.h>      // Windows系统下使用_access

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
        if (!route.points.empty())
        {
            cout << " 路径: " << route.points.front() << " -> ... -> " << route.points.back();
        }
        cout << endl;
    }
}

// 清空results文件夹
void clearResultsFolder() {
    system("del /Q .\\results\\*");  // Windows命令，删除文件夹中的所有文件
}

// 确保results文件夹存在
void ensureResultsFolder() {
    if (_access(".\\results", 0) == -1) {
        _mkdir(".\\results");
    }
}

// 从文件路径中提取问题名称
string extractProblemName(const string& filepath) {
    size_t start = filepath.find_last_of("/\\");
    size_t end = filepath.find_last_of(".");
    if (start == string::npos) start = -1;
    if (end == string::npos) end = filepath.length();
    return filepath.substr(start + 1, end - start - 1);
}

// 生成结果文件名
string generateResultFileName(const string& problem_name, int run_index) {
    return "results/" + problem_name + "-(" + to_string(run_index) + ").txt";
}

// 保存统计结果
void saveStatistics(const string& problem_name, int run_times, double avg_distance, 
                   double best_distance, double worst_distance, double std_dev, 
                   double avg_vehicles, const CVRPSolution& best_solution) {
    string stats_file = "results/" + problem_name + "_statistics.txt";
    ofstream file(stats_file);
    if (!file.is_open()) {
        cout << "无法创建统计文件!" << endl;
        return;
    }

    file << "问题名称: " << problem_name << endl;
    file << "运行次数: " << run_times << endl;
    file << "平均总距离: " << avg_distance << endl;
    file << "最短距离: " << best_distance << endl;
    file << "最长距离: " << worst_distance << endl;
    file << "距离标准差: " << std_dev << endl;
    file << "平均使用车辆数: " << avg_vehicles << endl;
    file << "\n最优解详细信息:" << endl;
    
    // 保存最优解的路径信息
    file << "总距离: " << best_solution.total_distance << endl;
    file << "使用车辆数: " << best_solution.routes.size() << endl;
    for (const auto& route : best_solution.routes) {
        file << "车辆 " << route.vehicle_id 
             << " [距离: " << route.total_distance 
             << ", 载重: " << route.total_demand 
             << ", 客户数: " << route.points.size() - 2 
             << "] 路径: ";
        for (size_t i = 0; i < route.points.size(); ++i) {
            file << route.points[i];
            if (i < route.points.size() - 1) file << " -> ";
        }
        file << endl;
    }
    
    file.close();
    cout << "统计信息已保存到: " << stats_file << endl;
    cout << "平均总距离: " << avg_distance << endl;
    cout << "最短距离: " << best_distance << endl;
    cout << "最长距离: " << worst_distance << endl;
    cout << "距离标准差: " << std_dev << endl;
    cout << "平均使用车辆数: " << avg_vehicles << endl;
}

int main()
{
    // 设置UTF-8控制台输出
    setUTF8Console();

    string problem_filename = "data/Vrp-Set-A/A/A-n60-k9.vrp"; // 默认问题文件
    string output_file = "results/A-n33-k5(1).sol";             // 默认输出文件
                                                                // 输出当前运行位置
    cout << "当前问题文件: " << problem_filename << endl;

    // 检查文件是否存在
    if (!fileExists(problem_filename))
    {
        cout << "错误: 问题文件不存在!" << endl;
        cout << "请确保文件位于正确的位置。" << endl;
        listDataFiles();
        return 1;
    }

    // 获取用户输入的运行次数
    int run_times;
    cout << "请输入要运行的次数: ";
    cin >> run_times;

    // 用于计算统计信息
    double total_distance = 0.0;
    vector<double> all_distances;
    vector<int> all_vehicle_counts;
    double best_distance = numeric_limits<double>::infinity();
    double worst_distance = 0.0;
    CVRPSolution best_solution;

    // 加载问题
    cout << "正在从 " << problem_filename << " 加载问题..." << endl;
    CVRPProblem problem = ProblemLoader::loadProblem(problem_filename);

    // 检查问题是否成功加载
    if (problem.customers.empty())
    {
        cout << "问题加载失败或无客户点！" << endl;
        return 1;
    }

    // 基于问题规模设置参数
    int problem_size = problem.customers.size();
    int population_size = min(100, max(60, static_cast<int>(problem_size * 1.5))); // 种群大小为问题规模的2倍，但不小于60不大于200
    int max_generations = 400;                                 // 保持不变
    int tournament_size = max(3, population_size / 15);        // 锦标赛大小约为种群大小的6-7%
    int elite_count = max(2, population_size / 30);            // 精英数量约为种群大小的3-4%
    bool use_local_search = true;
    int display_interval = 50;

    // 自适应参数（只在这里设置一次）
    double init_crossover_rate = 0.95; // 初始交叉率高一些促进开发
    double final_crossover_rate = 0.6; // 最终保持适度的交叉率
    double init_mutation_rate = 0.2;   // 初始变异率适中
    double final_mutation_rate = 0.05; // 最终降低变异率以稳定解

    // 惩罚参数
    double overload_tolerance = min(0.0, 1.0 / problem_size); // 根据问题规模调整容忍度
    double penalty_factor = 1.0;                              // 初始惩罚因子
    double penalty_increase = 0.01;                           // 惩罚增长率

    // 输出参数设置
    cout << "\n算法参数设置：" << endl;
    cout << "问题规模: " << problem_size << " 个客户点" << endl;
    cout << "种群大小: " << population_size << endl;
    cout << "最大代数: " << max_generations << endl;
    cout << "锦标赛大小: " << tournament_size << endl;
    cout << "精英个数: " << elite_count << endl;
    cout << "初始交叉率: " << init_crossover_rate << endl;
    cout << "最终交叉率: " << final_crossover_rate << endl;
    cout << "初始变异率: " << init_mutation_rate << endl;
    cout << "最终变异率: " << final_mutation_rate << endl;
    cout << "是否使用局部搜索: " << (use_local_search ? "是" : "否") << endl;
    cout << "超载容忍度: " << overload_tolerance << endl;
    cout << "显示间隔: " << display_interval << endl;
    cout << "\n开始求解..." << endl;

    // 确保results文件夹存在并清空
    ensureResultsFolder();
    clearResultsFolder();

    // 获取问题名称
    string problem_name = extractProblemName(problem_filename);
    
    // 多次运行求解器
    for (int run = 1; run <= run_times; run++) {
        cout << "\n第 " << run << " 次运行:" << endl;
        cout << "----------------------------------------" << endl;

        // 创建求解器并设置参数
        GeneticSolver solver(population_size, max_generations, init_crossover_rate, init_mutation_rate, tournament_size);
        
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

        solver.setDisplayInterval(display_interval);
        
        try {
            CVRPSolution solution = solver.solve(problem);
            
            // 更新统计信息
            total_distance += solution.total_distance;
            all_distances.push_back(solution.total_distance);
            all_vehicle_counts.push_back(solution.routes.size());

            // 更新最优解
            if (solution.total_distance < best_distance) {
                best_distance = solution.total_distance;
                best_solution = solution;
            }
            
            // 更新最差解
            if (solution.total_distance > worst_distance) {
                worst_distance = solution.total_distance;
            }

            // 保存当前运行的解决方案
            string result_file = generateResultFileName(problem_name, run);
            if (ProblemLoader::saveSolution(solution, result_file)) {
                cout << "解决方案已保存到: " << result_file << endl;
            }
        }
        catch (const exception& e) {
            cout << "求解过程中发生异常: " << e.what() << endl;
            continue;
        }
    }

    // 计算统计信息
    double avg_distance = total_distance / run_times;
    double variance = 0.0;
    for (double dist : all_distances) {
        variance += pow(dist - avg_distance, 2);
    }
    variance /= run_times;
    double std_dev = sqrt(variance);

    // 计算平均车辆数
    double avg_vehicles = 0.0;
    for (int count : all_vehicle_counts) {
        avg_vehicles += count;
    }
    avg_vehicles /= run_times;

    // 保存统计信息
    saveStatistics(problem_name, run_times, avg_distance, best_distance, 
                  worst_distance, std_dev, avg_vehicles, best_solution);

    cout << "\n按回车键退出程序...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();

    return 0;
}
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib> // 用于 system() 函数
#include <chrono>  // 用于计时
#include <algorithm> // 用于 min_element 和 max_element
#include <cmath> // 用于 pow 和 sqrt

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

int main() {
    // 设置UTF-8控制台输出
    setUTF8Console();

    // 输出当前运行位置
    cout << "当前工作目录: " << getCurrentDir() << endl;

    // 设置固定参数
    int population_size = 50;
    int max_generations = 500;
    double crossover_rate = 0.8;
    double mutation_rate = 0.2;
    int tournament_size = 3;
    int elite_count = 2;
    bool use_local_search = true;
    double overload_tolerance = 0.05;
    int display_interval = 50;

    string problem_filename = "data/Vrp-Set-A/A/A-n32-k5.vrp"; // 默认问题文件
    string output_file = "results/A-n32-k5.sol";        // 默认输出文件

    // 检查文件是否存在
    if (!fileExists(problem_filename)) {
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
    int best_run = 0;
    CVRPSolution best_solution;

    // 加载问题
    cout << "正在从 " << problem_filename << " 加载问题..." << endl;
    CVRPProblem problem = ProblemLoader::loadProblem(problem_filename);

    // 检查问题是否成功加载
    if (problem.customers.empty()) {
        cout << "问题加载失败或无客户点！" << endl;
        return 1;
    }

    // 打印问题信息
    printProblemInfo(problem);

    // 输出算法参数
    cout << "\n算法参数设置：" << endl;
    cout << "种群大小: " << population_size << endl;
    cout << "最大代数: " << max_generations << endl;
    cout << "交叉率: " << crossover_rate << endl;
    cout << "变异率: " << mutation_rate << endl;
    cout << "锦标赛大小: " << tournament_size << endl;
    cout << "精英个数: " << elite_count << endl;
    cout << "是否使用局部搜索: " << (use_local_search ? "是" : "否") << endl;
    cout << "超载容忍度: " << overload_tolerance << endl;
    cout << "显示间隔: " << display_interval << endl;
    cout << "\n开始求解..." << endl;

    // 多次运行求解器
    for (int run = 1; run <= run_times; run++) {
        cout << "\n第 " << run << " 次运行:" << endl;
        cout << "----------------------------------------" << endl;

        // 创建求解器并设置基本参数
        GeneticSolver solver(population_size, max_generations, crossover_rate, mutation_rate, tournament_size);
        
        // 设置高级参数
        double penalty_factor = 1.0;       // 超载惩罚因子
        double init_crossover_rate = 0.9;  // 初始交叉率
        double final_crossover_rate = 0.6; // 最终交叉率
        double init_mutation_rate = 0.2;   // 初始变异率
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
        
        // 开始求解
        try {
            CVRPSolution solution = solver.solve(problem);
            
            // 更新统计信息
            total_distance += solution.total_distance;
            all_distances.push_back(solution.total_distance);
            all_vehicle_counts.push_back(solution.routes.size());

            // 更新最优解
            if (solution.total_distance < best_distance) {
                best_distance = solution.total_distance;
                best_run = run;
                best_solution = solution;
            }
            
            // 更新最差解
            if (solution.total_distance > worst_distance) {
                worst_distance = solution.total_distance;
            }

            // 保存最优解决方案到文件
            if (run == best_run) {
                if (ProblemLoader::saveSolution(solution, output_file)) {
                    cout << "最优解已保存到: " << output_file << endl;
                }
            }
        }
        catch (const exception& e) {
            cout << "求解过程中发生异常: " << e.what() << endl;
            continue;
        }
        catch (...) {
            cout << "求解过程中发生未知异常!" << endl;
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

    // 计算车辆数统计
    double avg_vehicles = 0.0;
    for (int count : all_vehicle_counts) {
        avg_vehicles += count;
    }
    avg_vehicles /= run_times;

    // 输出统计信息
    cout << "\n----------------------------------------" << endl;
    cout << "运行统计信息:" << endl;
    cout << "总运行次数: " << run_times << endl;
    cout << "平均总距离: " << avg_distance << endl;
    cout << "最短距离: " << best_distance << " (第" << best_run << "次运行)" << endl;
    cout << "最长距离: " << worst_distance << endl;
    cout << "距离标准差: " << std_dev << endl;
    cout << "平均使用车辆数: " << avg_vehicles << endl;
    cout << "\n最优解详细信息:" << endl;
    printSolutionInfo(best_solution);

    // 等待用户输入后退出
    cout << "\n按回车键退出程序...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();

    return 0;
}
#ifndef GENETIC_SOLVER_H
#define GENETIC_SOLVER_H

#include "types.h"
#include <vector>
#include <random>
#include <functional>
#include <algorithm>

using namespace std;

// 个体结构（染色体）
struct Individual {
    vector<int> chromosome;     // 客户访问序列（不包含仓库）,即编码方式
    double fitness;             // 适应度值
    double total_distance;      // 总距离
    double overload_penalty;    // 超载惩罚值
    int num_routes;             // 路径数量
    double total_overload;      // 总超载量
    vector<Route> routes;       // 解码后的路径
    
    Individual() : fitness(0.0), total_distance(0.0), overload_penalty(0.0),
                  num_routes(0), total_overload(0.0) {}
    
    // 比较函数，用于排序个体
    static bool compare(const Individual& a, const Individual& b) {
        return a.fitness > b.fitness; // 大的适应度值更好
    }
};

// 遗传算法求解器
class GeneticSolver {
private:
    // 遗传算法参数
    size_t population_size;     // 种群大小
    int max_generations;        // 最大迭代次数
    double crossover_rate;      // 当前交叉概率
    double mutation_rate;       // 当前变异概率
    size_t tournament_size;     // 锦标赛选择的大小
    double initial_overload_tolerance; // 初始允许超载比例 (0.05表示5%)
    double overload_penalty_factor;    // 超载惩罚因子
    size_t elite_size;          // 精英数量
    bool use_local_search;      // 是否使用局部搜索
    int display_interval;       // 进度显示间隔
    
    // 自适应参数
    double initial_crossover_rate;  // 初始交叉率
    double initial_mutation_rate;   // 初始变异率
    double final_crossover_rate;    // 最终交叉率
    double final_mutation_rate;     // 最终变异率
    double overload_penalty_increase_rate; // 超载惩罚增加率
    
    // 随机数生成器
    mt19937 rng;  // 使用更好的随机数引擎
    
    // 问题实例
    CVRPProblem problem;
    
    // 种群
    vector<Individual> population;
    Individual best_individual;  // 历史最佳个体
    Individual worst_individual; // 当前最差个体
    double population_diversity; // 种群多样性指标
    
public:
    GeneticSolver(int pop_size = 100, int max_gen = 500, 
                  double c_rate = 0.8, double m_rate = 0.2, 
                  int tourn_size = 3);
    
    // 设置参数
    void setParameters(int pop_size, int max_gen, double c_rate, double m_rate, int tourn_size);
    
    // 设置高级参数
    void setAdvancedParameters(double overload_tolerance = 0.05, 
                             double penalty_factor = 1.0,
                             int elite_count = 1,
                             bool local_search = true,
                             double init_c_rate = 0.9,
                             double final_c_rate = 0.6,
                             double init_m_rate = 0.3,
                             double final_m_rate = 0.1,
                             double penalty_increase = 0.01);
    
    // 设置显示间隔
    void setDisplayInterval(int interval);
    
    // 获取求解器名称
    string getName() const;
    
    // 主要求解方法
    CVRPSolution solve(const CVRPProblem& problem);
    
private:
    // 问题初始化
    void initialize();
    
    // 评估适应度
    void evaluateIndividual(Individual& ind, int current_generation = 0);
    double calculateRouteDistance(const vector<int>& route_points) const;
    
    // 解码染色体为路径 (现在允许轻微超载)
    vector<Route> decodeChromosome(const vector<int>& chromosome, double overload_tolerance) const;
    
    // 选择操作
    Individual tournamentSelection(int tournament_size);
    
    // 交叉操作
    void orderCrossover(const Individual& parent1, const Individual& parent2, 
                       Individual& offspring1, Individual& offspring2);
    
    // 路径导向交叉
    void pathOrientedCrossover(const Individual& parent1, const Individual& parent2,
                              Individual& offspring1, Individual& offspring2);
    
    // 变异操作
    void mutate(Individual& ind);
    void swapMutation(vector<int>& chromosome);
    void insertionMutation(vector<int>& chromosome);
    void inversionMutation(vector<int>& chromosome);
    
    // 局部搜索 - 2-opt
    void twoOptLocalSearch(Individual& ind);
    bool apply2OptMove(vector<int>& route, int i, int j);
    
    // 更新自适应参数
    void updateAdaptiveParameters(int current_generation);
    
    // 计算种群多样性
    double calculatePopulationDiversity();
    
    // 应用精英策略
    void applyEliteStrategy();
    
    // 计算两点之间的欧几里得距离
    double calculateDistance(const Point& a, const Point& b) const;
    
    // 辅助函数
    int getRandomInt(int min, int max);
    double getRandomDouble(double min, double max);
};

#endif // GENETIC_SOLVER_H 
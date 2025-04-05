#include "../include/genetic_solver.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <limits>
#include <set>
#include <random>

using namespace std;

// 构造函数
GeneticSolver::GeneticSolver(int pop_size, int max_gen, double c_rate, double m_rate, int tourn_size) {
    setParameters(pop_size, max_gen, c_rate, m_rate, tourn_size);
    
    // 设置默认高级参数
    setAdvancedParameters();
    
    // 设置默认显示间隔
    display_interval = 50;
    
    // 初始化随机数生成器 - 使用当前时间作为种子而不仅仅依赖random_device
    random_device rd;
    auto current_time = chrono::high_resolution_clock::now().time_since_epoch().count();
    unsigned seed = rd() ^ static_cast<unsigned>(current_time);
    cout << "使用随机种子: " << seed << endl;  // 输出种子值，便于调试
    rng.seed(seed);
    random_engine.seed(seed);  // 确保两个随机引擎使用相同的种子
}

// 设置基本参数
void GeneticSolver::setParameters(int pop_size, int max_gen, double c_rate, double m_rate, int tourn_size) {
    population_size = pop_size;
    max_generations = max_gen;
    crossover_rate = c_rate;
    mutation_rate = m_rate;
    tournament_size = tourn_size;
}

// 设置高级参数
void GeneticSolver::setAdvancedParameters(double overload_tolerance, double penalty_factor,
                                         int elite_count, bool local_search,
                                         double init_c_rate, double final_c_rate,
                                         double init_m_rate, double final_m_rate,
                                         double penalty_increase) {
    initial_overload_tolerance = overload_tolerance;
    overload_penalty_factor = penalty_factor;
    elite_size = elite_count;
    use_local_search = local_search;
    
    // 自适应参数
    initial_crossover_rate = init_c_rate;
    final_crossover_rate = final_c_rate;
    initial_mutation_rate = init_m_rate;
    final_mutation_rate = final_m_rate;
    overload_penalty_increase_rate = penalty_increase;
}

// 获取求解器名称
string GeneticSolver::getName() const {
    return "自适应遗传算法求解器";
}

// 主求解方法
CVRPSolution GeneticSolver::solve(const CVRPProblem& prob) {
    try {
        // 保存问题实例
        problem = prob;
        
        cout << "初始化算法..." << endl;
        // 初始化算法
        initialize();
        
        // 主循环
        for (int gen = 0; gen < max_generations; ++gen) {
            // 计算种群多样性
            population_diversity = calculatePopulationDiversity();
            
            // 更新自适应参数
            updateAdaptiveParameters(gen);
            
            // 创建新一代
            vector<Individual> new_population;
            
            // 应用精英策略
            if (elite_size > 0) {
                applyEliteStrategy();
            }
            
            // 生成新个体
            while (new_population.size() < population_size) {
                // 选择父代
                Individual parent1 = tournamentSelection(tournament_size);
                Individual parent2 = tournamentSelection(tournament_size);
                
                // 创建子代
                Individual offspring1, offspring2;
                
                // 应用交叉
                if (getRandomDouble(0.0, 1.0) < crossover_rate) {
                    // 随机选择交叉方法
                    if (getRandomDouble(0.0, 1.0) < 0.5) {
                        orderCrossover(parent1, parent2, offspring1, offspring2);
                    } else {
                        pathOrientedCrossover(parent1, parent2, offspring1, offspring2);
                    }
                } else {
                    // 不交叉，直接复制
                    offspring1 = parent1;
                    offspring2 = parent2;
                }
                
                // 应用变异
                if (getRandomDouble(0.0, 1.0) < mutation_rate) {
                    mutate(offspring1);
                }
                if (getRandomDouble(0.0, 1.0) < mutation_rate) {
                    mutate(offspring2);
                }
                
                // 评估适应度
                evaluateIndividual(offspring1, gen);
                evaluateIndividual(offspring2, gen);
                
                // 局部搜索（如果启用）
                if (use_local_search && getRandomDouble(0.0, 1.0) < 0.1) {
                    twoOptLocalSearch(offspring1);
                    evaluateIndividual(offspring1, gen); // 重新评估
                }
                if (use_local_search && getRandomDouble(0.0, 1.0) < 0.1) {
                    twoOptLocalSearch(offspring2);
                    evaluateIndividual(offspring2, gen); // 重新评估
                }
                
                // 添加到新种群
                new_population.push_back(offspring1);
                if (new_population.size() < population_size) {
                    new_population.push_back(offspring2);
                }
            }
            
            // 替换旧种群
            population = new_population;
            
            // 更新最佳和最差个体
            for (const auto& ind : population) {
                if (ind.fitness > best_individual.fitness) {
                    best_individual = ind;
                }
                if (ind.fitness < worst_individual.fitness) {
                    worst_individual = ind;
                }
            }
            
            // 输出进度（按照设定的显示间隔）
            if (gen % display_interval == 0 || gen == max_generations - 1) {
                cout << "第 " << gen << " 代，最佳适应度: " << best_individual.fitness 
                     << "，总距离: " << best_individual.total_distance 
                     << "，车辆数: " << best_individual.routes.size() 
                     << "，多样性: " << population_diversity << endl;
            }
        }
        
        // 构造最终解决方案
        CVRPSolution solution;
        solution.problem_name = problem.name;
        solution.total_distance = best_individual.total_distance;
        
        // 复制最佳个体的路径到解决方案中
        for (size_t i = 0; i < best_individual.routes.size(); ++i) {
            Route route = best_individual.routes[i];
            // 设置车辆ID从1开始
            route.vehicle_id = i + 1;
            solution.routes.push_back(route);
        }
        
        // 输出最终结果
        cout << "\n最终解决方案:" << endl;
        cout << "总距离: " << solution.total_distance << endl;
        cout << "车辆数量: " << solution.routes.size() << endl;
        
        return solution;
    }
    catch (const exception& e) {
        cout << "GeneticSolver::solve 发生异常: " << e.what() << endl;
        throw;
    }
    catch (...) {
        cout << "GeneticSolver::solve 发生未知异常!" << endl;
        throw;
    }
}

// 计算两点之间的欧几里得距离
double GeneticSolver::calculateDistance(const Point& a, const Point& b) const {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// 计算路径距离
double GeneticSolver::calculateRouteDistance(const vector<int>& route_points) const {
    if (route_points.size() < 2) return 0.0;  // 路径至少需要包含仓库和一个客户
    
    double total_distance = 0.0;
    
    // 计算路径上相邻点之间的距离
    for (size_t i = 0; i < route_points.size() - 1; ++i) {
        int current_id = route_points[i];    // 当前点ID（1是仓库，2到n+1是客户）
        int next_id = route_points[i + 1];   // 下一个点ID
        
        // 检查点ID的有效性
        if (current_id < 1 || current_id > static_cast<int>(problem.customers.size() + 1)) {
            throw runtime_error("Invalid point ID in route: " + to_string(current_id));
        }
        if (next_id < 1 || next_id > static_cast<int>(problem.customers.size() + 1)) {
            throw runtime_error("Invalid point ID in route: " + to_string(next_id));
        }
        
        // 获取当前点和下一个点的坐标
        // 如果ID为1，使用仓库坐标；否则，使用客户坐标（需要将ID转换为索引：ID-2）
        Point current_point = (current_id == 1) ? problem.depot : problem.customers[current_id - 2];
        Point next_point = (next_id == 1) ? problem.depot : problem.customers[next_id - 2];
        
        // 计算两点之间的距离
        total_distance += calculateDistance(current_point, next_point);
    }
    
    return total_distance;
}

// 初始化种群
void GeneticSolver::initialize() {
    try {
        cout << "开始初始化种群..." << endl;
        
        // 清空当前种群
        population.clear();
        
        // 创建客户ID列表（从0开始，对应customers数组中的索引）
        vector<int> customer_ids(problem.customers.size());
        iota(customer_ids.begin(), customer_ids.end(), 0);  // 填充0到n-1的序列，n为客户数量
        
        // 重置最佳个体
        best_individual = Individual();
        best_individual.fitness = -numeric_limits<double>::infinity();
        
        // 设置初始化最差个体
        worst_individual = Individual();
        worst_individual.fitness = numeric_limits<double>::infinity();
        
        // 生成初始种群
        for (size_t i = 0; i < population_size; ++i) {
            Individual ind;
            
            // 复制客户ID列表并随机打乱
            ind.chromosome = customer_ids;
            shuffle(ind.chromosome.begin(), ind.chromosome.end(), rng);
            
            // 评估个体
            evaluateIndividual(ind, 0);
            
            // 添加到种群
            population.push_back(ind);
            
            // 更新最佳个体
            if (ind.fitness > best_individual.fitness) {
                best_individual = ind;
            }
            
            // 更新最差个体
            if (ind.fitness < worst_individual.fitness) {
                worst_individual = ind;
            }
        }
        
        // 计算初始种群多样性
        population_diversity = calculatePopulationDiversity();
        
        // 重置交叉率和变异率为初始值，用于自适应调整
        crossover_rate = initial_crossover_rate;
        mutation_rate = initial_mutation_rate;
        
        cout << "初始化完成! 初始多样性: " << population_diversity << endl;
    }
    catch (const exception& e) {
        cout << "初始化过程中发生异常: " << e.what() << endl;
        throw;
    }
    catch (...) {
        cout << "初始化过程中发生未知异常!" << endl;
        throw;
    }
}

// 评估个体适应度
void GeneticSolver::evaluateIndividual(Individual& ind, int current_generation) {
    // 解码染色体获取路径
    double current_overload_tolerance = initial_overload_tolerance * 
        pow(1.0 - overload_penalty_increase_rate, current_generation);
    ind.routes = decodeChromosome(ind.chromosome, current_overload_tolerance);
    
    // 计算总距离和超载惩罚
    ind.total_distance = 0.0;
    ind.total_overload = 0.0;
    ind.num_routes = static_cast<int>(ind.routes.size());
    
    for (const Route& route : ind.routes) {
        ind.total_distance += route.total_distance;
        double overload = route.total_demand - problem.vehicle_capacity;
        if (overload > 0) {
            ind.total_overload += overload;
        }
    }
    
    // 计算惩罚值
    ind.overload_penalty = ind.total_overload * overload_penalty_factor;
    
    // 计算路径数量惩罚
    double route_penalty = 0.0;
    if (ind.num_routes > max_vehicles) {
        route_penalty = (ind.num_routes - max_vehicles) * route_penalty_factor;
    }
    
    // 计算适应度值（最小化问题转换为最大化问题）
    ind.fitness = 1.0 / (ind.total_distance + ind.overload_penalty + route_penalty + 1.0);
}

// 解码染色体为路径
vector<Route> GeneticSolver::decodeChromosome(const vector<int>& chromosome, double overload_tolerance) const {
    vector<Route> routes;
    Route current_route;
    double current_load = 0.0;
    double capacity_with_tolerance = problem.vehicle_capacity * (1.0 + overload_tolerance);
    
    // 初始化第一条路径，从仓库开始（仓库ID为1）
    current_route.points.push_back(1);
    
    for (size_t i = 0; i < chromosome.size(); ++i) {
        int chromosome_idx = chromosome[i];  // 染色体中的索引（0到n-1）
        
        // 验证染色体索引的有效性
        if (chromosome_idx < 0 || chromosome_idx >= static_cast<int>(problem.customers.size())) {
            throw runtime_error("Invalid customer index: " + to_string(chromosome_idx));
        }
        
        // 转换为客户ID（客户ID从2开始）
        int customer_id = chromosome_idx + 2;
        
        // 获取客户需求（使用染色体索引访问customers数组）
        double demand = problem.customers[chromosome_idx].demand;
        
        // 如果添加当前客户会超过容量限制（考虑容忍度）
        if (current_load + demand > capacity_with_tolerance) {
            // 完成当前路径
            if (current_route.points.size() > 1) {  // 如果路径中有客户
                current_route.points.push_back(1);  // 返回仓库
                routes.push_back(current_route);
                // 开始新路径
                current_route.points.clear();
                current_route.points.push_back(1);  // 从仓库开始
                current_load = 0.0;
            }
        }
        
        // 添加客户到当前路径
        current_route.points.push_back(customer_id);
        current_load += demand;
    }
    
    // 添加最后一条路径（如果有的话）
    if (current_route.points.size() > 1) {  // 如果路径中有客户
        current_route.points.push_back(1);  // 返回仓库
        routes.push_back(current_route);
    }
    
    // 计算每条路径的距离和需求
    for (Route& route : routes) {
        route.total_distance = calculateRouteDistance(route.points);
        route.total_demand = 0.0;
        for (size_t i = 1; i < route.points.size() - 1; ++i) {  // 跳过起点和终点（仓库）
            int point_id = route.points[i];  // 这是客户ID（从2开始）
            int chromosome_idx = point_id - 2;  // 转换回染色体索引
            route.total_demand += problem.customers[chromosome_idx].demand;
        }
    }
    
    return routes;
}

// 锦标赛选择
Individual GeneticSolver::tournamentSelection(int tournament_size) {
    // 随机选择tournament_size个个体
    vector<reference_wrapper<Individual>> tournament;

    for (int i = 0; i < tournament_size; ++i) {
        int idx = getRandomInt(0, population_size - 1);
        tournament.push_back(ref(population[idx]));
    }

    // 找出锦标赛中适应度最高的个体
    auto best_it = max_element(tournament.begin(), tournament.end(),
                               [](const Individual &a, const Individual &b) {
                                   return a.fitness < b.fitness;
                               });

    return *best_it;
}

// 顺序交叉 (OX)
void GeneticSolver::orderCrossover(const Individual &parent1, const Individual &parent2,
                                   Individual &offspring1, Individual &offspring2)
{
    int n = parent1.chromosome.size();

    // 随机选择交叉点
    int start = getRandomInt(0, n - 1);
    int end = getRandomInt(0, n - 1);

    // 确保start <= end
    if (start > end)
    {
        swap(start, end);
    }

    // 初始化后代染色体
    offspring1.chromosome.resize(n, -1); // -1表示未填充
    offspring2.chromosome.resize(n, -1);

    // 复制交叉段
    for (int i = start; i <= end; ++i)
    {
        offspring1.chromosome[i] = parent1.chromosome[i];
        offspring2.chromosome[i] = parent2.chromosome[i];
    }

    // 填充剩余位置
    // 对于offspring1，从parent2中按顺序填充
    int j = (end + 1) % n; // 填充起点
    for (int i = 0; i < n; ++i)
    {
        int p_idx = (end + 1 + i) % n; // parent2中的索引
        int value = parent2.chromosome[p_idx];

        // 检查value是否已在offspring1中
        if (find(offspring1.chromosome.begin(), offspring1.chromosome.end(), value) == offspring1.chromosome.end())
        {
            // 找到下一个未填充位置
            while (offspring1.chromosome[j] != -1)
            {
                j = (j + 1) % n;
            }

            offspring1.chromosome[j] = value;
            j = (j + 1) % n;
        }
    }

    // 对于offspring2，从parent1中按顺序填充
    j = (end + 1) % n;
    for (int i = 0; i < n; ++i)
    {
        int p_idx = (end + 1 + i) % n;
        int value = parent1.chromosome[p_idx];

        if (find(offspring2.chromosome.begin(), offspring2.chromosome.end(), value) == offspring2.chromosome.end())
        {
            while (offspring2.chromosome[j] != -1)
            {
                j = (j + 1) % n;
            }

            offspring2.chromosome[j] = value;
            j = (j + 1) % n;
        }
    }
}

// 变异操作
void GeneticSolver::mutate(Individual &ind)
{
    if (getRandomDouble(0.0, 1.0) < mutation_rate)
    {
        // 随机选择变异类型
        int mutation_type = getRandomInt(0, 2);

        switch (mutation_type)
        {
        case 0:
            swapMutation(ind.chromosome);
            break;
        case 1:
            insertionMutation(ind.chromosome);
            break;
        case 2:
            inversionMutation(ind.chromosome);
            break;
        }
    }
}

// 交换变异
void GeneticSolver::swapMutation(vector<int> &chromosome)
{
    int n = chromosome.size();
    if (n < 2)
        return;

    int i = getRandomInt(0, n - 1);
    int j = getRandomInt(0, n - 1);

    // 确保i != j
    while (i == j)
    {
        j = getRandomInt(0, n - 1);
    }

    swap(chromosome[i], chromosome[j]);
}

// 插入变异
void GeneticSolver::insertionMutation(vector<int> &chromosome)
{
    int n = chromosome.size();
    if (n < 2)
        return;

    // 随机选择要移动的客户
    int from_idx = getRandomInt(0, n - 1);
    int to_idx = getRandomInt(0, n - 1);

    // 如果索引相同，不需要变异
    if (from_idx == to_idx)
        return;

    // 保存要移动的客户
    int customer = chromosome[from_idx];

    // 移除原位置的客户
    chromosome.erase(chromosome.begin() + from_idx);

    // 如果插入位置大于移除位置，需要调整
    if (to_idx > from_idx)
    {
        to_idx--;
    }

    // 在新位置插入客户
    chromosome.insert(chromosome.begin() + to_idx, customer);
}

// 反转变异
void GeneticSolver::inversionMutation(vector<int> &chromosome)
{
    int n = chromosome.size();
    if (n < 2)
        return;

    // 随机选择区间
    int start = getRandomInt(0, n - 1);
    int end = getRandomInt(0, n - 1);

    // 确保start < end
    if (start > end)
    {
        swap(start, end);
    }

    // 反转区间内的元素
    reverse(chromosome.begin() + start, chromosome.begin() + end + 1);
}

// 生成指定范围内的随机整数
int GeneticSolver::getRandomInt(int min, int max)
{
    uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

// 生成指定范围内的随机浮点数
double GeneticSolver::getRandomDouble(double min, double max)
{
    uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

// 更新自适应参数
void GeneticSolver::updateAdaptiveParameters(int current_generation) {
    // 计算当前代数的进度 (0.0 - 1.0)
    double progress = static_cast<double>(current_generation) / max_generations;
    
    // 更新交叉率 - 从初始值线性减少到最终值
    crossover_rate = initial_crossover_rate - progress * (initial_crossover_rate - final_crossover_rate);
    
    // 更新变异率 - 从初始值线性减少到最终值
    mutation_rate = initial_mutation_rate - progress * (initial_mutation_rate - final_mutation_rate);
    
    // 基于种群多样性进行调整
    // 如果多样性低，增加变异率以提高多样性
    if (population_diversity < 0.3) {
        mutation_rate = min(mutation_rate * 1.5, 0.5); // 增加变异率但不超过0.5
    }
    // 如果多样性高，增加交叉率以加速收敛
    else if (population_diversity > 0.7) {
        crossover_rate = min(crossover_rate * 1.2, 0.95); // 增加交叉率但不超过0.95
    }
    
    // 确保参数在有效范围内
    crossover_rate = max(0.5, min(0.95, crossover_rate));
    mutation_rate = max(0.05, min(0.5, mutation_rate));
}

// 计算种群多样性
// 种群多样性是基于染色体的汉明距离（相异位置的数量）计算的
double GeneticSolver::calculatePopulationDiversity() {
    if (population.size() <= 1) return 0.0;
    
    double total_distance = 0.0;
    int count = 0;
    int chromosome_length = population[0].chromosome.size();
    
    // 随机采样一部分个体对计算多样性（当种群较大时提高效率）
    int sample_size = min(20, (int)population.size()); // 最多采样20个个体
    vector<int> sample_indices;
    
    // 生成不重复的随机索引
    while (sample_indices.size() < sample_size) {
        int idx = getRandomInt(0, population.size() - 1);
        if (find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
            sample_indices.push_back(idx);
        }
    }
    
    // 计算采样个体间的平均差异
    for (size_t i = 0; i < sample_indices.size(); ++i) {
        for (size_t j = i + 1; j < sample_indices.size(); ++j) {
            // 计算两个染色体之间的汉明距离
            int diff_count = 0;
            for (int k = 0; k < chromosome_length; ++k) {
                if (population[sample_indices[i]].chromosome[k] != population[sample_indices[j]].chromosome[k]) {
                    diff_count++;
                }
            }
            
            // 归一化距离（除以染色体长度）
            double normalized_distance = static_cast<double>(diff_count) / chromosome_length;
            total_distance += normalized_distance;
            count++;
        }
    }
    
    // 返回平均多样性
    return count > 0 ? total_distance / count : 0.0;
}

// 应用精英策略
// 将最差的个体替换为历史最佳个体，确保优秀解不会丢失
void GeneticSolver::applyEliteStrategy() {
    // 如果种群中已经有与最佳个体相同适应度的个体，则不需要替换
    bool has_best = false;
    for (const auto& ind : population) {
        if (fabs(ind.fitness - best_individual.fitness) < 1e-6) {
            has_best = true;
            break;
        }
    }
    
    // 如果没有最佳个体，则替换最差个体
    if (!has_best && !population.empty()) {
        // 找到最差个体的索引
        int worst_index = 0;
        double worst_fitness = population[0].fitness;
        
        for (size_t i = 1; i < population.size(); ++i) {
            if (population[i].fitness < worst_fitness) {
                worst_fitness = population[i].fitness;
                worst_index = i;
            }
        }
        
        // 用历史最佳个体替换最差个体
        population[worst_index] = best_individual;
    }
}

// 路径导向交叉
void GeneticSolver::pathOrientedCrossover(const Individual& parent1, const Individual& parent2,
                                         Individual& offspring1, Individual& offspring2) {
    // 选择解码后的路径，而不是直接使用染色体
    vector<Route> routes_parent1 = decodeChromosome(parent1.chromosome, initial_overload_tolerance);
    vector<Route> routes_parent2 = decodeChromosome(parent2.chromosome, initial_overload_tolerance);
    
    // 清空后代染色体
    offspring1.chromosome.clear();
    offspring2.chromosome.clear();
    
    // 从父代1中随机选择一部分路径传递给后代1
    int num_routes_to_inherit1 = max(1, (int)(routes_parent1.size() * getRandomDouble(0.3, 0.7)));
    vector<int> selected_route_indices1;
    
    // 随机选择路径索引
    while (selected_route_indices1.size() < num_routes_to_inherit1 && selected_route_indices1.size() < routes_parent1.size()) {
        int route_idx = getRandomInt(0, routes_parent1.size() - 1);
        if (find(selected_route_indices1.begin(), selected_route_indices1.end(), route_idx) == selected_route_indices1.end()) {
            selected_route_indices1.push_back(route_idx);
        }
    }
    
    // 从父代2中随机选择一部分路径传递给后代2
    int num_routes_to_inherit2 = max(1, (int)(routes_parent2.size() * getRandomDouble(0.3, 0.7)));
    vector<int> selected_route_indices2;
    
    // 随机选择路径索引
    while (selected_route_indices2.size() < num_routes_to_inherit2 && selected_route_indices2.size() < routes_parent2.size()) {
        int route_idx = getRandomInt(0, routes_parent2.size() - 1);
        if (find(selected_route_indices2.begin(), selected_route_indices2.end(), route_idx) == selected_route_indices2.end()) {
            selected_route_indices2.push_back(route_idx);
        }
    }
    
    // 记录已经添加到后代1的客户
    set<int> added_customers1;
    
    // 将选定路径中的客户添加到后代1
    for (int route_idx : selected_route_indices1) {
        const Route& route = routes_parent1[route_idx];
        for (size_t i = 1; i < route.points.size() - 1; ++i) { // 跳过起点和终点（仓库）
            int point_id = route.points[i];
            int customer_idx = point_id - 2;  // 从ID转回索引
            offspring1.chromosome.push_back(customer_idx);
            added_customers1.insert(customer_idx);
        }
    }
    
    // 记录已经添加到后代2的客户
    set<int> added_customers2;
    
    // 将选定路径中的客户添加到后代2
    for (int route_idx : selected_route_indices2) {
        const Route& route = routes_parent2[route_idx];
        for (size_t i = 1; i < route.points.size() - 1; ++i) { // 跳过起点和终点（仓库）
            int point_id = route.points[i];
            int customer_idx = point_id - 2;  // 从ID转回索引
            offspring2.chromosome.push_back(customer_idx);
            added_customers2.insert(customer_idx);
        }
    }
    
    // 从父代2中获取未添加到后代1的客户
    for (int customer_idx : parent2.chromosome) {
        if (added_customers1.find(customer_idx) == added_customers1.end()) {
            offspring1.chromosome.push_back(customer_idx);
            added_customers1.insert(customer_idx);
        }
    }
    
    // 从父代1中获取未添加到后代2的客户
    for (int customer_idx : parent1.chromosome) {
        if (added_customers2.find(customer_idx) == added_customers2.end()) {
            offspring2.chromosome.push_back(customer_idx);
            added_customers2.insert(customer_idx);
        }
    }
}

// 局部搜索 - 2-opt算法
void GeneticSolver::twoOptLocalSearch(Individual& ind) {
    // 解码染色体获取路径
    vector<Route> routes = ind.routes;
    bool improved = false;
    
    // 对每条路径进行2-opt优化
    for (auto& route : routes) {
        if (route.points.size() <= 4) continue; // 至少需要4个点才能进行2-opt（仓库-客户-客户-仓库）
        
        // 反转路径中的两个点之间的段落，尝试改进
        for (size_t i = 1; i < route.points.size() - 2; ++i) {
            for (size_t j = i + 1; j < route.points.size() - 1; ++j) {
                if (apply2OptMove(route.points, i, j)) {
                    improved = true;
                    // 更新路径距离
                    route.total_distance = calculateRouteDistance(route.points);
                }
            }
        }
    }
    
    // 如果有改进，更新个体的染色体
    if (improved) {
        // 重新构造染色体序列
        vector<int> new_chromosome;
        for (const auto& route : routes) {
            for (size_t i = 1; i < route.points.size() - 1; ++i) {  // 跳过起点和终点（仓库）
                int point_id = route.points[i];  // 这是客户ID（从2开始）
                int chromosome_idx = point_id - 2;  // 转换回染色体索引
                new_chromosome.push_back(chromosome_idx);
            }
        }
        
        // 更新个体
        ind.chromosome = new_chromosome;
        ind.routes = routes;
        
        // 计算新的总距离
        double total_distance = 0.0;
        for (const auto& route : routes) {
            total_distance += route.total_distance;
        }
        ind.total_distance = total_distance;
    }
}

// 应用2-opt移动
// 通过反转路径中从i到j的段落来尝试减少总距离
bool GeneticSolver::apply2OptMove(vector<int>& route, int i, int j) {
    // 计算当前路径的距离
    double current_distance = calculateRouteDistance(route);
    
    // 创建临时路径并应用2-opt移动
    vector<int> new_route = route;
    reverse(new_route.begin() + i, new_route.begin() + j + 1);
    
    // 计算新路径的距离
    double new_distance = calculateRouteDistance(new_route);
    
    // 如果新路径更短，则接受这个改变
    if (new_distance < current_distance) {
        route = new_route;
        return true;
    }
    
    return false;
}

// 设置显示间隔
void GeneticSolver::setDisplayInterval(int interval) {
    if (interval > 0) {
        display_interval = interval;
    } else {
        display_interval = 50; // 默认值
    }
}
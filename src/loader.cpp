#include "../include/loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;


//读取文件的方法
CVRPProblem ProblemLoader::loadProblem(const string& filename) {
    CVRPProblem problem;
    ifstream file(filename);
    
    string line;
    string section = "";
    
    while (getline(file, line)) {
        // 跳过空行和注释行
        if (line.empty() || line[0] == '#') continue;
        
        // 处理节段标记
        if (line.find("NAME") != string::npos) {
            problem.name = line.substr(line.find(":") + 1);
           
            problem.name.erase(0, problem.name.find_first_not_of(" \t"));
            problem.name.erase(problem.name.find_last_not_of(" \t") + 1);
            continue;
        }
        
        // 处理节点数量
        if (line.find("DIMENSION") != string::npos) {
            string dim = line.substr(line.find(":") + 1);
            
            dim.erase(0, dim.find_first_not_of(" \t"));
            dim.erase(dim.find_last_not_of(" \t") + 1);
            // 客户点数量加1（包含仓库）
            int dimension = stoi(dim);
            problem.customers.reserve(dimension - 1); // 不包含仓库
            continue;
        }
        
        // 处理车辆容量
        if (line.find("CAPACITY") != string::npos) {
            string cap = line.substr(line.find(":") + 1);
            
            cap.erase(0, cap.find_first_not_of(" \t"));
            cap.erase(cap.find_last_not_of(" \t") + 1);
            problem.vehicle_capacity = stod(cap);
            continue;
        }
        
        // 读取到节点坐标
        if (line.find("NODE_COORD_SECTION") != string::npos) {
            section = "NODE_COORD";
            continue;
        }
        
        // 读取到需求
        if (line.find("DEMAND_SECTION") != string::npos) {
            section = "DEMAND";
            continue;
        }
        
        // 读取到仓库
        if (line.find("DEPOT_SECTION") != string::npos) {
            section = "DEPOT";
            continue;
        }
        
        if (line.find("EOF") != string::npos) {
            break;
        }
        
        // 解析点坐标
        if (section == "NODE_COORD") {
            parseDataSection(line, problem);
        }
        // 解析点需求
        else if (section == "DEMAND") {
            parseDemandSection(line, problem);
        }
        // 确认仓库的id（1 指示仓库id，-1 表示结束）
        else if (section == "DEPOT") {
            parseDepotSection(line, problem);
        }
    }
    
    file.close();
    
    // 确定最佳车辆数量（这只是初始估计，可以根据实际情况调整）
    double total_demand = 0.0;
    for (const auto& customer : problem.customers) {
        total_demand += customer.demand;
    }
    problem.vehicle_num = ceil(total_demand / problem.vehicle_capacity);
    
    return problem;
}

//文件里面默认仓库的id为1，直接区分仓库和客户即可
void ProblemLoader::parseDataSection(const string& line, CVRPProblem& problem) {
    istringstream iss(line);
    int id;
    double x, y;
    
    if (iss >> id >> x >> y) {
        // 先添加所有点，之后再区分仓库和客户
        Point point(id, x, y, 0.0);
        
        if (id == 1) {
            // 通常仓库的ID为1，先暂存
            problem.depot = point;
        } else {
            // 客户点
            problem.customers.push_back(point);
        }
    }
}

void ProblemLoader::parseDemandSection(const string& line, CVRPProblem& problem) {
    istringstream iss(line);
    int id;
    double demand;
    
    if (iss >> id >> demand) {
        
            // 查找对应的客户点并更新需求
            for (auto& customer : problem.customers) {
                if (customer.id == id) {
                    customer.demand = demand;
                    break;
                }
            
        }
    }
}

void ProblemLoader::parseDepotSection(const string& line, CVRPProblem& problem) {
    istringstream iss(line);
    int id;
    
    if (iss >> id && id != -1) {
        // 设置仓库ID
       if(id == 1){
        problem.depot.id = id;
       }
       //否则应该修改仓库的id 以及删除数组中对应的id，将1加进去
       else{
        problem.depot.id = id;
        for(auto it = problem.customers.begin(); it != problem.customers.end(); ++it){
            if(it->id == id){
                problem.customers.erase(it);
                break;
            }
        }
        problem.customers.insert(problem.customers.begin(), Point(1, problem.depot.x, problem.depot.y, 0.0));
       }

    }
}

bool ProblemLoader::saveSolution(const CVRPSolution& solution, const string& filename) {
    string actual_filename = filename;
    int counter = 1;
    
    // 检查文件是否存在，如果存在则重命名
    while (ifstream(actual_filename)) {
        // 获取文件名和扩展名
        size_t dot_pos = filename.find_last_of(".");
        string name = filename.substr(0, dot_pos);
        string ext = filename.substr(dot_pos);
        
        // 构造新文件名
        actual_filename = name + "(" + to_string(counter) + ")" + ext;
        counter++;
    }
    
    ofstream file(actual_filename);
    
    if (!file.is_open()) {
        cerr << "无法创建解决方案文件: " << actual_filename << endl;
        return false;
    }
    
    file << "NAME : " << solution.problem_name << endl;
    file << "TOTAL_DISTANCE : " << solution.total_distance << endl;
    file << "ROUTES : " << solution.routes.size() << endl;
    
    // 输出每条路径
    for (const auto& route : solution.routes) {
        file << "ROUTE_" << route.vehicle_id << " : ";
        
        // 路径上的点
        for (size_t i = 0; i < route.points.size(); ++i) {
            file << route.points[i];
            if (i < route.points.size() - 1) {
                file << " ";
            }
        }
        
        file << endl;
    }
    
    cout << "解决方案已保存到: " << actual_filename << endl;
    
    file.close();
    return true;
}

bool ProblemLoader::validateSolution(const CVRPSolution& solution, const CVRPProblem& problem) {
    // 检查是否所有客户都被访问
    vector<bool> visited(problem.customers.size() + 1, false);
    
    // 检查容量约束和路径有效性
    for (const auto& route : solution.routes) {
        double total_demand = 0.0;
        int prev_point = problem.depot.id;
        
        for (size_t i = 0; i < route.points.size(); ++i) {
            int current_point = route.points[i];
            
            // 跳过起始和结束的仓库点
            if (current_point == problem.depot.id) {
                if (i > 0 && i < route.points.size() - 1) {
                    // 如果路径中途有仓库点（不是开始或结束），这可能是错误的
                    return false;
                }
                continue;
            }
            
            // 标记客户为已访问
            for (const auto& customer : problem.customers) {
                if (customer.id == current_point) {
                    visited[current_point] = true;
                    total_demand += customer.demand;
                    break;
                }
            }
        }
        
        // 检查车辆容量约束
        if (total_demand > problem.vehicle_capacity) {
            return false;
        }
    }
    
    // 确保所有客户都被访问
    for (size_t i = 0; i < problem.customers.size(); ++i) {
        if (!visited[problem.customers[i].id]) {
            return false;
        }
    }
    
    return true;
} 
#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>

using namespace std;

// 点的结构定义，表示客户或仓库的位置
struct Point {
    int id;         
    double x;       
    double y;       
    double demand;  
    
    Point() : id(0), x(0), y(0), demand(0) {}
    Point(int _id, double _x, double _y, double _demand) : id(_id), x(_x), y(_y), demand(_demand) {}
};

// 路径结构，表示一辆车的访问顺序
struct Route {
    int vehicle_id;          
    vector<int> points;  // 访问点的顺序（点的ID）
    double total_distance;   
    double total_demand;      
    
    Route() : vehicle_id(0), total_distance(0), total_demand(0) {}
    Route(int _id) : vehicle_id(_id), total_distance(0), total_demand(0) {}
};

// CVRP问题的结构，包含问题所有信息
struct CVRPProblem {
    string name;             // 问题名称
    int vehicle_num;              // 车辆数量
    double vehicle_capacity;      // 车辆容量
    Point depot;                  // 记录仓库点
    vector<Point> customers; // 所有客户点
    
    CVRPProblem() : vehicle_num(0), vehicle_capacity(0) {}
};

// CVRP问题的解决方案
struct CVRPSolution {
    string problem_name;     // 问题名称
    double total_distance;        // 总距离
    vector<Route> routes;    // 所有路径
    
    CVRPSolution() : total_distance(0) {}
};

#endif // TYPES_H 
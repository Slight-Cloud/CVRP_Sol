#ifndef LOADER_H
#define LOADER_H

#include "types.h"
#include <string>

using namespace std;

// 问题加载器，负责从文件读取CVRP问题和保存解决方案
class ProblemLoader {
public:
    // 从文件加载CVRP问题
    static CVRPProblem loadProblem(const string& filename);
    
    // 将解决方案保存到文件
    static bool saveSolution(const CVRPSolution& solution, const string& filename);
    
    // 验证解决方案的有效性
    static bool validateSolution(const CVRPSolution& solution, const CVRPProblem& problem);
    
private:
    // 用于内部处理数据的辅助函数
    static void parseDataSection(const string& line, CVRPProblem& problem);
    static void parseDemandSection(const string& line, CVRPProblem& problem);
    static void parseDepotSection(const string& line, CVRPProblem& problem);
};

#endif // LOADER_H 
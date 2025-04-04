cmake_minimum_required(VERSION 3.10)
project(CVRP_Solution)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 生成遗传算法求解器可执行文件
add_executable(cvrp_genetic
    src/main_genetic.cpp
    src/genetic_solver.cpp
    src/loader.cpp
)

# 添加编译选项
if(MSVC)
    target_compile_options(cvrp_genetic PRIVATE /W4)
else()
    target_compile_options(cvrp_genetic PRIVATE -Wall -Wextra -pedantic)
endif()

# 创建必要的目录
add_custom_command(
    TARGET cvrp_genetic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/results"
    COMMENT "Creating data and results directories"
)

# 复制数据文件到构建目录（如果存在）
if(EXISTS "${PROJECT_SOURCE_DIR}/data")
    add_custom_command(
        TARGET cvrp_genetic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_SOURCE_DIR}/data"
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data"
        COMMENT "Copying data files to build directory"
    )
endif()

# 创建运行脚本
if(WIN32)
    # Windows批处理文件
    file(WRITE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run.bat
        "@echo off\n"
        "chcp 65001 > nul\n"
        "echo 正在运行CVRP求解程序...\n"
        "cvrp_genetic.exe\n"
        "echo.\n"
        "echo 程序运行完成，按任意键退出...\n"
        "pause > nul\n"
    )
else()
    # Linux/Mac shell脚本
    file(WRITE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run.sh
        "#!/bin/bash\n"
        "echo '正在运行CVRP求解程序...'\n"
        "./cvrp_genetic\n"
        "echo '程序运行完成，按Enter键退出...'\n"
        "read -p ''\n"
    )
    # 设置可执行权限
    execute_process(
        COMMAND chmod +x ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run.sh
    )
endif()
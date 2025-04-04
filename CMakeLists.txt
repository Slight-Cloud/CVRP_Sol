cmake_minimum_required(VERSION 3.10)
project(CVRP_Solution)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找所有源文件
file(GLOB COMMON_SOURCES 
    "src/loader.cpp"
)

# 包含头文件目录
include_directories(include)

# 生成原始求解器可执行文件
add_executable(cvrp_solver ${COMMON_SOURCES} "src/solver.cpp" "src/main.cpp")

# 生成遗传算法求解器可执行文件
add_executable(cvrp_genetic ${COMMON_SOURCES} "src/genetic_solver.cpp" "src/main_genetic.cpp")

# 如果需要额外链接库，可以在这里添加
# target_link_libraries(cvrp_solver ${EXTRA_LIBS})
# target_link_libraries(cvrp_genetic ${EXTRA_LIBS})

# 添加编译选项
if(MSVC)
    target_compile_options(cvrp_solver PRIVATE /W4)
    target_compile_options(cvrp_genetic PRIVATE /W4)
else()
    target_compile_options(cvrp_solver PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(cvrp_genetic PRIVATE -Wall -Wextra -pedantic)
endif()

# 添加一个自定义目标，自动创建data目录用于存放问题数据
add_custom_command(
    TARGET cvrp_solver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/data"
    COMMENT "Creating data directory"
) 
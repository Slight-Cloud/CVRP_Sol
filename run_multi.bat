@echo off
setlocal enabledelayedexpansion

rem Set console to UTF-8
chcp 65001 > nul

echo ===========================================
echo          CVRP Genetic Algorithm
echo ===========================================

rem Create results folder
if not exist "k" mkdir k

rem Get number of runs
set /p RUN_TIMES=Enter number of runs: 

rem Problem parameters
set "PROBLEM_FILE=data\Vrp-Set-A\A\A-n33-k5.vrp"
set "OUTPUT_FILE=results\A-n32-k5.sol"

rem Algorithm parameters


echo Parameters:
echo Problem file: !PROBLEM_FILE!
echo Output file: !OUTPUT_FILE!
echo Population size: !POPULATION_SIZE!
echo Max generations: !MAX_GENERATIONS!
echo Crossover rate: !CROSSOVER_RATE!
echo Mutation rate: !MUTATION_RATE!
echo Number of runs: !RUN_TIMES!
echo.

echo Starting genetic algorithm...
echo.

rem Create summary file
echo Run    Distance    Vehicles > k\summary.txt

rem Initialize total distance
set /a total_distance=0

rem Run multiple times
for /l %%i in (1,1,!RUN_TIMES!) do (
    echo Run %%i...
    
    rem Run program and save output
    genetic_solver.exe "!PROBLEM_FILE!" "!OUTPUT_FILE!" !POPULATION_SIZE! !MAX_GENERATIONS! !CROSSOVER_RATE! !MUTATION_RATE! > "k\run_%%i.txt"
    
    rem Process output file line by line
    set "found_distance="
    for /f "usebackq tokens=1,2 delims=:" %%a in ("k\run_%%i.txt") do (
        set "line=%%a"
        set "value=%%b"
        if "!line!"=="总距离" (
            if not defined found_distance (
                set "curr_distance=!value!"
                set "curr_distance=!curr_distance: =!"
                set "curr_distance=!curr_distance:.=!"
                set /a curr_distance_int=!curr_distance:~0,-3!
                echo Run %%i distance: !curr_distance_int!
                set /a total_distance+=!curr_distance_int!
                set "found_distance=1"
            )
        )
        if "!line!"=="车辆数量" (
            set "curr_vehicles=!value!"
            set "curr_vehicles=!curr_vehicles: =!"
            echo %%i    !curr_distance_int!    !curr_vehicles! >> k\summary.txt
        )
    )
)

rem Calculate average distance
set /a avg_distance=!total_distance!/!RUN_TIMES!

echo.
echo ===============================
echo Summary:
echo Total runs: !RUN_TIMES!
echo Total distance: !total_distance!
echo Average distance: !avg_distance!
echo Results saved to k folder
echo ===============================

rem Add summary to file
echo. >> k\summary.txt
echo ============================== >> k\summary.txt
echo Total distance: !total_distance! >> k\summary.txt
echo Average distance: !avg_distance! >> k\summary.txt

pause 
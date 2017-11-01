g++ query_ball_point.cpp -o query_ball_point
nvcc query_ball_point.cu -o query_ball_point_cuda
nvcc query_ball_point_block.cu -o query_ball_point_block
nvcc query_ball_point_grid.cu -o query_ball_point_grid
g++ -Wall selection_sort.cpp -o selection_sort
nvcc selection_sort.cu -o selection_sort_cuda

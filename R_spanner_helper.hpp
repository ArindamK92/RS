#ifndef R_SPANNER_HELPER_HPP
#define R_SPANNER_HELPER_HPP
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <map>




/// @brief structure to store the graph in CSR format
struct CSR
{
    std::vector<int> data;    // edge weight
    std::vector<int> col_idx; // column indices (neighbor indices)
    std::vector<int> row_ptr; // row pointers (points the starting of neighbor list for each vertex)
};

/// @brief reads the arguments and assigns filename, comm_filename, target_communities
/// @param argc 
/// @param argv 
/// @param filename 
/// @param comm_filename 
/// @param target_communities 
/// @return 
int read_args(int argc, char **argv, std::string& filename, std::string& comm_filename, std::vector<int>& target_communities);

/// @brief reads the community id for each vertices. We consider the comm id starts from 0 and goes till max_comm_id.
/// @brief We consider there is at least one vertex in each community
/// @param filename
/// @param C
void readCommunity(std::string filename, std::vector<int> &C, /*int *max_comm_id,*/ std::unordered_map<int, int> comm_map);

/// @brief read the graph in .mtx format and store in CSR format
/// @param filename
/// @param n
/// @param m
/// @return retured the stored graph in CSR format
CSR mtxToCSR(const std::string &filename, /*std::vector<int> &degree,*/ int *n, int *m);

#endif
#ifndef PRINTER_HPP
#define PRINTER_HPP
#include <iostream>
#include <vector>
#include <string>
#include "R_spanner_helper.hpp"

/// @brief prints the graph information
/// @param filename 
/// @param comm_filename 
/// @param n 
/// @param m 
/// @param total_community 
void print_graph_info(std::string filename, std::string comm_filename, int n, int m, int total_community);

/// @brief prints neighbors of vertices in G
/// @param csr 
/// @param C 
void print_neighbors(CSR& csr, std::vector<int> C);

/// @brief prints normalized weights of the edges in G'(V_B, E_B)
/// @param bv_total 
/// @param bv_edge_weights 
void print_normalized_weights(int bv_total, std::vector<double> bv_edge_weights);

/// @brief print neighbor community count for each vertices
/// @param comm_count_col_size 
/// @param comm_counts 
void print_ngbr_comm_count(int comm_count_col_size, std::vector<double> comm_counts);

/// @brief print neighbor community count for each vertices
/// @param comm_count_col_size 
/// @param comm_counts 
void print_ngbr_comm_count_new(int comm_count_col_size, std::vector<double> comm_counts, std::vector<int> bv);

#endif
#ifndef R_SPANNER_KERNELS_HPP
#define R_SPANNER_KERNELS_HPP
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

using namespace cl::sycl;

/// @brief 
/// @param q 
/// @param row_ptr_buf 
/// @param col_idx_buf 
/// @param bv_buf 
/// @param bv_id_buf 
/// @param C_buf 
/// @param count_buf 
/// @param n 
void filter_kernel(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &bv_id_buf, buffer<int, 1> &C_buf, buffer<int, 1> &count_buf, int n);

/// @brief 
/// @param q 
/// @param row_ptr_buf 
/// @param col_idx_buf 
/// @param bv_buf 
/// @param C_buf 
/// @param comm_counts_buf 
/// @param bv_total 
/// @param total_community 
void find_ngbr_comm_kernel(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &C_buf, buffer<double> &comm_counts_buf, int bv_total, int comm_count_col_size);

/// @brief Find the max possible predecessor count for each border vertices
/// @param q 
/// @param row_ptr_buf 
/// @param col_idx_buf 
/// @param bv_buf 
/// @param bv_id_buf 
/// @param bv_pred_count 
/// @param bv_total 
void find_bv_pred_count_kernel(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &bv_id_buf, buffer<int> &bv_pred_count_buf, buffer<int, 1> &C_buf, int bv_total);

/// @brief A parallel prefix sum to compute the row pointer for G'
/// @param in_buf 
/// @param out_buf 
/// @param q 
/// @param N 
void compute_row_ptr(buffer<int> &in_buf, buffer<int> &out_buf, queue q, int N);

/// @brief Fill column indexes in the CSR of G'
/// @param q 
/// @param row_ptr_buf 
/// @param col_idx_buf 
/// @param bv_buf 
/// @param bv_id_buf 
/// @param C_buf 
/// @param bv_total 
/// @param row_ptr_Gb_buf 
/// @param col_idx_Gb_buf 
void fill_col_idx(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &bv_id_buf, buffer<int, 1> &C_buf, int bv_total, buffer<int, 1> &row_ptr_Gb_buf, buffer<int, 1> &col_idx_Gb_buf);

/// @brief Compute edge weights of G'
/// @param q 
/// @param comm_counts_buf 
/// @param bv_total 
/// @param comm_count_col_size 
void compute_edge_weights(queue q, buffer<double> &comm_counts_buf, int bv_total, int comm_count_col_size);

/// @brief It considers that the actual ids in the neighbor list of border vertices (in G') are stored in ascending order.
/// NOTE: the bv local IDs may not be in ascending order in the neighborlist of G'
/// It finds the intersection of neighbors using a merge technique
/// @param q 
/// @param row_ptr_Gb_buf 
/// @param col_idx_Gb_buf 
/// @param comm_counts_buf 
/// @param C_buf 
/// @param bv_buf 
/// @param R_buf 
/// @param degree_buf 
/// @param bv_total 
/// @param comm_count_col_size 
void compute_score_MergeIntersection(queue q, buffer<int> &row_ptr_Gb_buf, buffer<int> &col_idx_Gb_buf, buffer<double> &comm_counts_buf, buffer<int> &C_buf, buffer<int> &bv_buf, buffer<double> &R_buf, buffer<int> &degree_buf, int bv_total, int comm_count_col_size);


#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <CL/sycl.hpp>
#include <cmath>
#include <numeric>
#include "R_spanner_helper.hpp"
#include "printer.hpp"
#include "R_spanner_kernels.hpp"

using namespace cl::sycl;

/// @brief main function accepts -g <graphFile.mtx> -c <communityFile> -t <target community id> -t <target community id> ...
/// @param argc
/// @param argv
/// @return
int main(int argc, char **argv)
{
    std::string filename;
    std::string comm_filename;
    int max_comm_id = 0, total_community = 0; // We consider the comm id starts from 0 and goes till max_comm_id.
    int n, m;                                 // total vertices n, total edges m
    int bv_total = 0;                         // stores total border vertices
    double maxWeight = 0.0;                   // stores max edge weight among bv-bv edges
    std::vector<int> target_communities;

    // An exception handler for SYCL asynchronous error
    auto exception_handler = [](exception_list exceptions)
    {
        for (std::exception_ptr const &e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (cl::sycl::exception const &e)
            {
                std::cout << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
            }
        }
    };

    read_args(argc, argv, filename, comm_filename, target_communities);

    // Map the targeted community ids to contiguous local community ids starting from 0. (0 for all non-targeted comm. non zero for targeted comm)
    std::unordered_map<int, int> comm_map; // comm_map stores a local comm id (starting from 1) for the targeted community
    int c_id = 1;
    for (auto c : target_communities)
    {
        std::cout << c << " : ";
        comm_map[c] = c_id;
        c_id++;
    }
    std::cout << "\n";
    total_community = c_id; // we include comm id 0 for all vertices for which the community was not targeted

    CSR csr = mtxToCSR(filename, &n, &m);                        // Read the graph in .mtx format and store in CSR format
    std::vector<int> C(n);                                       // stores community of each vertex
    std::vector<double> R(n, 0.0);                               // Stores R-score for all the vertices
    readCommunity(comm_filename, C, /*&max_comm_id,*/ comm_map); // Read and store community for each vertex

    print_graph_info(filename, comm_filename, n, m, total_community);

    // store the degree
    std::vector<int> degree; // stores the degree of each vertex
    for (int i = 0; i < n; i++)
    {
        int x = csr.row_ptr[i + 1] - csr.row_ptr[i];
        degree.push_back(x);
    }

    std::vector<int> bv(n, -1); // list of border vertices (stores actual vertex id)
    // bv_id value is non-negative if the vertex is a border vertex.
    // If bv_id for a vertex is i, then the vertex is the ith border vertex in bv list.
    // bv_id is used as map between actual vertex id and border vertex id.
    std::vector<int> bv_id(n, -1); // stores local bv id
    std::vector<int> bv_count(1);
    bv_count[0] = 0;
    try
    {
        queue q(exception_handler);
        buffer<int> row_ptr_buf(csr.row_ptr.data(), csr.row_ptr.size());
        buffer<int> col_idx_buf(csr.col_idx.data(), csr.col_idx.size());
        buffer<int, 1> bv_buf(bv.data(), range<1>(n));
        buffer<int, 1> bv_id_buf(bv_id.data(), range<1>(n));
        buffer<int, 1> C_buf(C.data(), range<1>(n));
        buffer<int, 1> count_buf(bv_count.data(), range<1>(1));
        buffer<double> maxBuf{&maxWeight, 1};

        auto start_time = std::chrono::high_resolution_clock::now();

        // step 1: Filter the border vertices.
        filter_kernel(q, row_ptr_buf, col_idx_buf, bv_buf, bv_id_buf, C_buf, count_buf, n);
        bv_total = count_buf.get_host_access()[0];

        auto end_time_s1 = std::chrono::high_resolution_clock::now();
        auto elapsed_s1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_s1 - start_time).count();
        std::cout << "Total border vertices: " << bv_total << "\n"; // **Exclude it from timing
        std::cout << "Time taken (step 1): " << elapsed_s1 << "ms" << std::endl;

        auto start_time_s21 = std::chrono::high_resolution_clock::now();

        // step 2A: Find the communities connected to each bv and their count
        // comm_counts is a flattened 2D vector of size bv_count * (total_community+1).
        // element [i][j] in comm_counts stores the number of neighbors from community j (j < total_community) for ith border vertex.
        // the last element of each row, i.e., [i][total_community] stores the total neighbors of ith border vertex from the interested communities.
        std::vector<double> comm_counts;
        int comm_count_col_size = (total_community + 1);
        try
        {
            comm_counts.assign(bv_total * comm_count_col_size, 0);
        }
        catch (const std::bad_alloc &e)
        {
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        }
        {
            buffer<double> comm_counts_buf(comm_counts.data(), bv_total * comm_count_col_size);
            find_ngbr_comm_kernel(q, row_ptr_buf, col_idx_buf, bv_buf, C_buf, comm_counts_buf, bv_total, comm_count_col_size);

            // step 2B: Compute edge weights for G'
            compute_edge_weights(q, comm_counts_buf, bv_total, comm_count_col_size);
        }

        auto end_time_s21 = std::chrono::high_resolution_clock::now();
        auto elapsed_s21 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_s21 - start_time_s21).count();
        std::cout << "Time taken (step 2A, 2B): " << elapsed_s21 << "ms" << std::endl;

        auto start_time_s22 = std::chrono::high_resolution_clock::now();

        // // step 2C: Find max weight
        int size1 = (bv_total * comm_count_col_size + 255) / 256 * 256; // we need to round it for LLVM SYCL as Non-uniform work-groups are not supported
        comm_counts.resize(size1, 0);
        buffer<double> comm_counts_buf(comm_counts.data(), bv_total * comm_count_col_size);
        q.submit([&](handler &cgh)
                 {
        auto comm_counts_access = comm_counts_buf.get_access<access::mode::read>(cgh);
        auto maxReduction  = reduction(maxBuf, cgh, maximum<>());
        cgh.parallel_for<class find_max_weight_kernel>(nd_range<1>{size1, 256}, maxReduction ,
                        [=](nd_item<1> item, auto &max) {
                        int glob_id = item.get_global_id(0);
                        if ((glob_id + 1) % comm_count_col_size != 0){
                            max.combine(comm_counts_access[glob_id]);
                        }
                        }); });
        q.wait_and_throw();

        // Test: print max weight
        std::cout << "max weight (from reduction): " << maxBuf.get_host_access()[0] << "\n";

        // // step 2D: Divide all weights by the max weight (normalize)
        q.submit([&](handler &cgh)
                 {
        auto comm_counts_access = comm_counts_buf.get_access<access::mode::read_write>(cgh);
        auto max_access  = maxBuf.get_access<access::mode::read>(cgh);
        cgh.parallel_for<class normalize_weight_kernel>(
            range<1>(size1),
            [=](id<1> i) {
                comm_counts_access[i] /= max_access[0];
            }
        ); });
        q.wait();

        auto end_time_s22 = std::chrono::high_resolution_clock::now();
        auto elapsed_s22 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_s22 - start_time_s22).count();
        std::cout << "Time taken (step 2): " << elapsed_s22 << "ms" << std::endl;

        auto start_time_s3 = std::chrono::high_resolution_clock::now();
        // step 3: create G'(V_b. E_b)
        // step 3A: Find the max possible predecessor count for each border vertices
        // An existing edge (u,v) might not create a predecessor v for u if the edge weight \omega_v(u) = 0
        int size_ = bv_total + 1; // we take 1 extra element to find the last column idx easily
        std::vector<int> bv_pred_count(size_, 0);
        buffer<int> bv_pred_count_buf(bv_pred_count.data(), size_);
        find_bv_pred_count_kernel(q, row_ptr_buf, col_idx_buf, bv_buf, bv_id_buf, bv_pred_count_buf, C_buf, bv_total);

        // step 3B: create row ptr for G' using prefix sum on bv_pred_count
        std::vector<int> row_ptr_Gb(size_, 0);
        buffer<int> row_ptr_Gb_buf(row_ptr_Gb.data(), size_);
        compute_row_ptr(bv_pred_count_buf, row_ptr_Gb_buf, q, size_);

        // step 3C: create the column idx for G'
        auto host_row_ptr_Gb = row_ptr_Gb_buf.get_access<sycl::access::mode::read>(); // without host accessor we get wrong results in LLVM SYCL
        // auto host_bv_pred_count = bv_pred_count_buf.get_access<sycl::access::mode::read>();
        // int total_edges_Gb = host_row_ptr_Gb[bv_total - 1] + host_bv_pred_count[bv_total - 1]; // Total edges in G'(V_b, E_b)
        int total_edges_Gb = host_row_ptr_Gb[size_ - 1];
        std::vector<int> col_idx_Gb(total_edges_Gb, 0);

        // Test ***** make a plot out of it for different datasets *****
        std::cout << "total undirected edges * 2 in G: " << csr.col_idx.size() << std::endl; // total int units used to store edges of G //count can be less than |E| if there is both "a b" and "b a" in graph data
        std::cout << "total directed edges in G': " << total_edges_Gb << std::endl;          // total int units required to store edges of G'

        buffer<int> col_idx_Gb_buf(col_idx_Gb.data(), bv_total);
        fill_col_idx(q, row_ptr_buf, col_idx_buf, bv_buf, bv_id_buf, C_buf, bv_total, row_ptr_Gb_buf, col_idx_Gb_buf);

        auto end_time_s3 = std::chrono::high_resolution_clock::now();
        auto elapsed_s3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_s3 - start_time_s3).count();
        std::cout << "Time taken (step 3): " << elapsed_s3 << "ms" << std::endl;

        auto start_time_s4 = std::chrono::high_resolution_clock::now();

        // // step 4: Compute the R score
        buffer<double> R_buf(R.data(), n);
        buffer<int, 1> degree_buf(degree.data(), range<1>(n));
        compute_score_MergeIntersection(q, row_ptr_Gb_buf, col_idx_Gb_buf, comm_counts_buf, C_buf, bv_buf, R_buf, degree_buf, bv_total, comm_count_col_size);
        q.wait_and_throw();

        auto end_time_s4 = std::chrono::high_resolution_clock::now();
        auto elapsed_s4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_s4 - start_time_s4).count();
        std::cout << "Time taken (step 4): " << elapsed_s4 << "ms" << std::endl;

        auto elapsed = elapsed_s1 + elapsed_s21 + elapsed_s22 + elapsed_s3 + elapsed_s4;

        std::cout << "Time taken (total): " << elapsed << "ms" << std::endl;
    }
    catch (exception const &e)
    {
        std::cerr << "A SYCL exception occurred: " << e.what() << std::endl;
    }
    catch (std::exception const &e)
    {
        std::cerr << "A standard exception occurred: " << e.what() << std::endl;
    }


    // // Test: print R scores
    // std::cout << "R-scores: ";
    // int total_print = 10; // print this many values
    // int itr_t = 0;
    // for (int i = 0; i < n; ++i)
    // {
    //     if (bv_id[i] > -1)
    //     {
    //         std::cout << "R-score(" << i + 1 << ")[bv_id:" << bv_id[i] << "] = " << R[i] << " || \n"; // printing v_id as i+1 (1-indexed)
    //         itr_t++;
    //     }
    //     if (itr_t >= total_print)
    //     {
    //         break;
    //     }
    // }
    // std::cout << std::endl;
    return 0;
}

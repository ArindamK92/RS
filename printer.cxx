#include "printer.hpp"

void print_graph_info(std::string filename, std::string comm_filename, int n, int m, int total_community)
{
    std::cout << "File name: " << filename << std::endl;
    std::cout << "Comm name: " << comm_filename << std::endl;
    std::cout << "|V|: " << n << " |E|: " << m << "\n";
    std::cout << "Total community (targeted comm. + 1):" << total_community << "\n"; // All non targeted communities are considered as 1 community
}

void print_neighbors(CSR &csr, std::vector<int> C)
{
    for (int i = 0; i < 2; ++i)
    {
        std::cout << "vertex id: " << i + 1 << " comm: " << C[i] << "\n"; //printing 1-indexed vertex id
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; ++j)
        {
            int ngbr = csr.col_idx[j]; 
            std::cout << ngbr + 1 << " : (comm: " << C[ngbr] << ") "; //printing 1-indexed vertex id
        }
        std::cout << std::endl;
    }
}

// Works on adjacency matrix. It will be obsolete if CSR is used for G'
void print_normalized_weights(int bv_total, std::vector<double> bv_edge_weights)
{
    std::cout << "Normalized Weights: ";
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < bv_total; j++)
        {
            if (bv_edge_weights[i * bv_total + j] > 0)
            {
                std::cout << "weight[ " << i << "][" << j << "] = " << bv_edge_weights[i * bv_total + j] << "\n";
            }
        }
    }
    std::cout << std::endl;
}

void print_ngbr_comm_count(int comm_count_col_size, std::vector<double> comm_counts)
{
    std::cout << "Neighbor community count: ";
    for (int i = 0; i < 2; ++i)
    {
        std::cout << "vertex: " << i << "\n";
        for (int j = 0; j < comm_count_col_size; j++)
        {
            if(j < (comm_count_col_size - 1)){
                std::cout << "Comm: " << j << " count: " << comm_counts[i * comm_count_col_size + j] << " ";
            }
            else{
                std::cout << "total:: " << comm_counts[i * comm_count_col_size + j] << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void print_ngbr_comm_count_new(int comm_count_col_size, std::vector<double> comm_counts, std::vector<int> bv)
{
    std::cout << "Neighbor community count: ";
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "vertex: " << bv[i] + 1  << "\n"; //print vertex id in 1-indexed format
        for (int j = 0; j < comm_count_col_size; j++)
        {
            if(j < (comm_count_col_size - 1)){
                std::cout << "Comm: " << j << " count: " << comm_counts[i * comm_count_col_size + j] << " ";
            }
            else{
                std::cout << "total:: " << comm_counts[i * comm_count_col_size + j] << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}
#include "R_spanner_kernels.hpp"
#include <cmath>

/// @brief It filters out the community border vertices.
/// An effective border vertex should have at least 3 communities (it may include comm 0 or own comm also) connected to it.
/// If 2 or less communities are connected freq becomes 1 => H becomes 0 => edge weight becomes 0
/// @param q
/// @param row_ptr_buf
/// @param col_idx_buf
/// @param bv_buf
/// @param bv_id_buf
/// @param C_buf
/// @param count_buf
/// @param n
void filter_kernel(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &bv_id_buf, buffer<int, 1> &C_buf, buffer<int, 1> &count_buf, int n)
{
    q.submit([&](handler &cgh)
             {
        auto row_ptr_access = row_ptr_buf.get_access<access::mode::read>(cgh);
        auto col_idx_access = col_idx_buf.get_access<access::mode::read>(cgh);
        auto C_access = C_buf.get_access<access::mode::read>(cgh);
        auto bv_access = bv_buf.get_access<access::mode::write>(cgh);
        auto bv_id_access = bv_id_buf.get_access<access::mode::write>(cgh);
        auto count_access = count_buf.get_access<access::mode::atomic>(cgh);

        cgh.parallel_for<class kernel_filter>( //Can we optimize it more? 2steps, mark 1, then filter
            range<1>(n),
            [=](id<1> i) {
                int start_n = row_ptr_access[i];
                int stop_n = row_ptr_access[i+1];
                //int distinct_ngbr_comm = 0; //count of distinct neighbor communities other than its own community
                //int distinct_comm = -1;
                int own_comm = C_access[i]; //own community id
                int x = 0, y = 0; //stores the community in binary format. 101 means there are comm 0 and 2
                int count = 0;
                for (int j = start_n; j < stop_n; ++j){
                    int ngbr = col_idx_access[j];
                    int ngbr_comm = C_access[ngbr];
                    if(own_comm == 0){ // Added new 
                        break;
                    }                    
                    x = x | (1 << ngbr_comm); // (1 << num1) is equal to pow(2,num1)

                    //Brian Kernighan's algorithm to count 1 in binary representation of x
                    //If using it chaange "count" to "count_b" in the next if logic
                    // int count_b = 0;
                    // while (x) {
                    //     x &= (x - 1); // Flip the least significant bit set to 1
                    //     count_b++;
                    // }

                    //alternative of the while loop. As x changes whenever a new comm. is found (In such case x never takes an old value)
                    if(x != y){
                        count++;
                        y = x;
                    }

                    //If at least 3 neighbor communities found the vertex is an effective border vertex
                    if(count >= 3){
                        int idx = count_access[0].fetch_add(1);
                        bv_access[idx] = i;
                        bv_id_access[i] = idx;
                        break;
                    }
                }
            }
        ); });
    q.wait();
}

/// @brief it visits the neighbors of each border vertices and counts the number of neighbors from each community (count neighbor community kernel)
/// @param q
/// @param row_ptr_buf
/// @param col_idx_buf
/// @param bv_buf
/// @param C_buf
/// @param comm_counts_buf
/// @param bv_total
/// @param comm_count_col_size
void find_ngbr_comm_kernel(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &C_buf, buffer<double> &comm_counts_buf, int bv_total, int comm_count_col_size)
{
    q.submit([&](handler &cgh)
             {
        auto row_ptr_access = row_ptr_buf.get_access<access::mode::read>(cgh);
        auto col_idx_access = col_idx_buf.get_access<access::mode::read>(cgh);
        auto C_access = C_buf.get_access<access::mode::read>(cgh);
        auto bv_access = bv_buf.get_access<access::mode::read>(cgh);
        auto comm_counts_access = comm_counts_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class kernel_find_ngbr_comm>(
            range<1>(bv_total),
            [=](id<1> i) { //id i is bv local id
                int bv = bv_access[i]; //get actual vertex id from bv local id
                int start_n = row_ptr_access[bv];
                int stop_n = row_ptr_access[bv+1];
                int total = 0;
                int offset = i * comm_count_col_size;
                for (int j = start_n; j < stop_n; ++j){
                    int ngbr = col_idx_access[j];
                    int ngbr_comm = C_access[ngbr];
                    int flat_idx = offset + ngbr_comm;
                    comm_counts_access[flat_idx] += 1;
                    total += 1;
                }
                int last_idx = offset + comm_count_col_size - 1;
                comm_counts_access[last_idx] = total;
            }
        ); });
}

/// @brief Find the max possible predecessor count for each border vertices. If a neighbor is also a bv it can be a predecessor in G'.
/// @param q
/// @param row_ptr_buf
/// @param col_idx_buf
/// @param bv_buf
/// @param bv_id_buf
/// @param bv_pred_count
/// @param bv_total
void find_bv_pred_count_kernel(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &bv_id_buf, buffer<int> &bv_pred_count_buf, buffer<int, 1> &C_buf, int bv_total)
{
    q.submit([&](handler &cgh)
             {
        auto row_ptr_access = row_ptr_buf.get_access<access::mode::read>(cgh);
        auto col_idx_access = col_idx_buf.get_access<access::mode::read>(cgh);
        auto bv_access = bv_buf.get_access<access::mode::read>(cgh);
        auto bv_id_access = bv_id_buf.get_access<access::mode::read>(cgh);
        auto bv_pred_count_access = bv_pred_count_buf.get_access<access::mode::write>(cgh);
        auto C_access = C_buf.get_access<access::mode::read>(cgh);

        cgh.parallel_for<class kernel_find_bv_pred_count>(
            range<1>(bv_total),
            [=](id<1> i) {
                int vertex_id = bv_access[i]; // get actual vertex id from bv local id
                int start_n = row_ptr_access[vertex_id]; // start index of neighbor list in G
                int stop_n = row_ptr_access[vertex_id+1]; // end index of neighbor list in G
                int count = 0;
                for (int j = start_n; j < stop_n; ++j){
                    int ngbr = col_idx_access[j];
                    if(bv_id_access[ngbr] >= 0 && C_access[ngbr] != C_access[vertex_id]){ 
                        // a vertex has non-negative value (value is bv local id) if it is a bv
                        // a neighbor in same community is not included as they creates Type-II triads
                        count++;
                    }
                }
                bv_pred_count_access[i] = count;
            }
        ); });
    q.wait();
}

/// @brief A parallel prefix sum to compute the row pointer for G'
/// @param in_buf
/// @param out_buf
/// @param q
/// @param N
void compute_row_ptr(buffer<int> &in_buf, buffer<int> &out_buf, queue q, int N)
{
    int group_size = 512;

    q.submit([&](handler &cgh)
             {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<class kernel_name2>(nd_range<1>(group_size, group_size), [=](nd_item<1> it) { //don't use nd_range<1>(N, group_size): it increases time significantly
        group<1> g = it.get_group();
        joint_exclusive_scan(
            g, in.get_pointer(),
            in.get_pointer() + N,
            out.get_pointer(), plus<>());
      }); });
    q.wait();
}

/// @brief
/// @param q
/// @param row_ptr_buf
/// @param col_idx_buf
/// @param bv_buf
/// @param bv_id_buf
/// @param bv_pred_count_buf
/// @param C_buf
/// @param bv_total
/// @param row_ptr_Gb_buf
/// @param col_idx_Gb_buf
void fill_col_idx(queue q, buffer<int> &row_ptr_buf, buffer<int> &col_idx_buf, buffer<int, 1> &bv_buf, buffer<int, 1> &bv_id_buf, buffer<int, 1> &C_buf, int bv_total, buffer<int, 1> &row_ptr_Gb_buf, buffer<int, 1> &col_idx_Gb_buf)
{
    q.submit([&](handler &cgh)
             {
        auto row_ptr_access = row_ptr_buf.get_access<access::mode::read>(cgh);
        auto col_idx_access = col_idx_buf.get_access<access::mode::read>(cgh);
        auto bv_access = bv_buf.get_access<access::mode::read>(cgh);
        auto bv_id_access = bv_id_buf.get_access<access::mode::read>(cgh);
        auto C_access = C_buf.get_access<access::mode::read>(cgh);
        auto row_ptr_Gb_access = row_ptr_Gb_buf.get_access<access::mode::read>(cgh);
        auto col_idx_Gb_access = col_idx_Gb_buf.get_access<access::mode::write>(cgh);

        cgh.parallel_for<class fill_col_idx>(
            range<1>(bv_total),
            [=](id<1> i) {
                int vertex_id = bv_access[i]; // get actual vertex id from bv local id
                int start_n = row_ptr_access[vertex_id]; // start index of neighbor list in G
                int stop_n = row_ptr_access[vertex_id+1]; // end index of neighbor list in G
                int ngbr_idx = row_ptr_Gb_access[i]; // start index for the neighbors of bv_id i
                for (int j = start_n; j < stop_n; ++j){
                    int ngbr = col_idx_access[j];
                    int ngbr_bv_id = bv_id_access[ngbr];
                    if(ngbr_bv_id >= 0 && C_access[ngbr] != C_access[vertex_id]){
                        // a vertex has non-negative value (value is bv local id) if it is a bv
                        // a neighbor in same community is not included as they creates Type-II triads and Type=II triads are handled seperately
                        col_idx_Gb_access[ngbr_idx] = ngbr_bv_id;
                        ngbr_idx += 1;
                    }
                }
            }
        ); });
    q.wait();
}

/// @brief computes edge weights and store in the data structure comm_counts
/// @param q
/// @param comm_counts_buf
/// @param bv_total
/// @param comm_count_col_size
void compute_edge_weights(queue q, buffer<double> &comm_counts_buf, int bv_total, int comm_count_col_size)
{
    q.submit([&](handler &cgh)
             {
        //stream out(1024, 256, cgh);
        auto comm_counts_access = comm_counts_buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class compute_edge_weight_kernel>(
            range<1>(bv_total),
            [=](id<1> i) {
                int flat_idx = 0;
                int last_idx = 0; // last element of each row of comm_counts_access stores the total of comm. freq.
                double weight = 0.0;
                int offset = i * comm_count_col_size;
                int f_c = 0;
                double X_1 = 0.0, X_2 = 0.0;
                double Y = 0.0;
                int mod_L = 0;
                for (int c = 0; c < comm_count_col_size - 1; ++c){
                    flat_idx = offset + c;
                    f_c = comm_counts_access[flat_idx]; //frequency of community c
                    if (f_c != 0){
                        X_1 += (double)f_c  * std::log2((double)f_c);
                        X_2 += (double)f_c;
                        mod_L += 1;
                    }
                }
                for (int c = 0; c < comm_count_col_size - 1; ++c){
                    flat_idx = offset + c;
                    last_idx = offset + comm_count_col_size -1;
                    f_c = comm_counts_access[flat_idx]; //frequency of community c
                    Y = comm_counts_access[last_idx] - f_c; // total - freq[this comm.]
                    weight = 0.0;
                    if (f_c != 0 and Y != 0){
                        weight = (-1)* (X_1 - std::log2(Y) * X_2 - f_c * std::log2(f_c) + f_c * std::log2(Y)) / Y * (mod_L - 1);
                    }
                    if(weight > 0){
                        comm_counts_access[flat_idx] = weight;
                    }
                    else{
                        comm_counts_access[flat_idx] = 0.0;
                    }
                    
                }

            }); });
}


/// @brief It considers that the actual ids in the neighbor list of border vertices (in G') are stored in ascending order.
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
void compute_score_MergeIntersection(queue q, buffer<int> &row_ptr_Gb_buf, buffer<int> &col_idx_Gb_buf, buffer<double> &comm_counts_buf, buffer<int> &C_buf, buffer<int> &bv_buf, buffer<double> &R_buf, buffer<int> &degree_buf, int bv_total, int comm_count_col_size)
{
    q.submit([&](handler &cgh)
             {
        //stream out(1024, 256, cgh);
        auto row_ptr_Gb_access = row_ptr_Gb_buf.get_access<access::mode::read>(cgh);
        auto col_idx_Gb_access = col_idx_Gb_buf.get_access<access::mode::read>(cgh);
        auto comm_counts_access = comm_counts_buf.get_access<access::mode::read>(cgh);
        auto C_access = C_buf.get_access<access::mode::read>(cgh);
        auto bv_access = bv_buf.get_access<access::mode::read>(cgh);
        auto R_access = R_buf.get_access<access::mode::write>(cgh);
        auto degree_access = degree_buf.get_access<access::mode::write>(cgh);

        cgh.parallel_for<class compute_score_MergeIntersection>(
            range<1>(bv_total),
            [=](id<1> u) {
                int u_id = bv_access[u]; // get actual vertex id from bv local id
                int start_nu = row_ptr_Gb_access[u]; // start index of neighbor list of u in G'
                int stop_nu = row_ptr_Gb_access[u+1] - 1; // end index of neighbor list of u in G'
                double score = 0.0;
                int u_c = C_access[u_id]; // community id of vertex u
                double W_vu = 0.0, W_wu = 0.0, W_wv = 0.0;
                int begin_nu_ptr = start_nu;
                int end_nu_ptr = stop_nu;


                // Triad-I and -II combined
                for (int j = start_nu; j <= stop_nu; ++j){ // for w \in N_u : or for each edge (u,w)
                    int w = col_idx_Gb_access[j]; // bv local id of w (the neighbor vertex of u)
                    int w_id = bv_access[w]; // actual vertex id of w
                    int w_c = C_access[w_id]; // community id of vertex w
                    if(w_c == u_c){ // (u,w) should be the edge crossing community border
                        break;
                    }
                    int begin_nw = row_ptr_Gb_access[w]; // start index of neighbor list of w in G'
                    int end_nw = row_ptr_Gb_access[w+1] - 1; // end index of neighbor list of w in G'
                    int begin_nu = begin_nu_ptr; // reset begin_nu
                    int end_nu = end_nu_ptr; // reset end_nu
                    int comp1, comp2, nw_bound, nu_bound;

                    // Find the common neighbor vertices of w and u in O(deg_w + deg_u) time. 
                    //**We consider actual vertex ids in the neighbor list are sorted**
                    int nw = col_idx_Gb_access[begin_nw]; // neighbor of w (local id)
                    int nu = col_idx_Gb_access[begin_nu]; // neighbor of u (local id)
                    int nw_id = bv_access[nw]; // actual vertex ID of the neighbor of w
                    int nu_id = bv_access[nu]; // actual vertex ID of the neighbor of u
                    while (begin_nw <= end_nw && begin_nu <= end_nu) {
                        if(u_c == C_access[nw_id] && u_id != nw_id){ //Triad-II: If neighbor of w, i.e., nw_id is in community u_c => Triad-II. It means nw = v in triad (u,v,w)
                            int vu_idx = nw * comm_count_col_size + u_c; // u_c = v_c here
                            int wu_idx = w * comm_count_col_size + u_c;
                            W_vu = comm_counts_access[vu_idx];
                            W_wu = comm_counts_access[wu_idx];
                            score += std::cbrt((double)(W_vu * W_wu * W_wu)); // as per Lemma 2, W_wu = W_wv in triad-II as u_c = v_c
                        }
                        else if(nw_id == nu_id && w_c != C_access[nw_id]){ //Triad-I: It means nw = nu = v in triad (u,v,w)
                            int vu_idx = nw * comm_count_col_size + u_c; // here v = nw = nu
                            int wu_idx = w * comm_count_col_size + u_c;
                            int wv_idx = w * comm_count_col_size + C_access[nw_id]; // here v_id = nw_id
                            W_vu = comm_counts_access[vu_idx];
                            W_wu = comm_counts_access[wu_idx];
                            W_wv = comm_counts_access[wv_idx];
                            score += std::cbrt((double)(W_vu * W_wu * W_wv)); // accumulating RSI for u
                        }
                        comp1 = (nw_id >= nu_id);
                        comp2 = (nw_id <= nu_id);
                        nw_bound = (begin_nw == end_nw);
                        nu_bound = (begin_nu == end_nu);
                        // early termination
                        if ((nw_bound && comp2) || (nu_bound && comp1))
                            break;
                        if ((comp1 && !nu_bound) || nw_bound){
                            begin_nu += 1;
                            nu = col_idx_Gb_access[begin_nu]; // next neighbor of u (local bv id)
                            nu_id = bv_access[nu]; // actual vertex ID of the next neighbor of u
                        }   
                        if ((comp2 && !nw_bound) || nu_bound){
                            begin_nw += 1;
                            nw = col_idx_Gb_access[begin_nw]; // next neighbor of w (local bv id)
                            nw_id = bv_access[nw]; // actual vertex ID of the next neighbor of w
                        }
                            
                    }

                }
                int degree = degree_access[u_id];
                if(degree > 1){
                    score = score / (degree_access[u_id] * (degree_access[u_id] - 1));
                    float scale = std::pow(10.0f, 4); // rounding upto 4 decimal places
                    score = std::round(score * scale) / scale; // rounding upto 4 decimal places
                    R_access[u_id] = score;
                }
                
            }); });
}
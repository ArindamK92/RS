#include "R_spanner_helper.hpp"

/// @brief reads the arguments and assigns filename, comm_filename, target_communities
/// @param argc 
/// @param argv 
/// @param filename 
/// @param comm_filename 
/// @param target_communities 
/// @return 
int read_args(int argc, char **argv, std::string& filename, std::string& comm_filename, std::vector<int>& target_communities){
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "-g")
        {
            if (i + 1 < argc)
            {
                filename = argv[++i];
            }
            else
            {
                std::cerr << "Error: -g flag requires a filename argument" << std::endl;
                return 1;
            }
        }
        else if (std::string(argv[i]) == "-c")
        {
            if (i + 1 < argc)
            {
                comm_filename = argv[++i];
            }
            else
            {
                std::cerr << "Error: -c flag requires a commname argument" << std::endl;
                return 1;
            }
        }
        else if (std::string(argv[i]) == "-t")
        {
            if (i + 1 < argc)
            {
                target_communities.push_back(atoi(argv[++i]));
            }
            else
            {
                std::cerr << "Error: -t flag requires a target community id" << std::endl;
                return 1;
            }
        }
        else
        {
            std::cerr << "Error: Unknown flag '" << argv[i] << "'" << std::endl;
            return 1;
        }
    }
    return 0;
}

/// @brief reads the community id for each vertices. We consider the comm id starts from 0 and goes till max_comm_id.
/// @brief We consider there is at least one vertex in each community
/// @param filename
/// @param C
void readCommunity(std::string filename, std::vector<int> &C, /*int *max_comm_id,*/ std::unordered_map<int, int> comm_map)
{
    std::ifstream file(filename);
    std::string line;
    int i = 0, c;

    if (file.is_open())
    {
        while (std::getline(file, line))
        {
            if (line[0] == '%' || line[0] == '#')
                continue; // Ignore comments

            std::istringstream iss(line);
            iss >> c;

            // comm_map stores a local comm id (starting from 1) for the targeted community
            // All non-targeted community vertices are given local id 0
            if(comm_map.find(c) == comm_map.end()){ 
                C[i] = 0;
            }
            else{
                C[i] = comm_map[c];
            }
            i++; // nodes in .mtx are 1-based, adjust to 0-based
        }
        //std::cout << "comm found for total vertices: " << i << "\n";
        file.close();
    }
    else
    {
        std::cout << "Unable to open file: " << filename << std::endl;
        exit(1);
    }

    return;
}

/// @brief read the graph in .mtx format and store in CSR format
/// @param filename
/// @param n
/// @param m
/// @return retured the stored graph in CSR format
CSR mtxToCSR(const std::string &filename, /*std::vector<int> &degree,*/ int *n, int *m)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    std::string line;

    while (std::getline(file, line) && (line[0] == '%' || line[0] == '#'));

    std::istringstream dims(line);
    int numRows, numCols, numEntries;
    dims >> numRows >> numCols >> numEntries;
    *n = numRows;
    *m = numEntries;

    std::vector<std::map<int, int>> rows(numRows);

    for (int i = 0; i < numEntries; i++)
    {
        std::getline(file, line);
        std::istringstream iss(line);

        int row, col, value = 0;
        iss >> row >> col;

        // .mtx format is 1-based, so we adjust
        rows[row - 1][col - 1] = value;
        rows[col - 1][row - 1] = value;
    }

    CSR csr;
    csr.row_ptr.push_back(0);

    //int count = 0;
    for (const auto &row : rows)
    {
        for (const auto &kv : row)
        {
            csr.data.push_back(kv.second);
            csr.col_idx.push_back(kv.first);
            //count ++;
        }
        csr.row_ptr.push_back(csr.data.size());
    }
    //std::cout << "edge count after reading file:" << count; //count can be less than m if there is both "a b" and "b a" in graph data
    return csr;
}
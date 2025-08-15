
#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <boost/program_options.hpp>
#include <openvdb/openvdb.h>
#include <tira/volume.h>
#include <openvdb/tools/ChangeBackground.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range3d.h>
#include <unordered_set>
#include <queue>
#include <set>

using GridType = openvdb::FloatGrid::Ptr;
using Point = openvdb::Coord;

template <typename GridType>
void prepare_data_full(GridType grid, float threshold = 0.0f) {
    auto accessor = grid->getAccessor();

   // openvdb::CoordBBox bbox;
    openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();


    for (int z = bbox.min().z(); z <= bbox.max().z(); ++z) {
        for (int y = bbox.min().y(); y <= bbox.max().y(); ++y) {
            for (int x = bbox.min().x(); x <= bbox.max().x(); ++x) {
                openvdb::Coord c(x, y, z);
                float val = accessor.getValue(c);  // works for inactive too
                accessor.setValue(c, val < threshold ? 1.0f : 0.0f);  // binarize all
            }
        }
    }
}


template <typename GridType>
void invert_binarized(GridType grid) {
    auto accessor = grid->getAccessor();
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float value = iter.getValue();
        accessor.setValue(iter.getCoord(), value == 0.0f ? 1.0f : 0.0f);
    }
}

// binarizing 
void prepare_data(GridType grid) {
    auto accessor = grid->getAccessor();

    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float val = iter.getValue();
        iter.setValue(val <= 0 ? 1.0f : 0.0f);  // binary mask, only within vessel
    }
}


//vdb to image conversion for 3D
template<class GridType> void vdb2img3D(GridType& grid, tira::volume<float>& img) {
    using ValueT = typename GridType::ValueType;

    typename GridType::Accessor accessor = grid.getAccessor();

    //openvdb::Coord dim = grid.evalActiveVoxelDim();
    openvdb::Coord ijk;
    int& i = ijk[0], & j = ijk[1], & k = ijk[2];
    for (i = 0; i < img.X(); i++) {
        for (j = 0; j < img.Y(); j++) {
            for (k = 0; k < img.Z(); k++) {
                float pixel = (float)accessor.getValue(ijk);
                img(i, j, k) = pixel;
            }
        }
    }
}

// Get pixel value from the grid (returns 0 if out of bounds)
float get_pixel(GridType grid, int x, int y, int z) {
    openvdb::Coord ijk(x, y, z);
    auto accessor = grid->getAccessor();
    if (accessor.isValueOn(ijk)) {
        return accessor.getValue(ijk);
    }
    return 0.0f;
}


// Get the 3x3x3 neighborhood around a given voxel (x, y, z)
std::array<float, 27> get_neighborhood(openvdb::FloatGrid::Ptr grid, int x, int y, int z) {
    std::array<float, 27> neighborhood;
    auto accessor = grid->getAccessor();

    neighborhood[0] = get_pixel(grid, x - 1, y - 1, z - 1);
    neighborhood[1] = get_pixel(grid, x, y - 1, z - 1);
    neighborhood[2] = get_pixel(grid, x + 1, y - 1, z - 1);

    neighborhood[3] = get_pixel(grid, x - 1, y, z - 1);
    neighborhood[4] = get_pixel(grid, x, y, z - 1);
    neighborhood[5] = get_pixel(grid, x + 1, y, z - 1);

    neighborhood[6] = get_pixel(grid, x - 1, y + 1, z - 1);
    neighborhood[7] = get_pixel(grid, x, y + 1, z - 1);
    neighborhood[8] = get_pixel(grid, x + 1, y + 1, z - 1);

    neighborhood[9] = get_pixel(grid, x - 1, y - 1, z);
    neighborhood[10] = get_pixel(grid, x, y - 1, z);
    neighborhood[11] = get_pixel(grid, x + 1, y - 1, z);

    neighborhood[12] = get_pixel(grid, x - 1, y, z);
    neighborhood[13] = get_pixel(grid, x, y, z);
    neighborhood[14] = get_pixel(grid, x + 1, y, z);

    neighborhood[15] = get_pixel(grid, x - 1, y + 1, z);
    neighborhood[16] = get_pixel(grid, x, y + 1, z);
    neighborhood[17] = get_pixel(grid, x + 1, y + 1, z);

    neighborhood[18] = get_pixel(grid, x - 1, y - 1, z + 1);
    neighborhood[19] = get_pixel(grid, x, y - 1, z + 1);
    neighborhood[20] = get_pixel(grid, x + 1, y - 1, z + 1);

    neighborhood[21] = get_pixel(grid, x - 1, y, z + 1);
    neighborhood[22] = get_pixel(grid, x, y, z + 1);
    neighborhood[23] = get_pixel(grid, x + 1, y, z + 1);

    neighborhood[24] = get_pixel(grid, x - 1, y + 1, z + 1);
    neighborhood[25] = get_pixel(grid, x, y + 1, z + 1);
    neighborhood[26] = get_pixel(grid, x + 1, y + 1, z + 1);

    return neighborhood;
}

// Check if a voxel at (x, y, z) is an endpoint
bool is_endpoint(openvdb::FloatGrid::Ptr grid, int x, int y, int z) {
    auto neighbor = get_neighborhood(grid, x, y, z);
    int count = -1;  // Start from -1 to exclude the center point itself
    for (int i = 0; i < 27; ++i) {
        if (neighbor[i] == 1.0f) count++;
    }
    return count == 1;
}

// Fill Euler Lookup Table (LUT) for Euler characteristic 
std::array<int, 256> fill_euler_LUT() {
    std::array<int, 256> LUT = { 0 };

    LUT[1] = 1;  LUT[3] = -1; LUT[5] = -1; LUT[7] = 1; LUT[9] = -3; LUT[11] = -1;
    LUT[13] = -1; LUT[15] = 1; LUT[17] = -1; LUT[19] = 1; LUT[21] = 1; LUT[23] = -1;
    LUT[25] = 3; LUT[27] = 1; LUT[29] = 1; LUT[31] = -1; LUT[33] = -3; LUT[35] = -1;
    LUT[37] = 3; LUT[39] = 1; LUT[41] = 1; LUT[43] = -1; LUT[45] = 3; LUT[47] = 1;
    LUT[49] = -1; LUT[51] = 1; LUT[53] = 1; LUT[55] = -1; LUT[57] = 3; LUT[59] = 1;
    LUT[61] = 1; LUT[63] = -1; LUT[65] = -3; LUT[67] = 3; LUT[69] = -1; LUT[71] = 1;
    LUT[73] = 1; LUT[75] = 3; LUT[77] = -1; LUT[79] = 1; LUT[81] = -1; LUT[83] = 1;
    LUT[85] = 1; LUT[87] = -1; LUT[89] = 3; LUT[91] = 1; LUT[93] = 1; LUT[95] = -1;
    LUT[97] = 1; LUT[99] = 3; LUT[101] = 3; LUT[103] = 1; LUT[105] = 5; LUT[107] = 3;
    LUT[109] = 3; LUT[111] = 1; LUT[113] = -1; LUT[115] = 1; LUT[117] = 1; LUT[119] = -1;
    LUT[121] = 3; LUT[123] = 1; LUT[125] = 1; LUT[127] = -1; LUT[129] = -7; LUT[131] = -1;
    LUT[133] = -1; LUT[135] = 1; LUT[137] = -3; LUT[139] = -1; LUT[141] = -1; LUT[143] = 1;
    LUT[145] = -1; LUT[147] = 1; LUT[149] = 1; LUT[151] = -1; LUT[153] = 3; LUT[155] = 1;
    LUT[157] = 1; LUT[159] = -1; LUT[161] = -3; LUT[163] = -1; LUT[165] = 3; LUT[167] = 1;
    LUT[169] = 1; LUT[171] = -1; LUT[173] = 3; LUT[175] = 1; LUT[177] = -1; LUT[179] = 1;
    LUT[181] = 1; LUT[183] = -1; LUT[185] = 3; LUT[187] = 1; LUT[189] = 1; LUT[191] = -1;
    LUT[193] = -3; LUT[195] = 3; LUT[197] = -1; LUT[199] = 1; LUT[201] = 1; LUT[203] = 3;
    LUT[205] = -1; LUT[207] = 1; LUT[209] = -1; LUT[211] = 1; LUT[213] = 1; LUT[215] = -1;
    LUT[217] = 3; LUT[219] = 1; LUT[221] = 1; LUT[223] = -1; LUT[225] = 1; LUT[227] = 3;
    LUT[229] = 3; LUT[231] = 1; LUT[233] = 5; LUT[235] = 3; LUT[237] = 3; LUT[239] = 1;
    LUT[241] = -1; LUT[243] = 1; LUT[245] = 1; LUT[247] = -1; LUT[249] = 3; LUT[251] = 1;
    LUT[253] = 1; LUT[255] = -1;

    return LUT;
}


// Directional neighbor access functions
float N(openvdb::FloatGrid::Ptr grid, int x, int y, int z) { return get_pixel(grid, x, y - 1, z); }
float S(openvdb::FloatGrid::Ptr grid, int x, int y, int z) { return get_pixel(grid, x, y + 1, z); }
float E(openvdb::FloatGrid::Ptr grid, int x, int y, int z) { return get_pixel(grid, x + 1, y, z); }
float W(openvdb::FloatGrid::Ptr grid, int x, int y, int z) { return get_pixel(grid, x - 1, y, z); }
float U(openvdb::FloatGrid::Ptr grid, int x, int y, int z) { return get_pixel(grid, x, y, z + 1); }
float B(openvdb::FloatGrid::Ptr grid, int x, int y, int z) { return get_pixel(grid, x, y, z - 1); }

// Get pixel without boundary check
float get_pixel_nocheck(openvdb::FloatGrid::Ptr grid, int x, int y, int z) {
    openvdb::Coord coord(x, y, z);
    return grid->tree().getValue(coord);
}

// Set pixel value with boundary check
void set_pixel(openvdb::FloatGrid::Ptr grid, int x, int y, int z, float value) {
    openvdb::Coord coord(x, y, z);
    auto accessor = grid->getAccessor();
    accessor.setValue(coord, value);
}

// Create a LUT that maps integers 0–255 to their number of 1-bits (i.e., how many neighbors are on).
void fill_num_of_points_LUT(std::array<int, 256>& LUT) {
    for (int i = 0; i < 256; ++i) {
        int count = 0;
        int val = i;
        while (val) {
            count += val & 1;
            val >>= 1;
        }
        LUT[i] = count;
    }
}


uint8_t index_octant_NEB(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[2])   v |= 128;
    if (n[1])   v |= 64;
    if (n[11])  v |= 32;
    if (n[10])  v |= 16;
    if (n[5])   v |= 8;
    if (n[4])   v |= 4;
    if (n[14])  v |= 2;
    return v;
}

uint8_t index_octant_NWB(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[0])   v |= 128;
    if (n[9])   v |= 64;
    if (n[3])   v |= 32;
    if (n[12])  v |= 16;
    if (n[1])   v |= 8;
    if (n[10])  v |= 4;
    if (n[4])   v |= 2;
    return v;
}

uint8_t index_octant_SEB(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[8])   v |= 128;
    if (n[7])   v |= 64;
    if (n[17])  v |= 32;
    if (n[16])  v |= 16;
    if (n[5])   v |= 8;
    if (n[4])   v |= 4;
    if (n[14])  v |= 2;
    return v;
}

uint8_t index_octant_SWB(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[6])   v |= 128;
    if (n[15])  v |= 64;
    if (n[7])   v |= 32;
    if (n[16])  v |= 16;
    if (n[3])   v |= 8;
    if (n[12])  v |= 4;
    if (n[4])   v |= 2;
    return v;
}

uint8_t index_octant_NEU(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[20])  v |= 128;
    if (n[23])  v |= 64;
    if (n[19])  v |= 32;
    if (n[22])  v |= 16;
    if (n[11])  v |= 8;
    if (n[14])  v |= 4;
    if (n[10])  v |= 2;
    return v;
}

uint8_t index_octant_NWU(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[18]) v |= 128;
    if (n[21]) v |= 64;
    if (n[9])  v |= 32;
    if (n[12]) v |= 16;
    if (n[19]) v |= 8;
    if (n[22]) v |= 4;
    if (n[10]) v |= 2;
    return v;
}

uint8_t index_octant_SEU(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[26]) v |= 128;
    if (n[23]) v |= 64;
    if (n[17]) v |= 32;
    if (n[14]) v |= 16;
    if (n[25]) v |= 8;
    if (n[22]) v |= 4;
    if (n[16]) v |= 2;
    return v;
}

uint8_t index_octant_SWU(const std::array<uint8_t, 27>& n) {
    uint8_t v = 1;
    if (n[24]) v |= 128;
    if (n[25]) v |= 64;
    if (n[15]) v |= 32;
    if (n[16]) v |= 16;
    if (n[21]) v |= 8;
    if (n[22]) v |= 4;
    if (n[12]) v |= 2;
    return v;
}


bool is_euler_invariant(const std::array<uint8_t, 27>& neighbors, const std::array<int, 256>& LUT) {
    int eulerChar = 0;
    eulerChar += LUT[index_octant_SWU(neighbors)];
    eulerChar += LUT[index_octant_SEU(neighbors)];
    eulerChar += LUT[index_octant_NWU(neighbors)];
    eulerChar += LUT[index_octant_NEU(neighbors)];
    eulerChar += LUT[index_octant_SWB(neighbors)];
    eulerChar += LUT[index_octant_SEB(neighbors)];
    eulerChar += LUT[index_octant_NWB(neighbors)];
    eulerChar += LUT[index_octant_NEB(neighbors)];
    return eulerChar == 0;
}

void octree_labeling(int octant, int label, std::array<int, 26>& cube) {
    if (octant == 1) {
        if (cube[0] == 1) cube[0] = label;
        if (cube[1] == 1) { cube[1] = label; octree_labeling(2, label, cube); }
        if (cube[3] == 1) { cube[3] = label; octree_labeling(3, label, cube); }
        if (cube[4] == 1) {
            cube[4] = label;
            octree_labeling(2, label, cube);
            octree_labeling(3, label, cube);
            octree_labeling(4, label, cube);
        }
        if (cube[9] == 1) { cube[9] = label; octree_labeling(5, label, cube); }
        if (cube[10] == 1) {
            cube[10] = label;
            octree_labeling(2, label, cube);
            octree_labeling(5, label, cube);
            octree_labeling(6, label, cube);
        }
        if (cube[12] == 1) {
            cube[12] = label;
            octree_labeling(3, label, cube);
            octree_labeling(5, label, cube);
            octree_labeling(7, label, cube);
        }
    }
    if (octant == 2) {
        if (cube[1] == 1) { cube[1] = label; octree_labeling(1, label, cube); }
        if (cube[4] == 1) {
            cube[4] = label;
            octree_labeling(1, label, cube);
            octree_labeling(3, label, cube);
            octree_labeling(4, label, cube);
        }
        if (cube[10] == 1) {
            cube[10] = label;
            octree_labeling(1, label, cube);
            octree_labeling(5, label, cube);
            octree_labeling(6, label, cube);
        }
        if (cube[2] == 1) cube[2] = label;
        if (cube[5] == 1) { cube[5] = label; octree_labeling(4, label, cube); }
        if (cube[11] == 1) { cube[11] = label; octree_labeling(6, label, cube); }
        if (cube[13] == 1) {
            cube[13] = label;
            octree_labeling(4, label, cube);
            octree_labeling(6, label, cube);
            octree_labeling(8, label, cube);
        }
    }
    if (octant == 3) {
        if (cube[3] == 1) { cube[3] = label; octree_labeling(1, label, cube); }
        if (cube[4] == 1) {
            cube[4] = label;
            octree_labeling(1, label, cube);
            octree_labeling(2, label, cube);
            octree_labeling(4, label, cube);
        }
        if (cube[12] == 1) {
            cube[12] = label;
            octree_labeling(1, label, cube);
            octree_labeling(5, label, cube);
            octree_labeling(7, label, cube);
        }
        if (cube[6] == 1) cube[6] = label;
        if (cube[7] == 1) { cube[7] = label; octree_labeling(4, label, cube); }
        if (cube[14] == 1) { cube[14] = label; octree_labeling(7, label, cube); }
        if (cube[15] == 1) {
            cube[15] = label;
            octree_labeling(4, label, cube);
            octree_labeling(7, label, cube);
            octree_labeling(8, label, cube);
        }
    }
    if (octant == 4) {
        if (cube[4] == 1) {
            cube[4] = label;
            octree_labeling(1, label, cube);
            octree_labeling(2, label, cube);
            octree_labeling(3, label, cube);
        }
        if (cube[5] == 1) { cube[5] = label; octree_labeling(2, label, cube); }
        if (cube[13] == 1) {
            cube[13] = label;
            octree_labeling(2, label, cube);
            octree_labeling(6, label, cube);
            octree_labeling(8, label, cube);
        }
        if (cube[7] == 1) { cube[7] = label; octree_labeling(3, label, cube); }
        if (cube[15] == 1) {
            cube[15] = label;
            octree_labeling(3, label, cube);
            octree_labeling(7, label, cube);
            octree_labeling(8, label, cube);
        }
        if (cube[8] == 1) cube[8] = label;
        if (cube[16] == 1) { cube[16] = label; octree_labeling(8, label, cube); }
    }
    if (octant == 5) {
        if (cube[9] == 1) { cube[9] = label; octree_labeling(1, label, cube); }
        if (cube[10] == 1) {
            cube[10] = label;
            octree_labeling(1, label, cube);
            octree_labeling(2, label, cube);
            octree_labeling(6, label, cube);
        }
        if (cube[12] == 1) {
            cube[12] = label;
            octree_labeling(1, label, cube);
            octree_labeling(3, label, cube);
            octree_labeling(7, label, cube);
        }
        if (cube[17] == 1) cube[17] = label;
        if (cube[18] == 1) { cube[18] = label; octree_labeling(6, label, cube); }
        if (cube[20] == 1) { cube[20] = label; octree_labeling(7, label, cube); }
        if (cube[21] == 1) {
            cube[21] = label;
            octree_labeling(6, label, cube);
            octree_labeling(7, label, cube);
            octree_labeling(8, label, cube);
        }
    }
    if (octant == 6) {
        if (cube[10] == 1) {
            cube[10] = label;
            octree_labeling(1, label, cube);
            octree_labeling(2, label, cube);
            octree_labeling(5, label, cube);
        }
        if (cube[11] == 1) { cube[11] = label; octree_labeling(2, label, cube); }
        if (cube[13] == 1) {
            cube[13] = label;
            octree_labeling(2, label, cube);
            octree_labeling(4, label, cube);
            octree_labeling(8, label, cube);
        }
        if (cube[18] == 1) { cube[18] = label; octree_labeling(5, label, cube); }
        if (cube[21] == 1) {
            cube[21] = label;
            octree_labeling(5, label, cube);
            octree_labeling(7, label, cube);
            octree_labeling(8, label, cube);
        }
        if (cube[19] == 1) cube[19] = label;
        if (cube[22] == 1) { cube[22] = label; octree_labeling(8, label, cube); }
    }
    if (octant == 7) {
        if (cube[12] == 1) {
            cube[12] = label;
            octree_labeling(1, label, cube);
            octree_labeling(3, label, cube);
            octree_labeling(5, label, cube);
        }
        if (cube[14] == 1) { cube[14] = label; octree_labeling(3, label, cube); }
        if (cube[15] == 1) {
            cube[15] = label;
            octree_labeling(3, label, cube);
            octree_labeling(4, label, cube);
            octree_labeling(8, label, cube);
        }
        if (cube[20] == 1) { cube[20] = label; octree_labeling(5, label, cube); }
        if (cube[21] == 1) {
            cube[21] = label;
            octree_labeling(5, label, cube);
            octree_labeling(6, label, cube);
            octree_labeling(8, label, cube);
        }
        if (cube[23] == 1) cube[23] = label;
        if (cube[24] == 1) { cube[24] = label; octree_labeling(8, label, cube); }
    }
    if (octant == 8) {
        if (cube[13] == 1) {
            cube[13] = label;
            octree_labeling(2, label, cube);
            octree_labeling(4, label, cube);
            octree_labeling(6, label, cube);
        }
        if (cube[15] == 1) {
            cube[15] = label;
            octree_labeling(3, label, cube);
            octree_labeling(4, label, cube);
            octree_labeling(7, label, cube);
        }
        if (cube[16] == 1) { cube[16] = label; octree_labeling(4, label, cube); }
        if (cube[21] == 1) {
            cube[21] = label;
            octree_labeling(5, label, cube);
            octree_labeling(6, label, cube);
            octree_labeling(7, label, cube);
        }
        if (cube[22] == 1) { cube[22] = label; octree_labeling(6, label, cube); }
        if (cube[24] == 1) { cube[24] = label; octree_labeling(7, label, cube); }
        if (cube[25] == 1) cube[25] = label;
    }
}

bool is_simple_point(const std::array<uint8_t, 27>& neighbors) {
    std::array<int, 26> cube;

    // Copy first 13 elements as-is
    for (int i = 0; i < 13; ++i)
        cube[i] = neighbors[i];

    // Skip index 13 (center), then continue from 14 to 26 into cube[13 to 25]
    for (int i = 14; i < 27; ++i)
        cube[i - 1] = neighbors[i];

    int label = 2;

    for (int i = 0; i < 26; ++i) {
        if (cube[i] == 1) {
            switch (i) {
            case 0: case 1: case 3: case 4: case 9: case 10: case 12:
                octree_labeling(1, label, cube);
                break;
            case 2: case 5: case 11: case 13:
                octree_labeling(2, label, cube);
                break;
            case 6: case 7: case 14: case 15:
                octree_labeling(3, label, cube);
                break;
            case 8: case 16:
                octree_labeling(4, label, cube);
                break;
            case 17: case 18: case 20: case 21:
                octree_labeling(5, label, cube);
                break;
            case 19: case 22:
                octree_labeling(6, label, cube);
                break;
            case 23: case 24:
                octree_labeling(7, label, cube);
                break;
            case 25:
                octree_labeling(8, label, cube);
                break;
            }
            label++;
            if ((label - 2) >= 2) {
                return false;
            }
        }
    }

    return true;
}

/*..........slow...............
void computeThinImage(openvdb::FloatGrid::Ptr grid) {
    int width = grid->evalActiveVoxelDim().x();
    int height = grid->evalActiveVoxelDim().y();
    int depth = grid->evalActiveVoxelDim().z();

    std::array<int, 256> eulerLUT = fill_euler_LUT();
    std::array<int, 256> pointsLUT;
    fill_num_of_points_LUT(pointsLUT);

    std::vector<openvdb::Coord> simpleBorderPoints;
    int iterations = 0;
    int unchangedBorders = 0;

    while (unchangedBorders < 6) {
        unchangedBorders = 0;
        iterations++;
        std::cout << "Iteration: " << iterations << std::endl;

        for (int currentBorder = 1; currentBorder <= 6; currentBorder++) {
            bool noChange = true;

            for (int z = 0; z < depth; z++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        if (get_pixel_nocheck(grid, x, y, z) != 1.0f)
                            continue;

                        bool isBorderPoint = false;

                        if (currentBorder == 1 && N(grid, x, y, z) <= 0) isBorderPoint = true;
                        if (currentBorder == 2 && S(grid, x, y, z) <= 0) isBorderPoint = true;
                        if (currentBorder == 3 && E(grid, x, y, z) <= 0) isBorderPoint = true;
                        if (currentBorder == 4 && W(grid, x, y, z) <= 0) isBorderPoint = true;
                        if (currentBorder == 5 && U(grid, x, y, z) <= 0) isBorderPoint = true;
                        if (currentBorder == 6 && B(grid, x, y, z) <= 0) isBorderPoint = true;

                        if (!isBorderPoint) continue;
                        if (is_endpoint(grid, x, y, z)) continue;

                        std::array<float, 27> neighborhood_float = get_neighborhood(grid, x, y, z);
                        std::array<uint8_t, 27> neighborhood;
                        for (int i = 0; i < 27; ++i)
                            neighborhood[i] = static_cast<uint8_t>(neighborhood_float[i]);

                        if (!is_euler_invariant(neighborhood, eulerLUT)) continue;
                        if (!is_simple_point(neighborhood)) continue;

                        simpleBorderPoints.push_back(openvdb::Coord(x, y, z));
                    }
                }
            }

            for (const auto& index : simpleBorderPoints) {
                std::array<float, 27> neighbors_float = get_neighborhood(grid, index.x(), index.y(), index.z());
                std::array<uint8_t, 27> neighbors;
                for (int i = 0; i < 27; ++i)
                    neighbors[i] = static_cast<uint8_t>(neighbors_float[i]);

                if (is_simple_point(neighbors)) {
                    set_pixel(grid, index.x(), index.y(), index.z(), 0.0f);
                    noChange = false;
                }
            }

            if (noChange) {
                unchangedBorders++;
                std::cout << "No change for border: " << currentBorder << std::endl;
            }

            simpleBorderPoints.clear();
        }

        if (iterations > 60) {  // Safety break for infinite loop
            std::cout << "Error: Exceeded maximum iterations" << std::endl;
            break;
        }
    }
    std::cout << "Thinning completed in " << iterations << " iterations." << std::endl;
}
*/

// Final thinning 
void computeThinImage(openvdb::FloatGrid::Ptr grid) {
    std::array<int, 256> eulerLUT = fill_euler_LUT();
    std::array<int, 256> pointsLUT;
    fill_num_of_points_LUT(pointsLUT);

    std::vector<openvdb::Coord> simpleBorderPoints;
    int iterations = 0;
    int unchangedBorders = 0;

    while (unchangedBorders < 6) {
        unchangedBorders = 0;
        iterations++;
        std::cout << "Iteration: " << iterations << std::endl;

        for (int currentBorder = 1; currentBorder <= 6; currentBorder++) {
            bool noChange = true;

            // Iterate only over active foreground voxels (value == 1.0f)
            for (auto iter = grid->beginValueOn(); iter; ++iter) {
                if (iter.getValue() != 1.0f) continue;

                const openvdb::Coord& c = iter.getCoord();
                int x = c.x(), y = c.y(), z = c.z();

                bool isBorderPoint = false;

                if (currentBorder == 1 && N(grid, x, y, z) <= 0) isBorderPoint = true;
                if (currentBorder == 2 && S(grid, x, y, z) <= 0) isBorderPoint = true;
                if (currentBorder == 3 && E(grid, x, y, z) <= 0) isBorderPoint = true;
                if (currentBorder == 4 && W(grid, x, y, z) <= 0) isBorderPoint = true;
                if (currentBorder == 5 && U(grid, x, y, z) <= 0) isBorderPoint = true;
                if (currentBorder == 6 && B(grid, x, y, z) <= 0) isBorderPoint = true;

                if (!isBorderPoint) continue;
                if (is_endpoint(grid, x, y, z)) continue;

                std::array<float, 27> neighborhood_float = get_neighborhood(grid, x, y, z);
                std::array<uint8_t, 27> neighborhood;
                for (int i = 0; i < 27; ++i)
                    neighborhood[i] = static_cast<uint8_t>(neighborhood_float[i]);

                if (!is_euler_invariant(neighborhood, eulerLUT)) continue;
                if (!is_simple_point(neighborhood)) continue;

                simpleBorderPoints.push_back(c);
            }

            for (const auto& index : simpleBorderPoints) {
                std::array<float, 27> neighbors_float = get_neighborhood(grid, index.x(), index.y(), index.z());
                std::array<uint8_t, 27> neighbors;
                for (int i = 0; i < 27; ++i)
                    neighbors[i] = static_cast<uint8_t>(neighbors_float[i]);

                if (is_simple_point(neighbors)) {
                    set_pixel(grid, index.x(), index.y(), index.z(), 0.0f);
                    noChange = false;
                }
            }

            if (noChange) {
                unchangedBorders++;
                std::cout << "No change for border: " << currentBorder << std::endl;
            }

            simpleBorderPoints.clear();
        }

        if (iterations > 60) {
            std::cout << "Error: Exceeded maximum iterations" << std::endl;
            break;
        }
    }

    std::cout << "Thinning completed in " << iterations << " iterations." << std::endl;
}


struct CoordHash {
    std::size_t operator()(const openvdb::Coord& coord) const {
        std::size_t h1 = std::hash<int>()(coord.x());
        std::size_t h2 = std::hash<int>()(coord.y());
        std::size_t h3 = std::hash<int>()(coord.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};


openvdb::FloatGrid::Ptr fill_holes_openvdb_tbb(openvdb::FloatGrid::Ptr& input_grid) {
    using GridT = openvdb::FloatGrid;
    using Coord = openvdb::Coord;

    openvdb::FloatGrid::Ptr output = input_grid->deepCopy();
    GridT::Accessor accessor = output->getAccessor();

    openvdb::CoordBBox bbox = output->evalActiveVoxelBoundingBox();
    if (bbox.empty()) return output;

    Coord min = bbox.min();
    Coord max = bbox.max();

    // Flood-fill from boundary
    std::queue<Coord> queue;
    std::unordered_set<openvdb::Coord, CoordHash> visited;


    auto try_enqueue = [&](int x, int y, int z) {
        Coord c(x, y, z);
        if (accessor.getValue(c) == 0.0f && visited.insert(c).second) {
            queue.push(c);
        }
        };

    // All 6 faces
    for (int x = min.x(); x <= max.x(); ++x)
        for (int y = min.y(); y <= max.y(); ++y) {
            try_enqueue(x, y, min.z());
            try_enqueue(x, y, max.z());
        }

    for (int x = min.x(); x <= max.x(); ++x)
        for (int z = min.z(); z <= max.z(); ++z) {
            try_enqueue(x, min.y(), z);
            try_enqueue(x, max.y(), z);
        }

    for (int y = min.y(); y <= max.y(); ++y)
        for (int z = min.z(); z <= max.z(); ++z) {
            try_enqueue(min.x(), y, z);
            try_enqueue(max.x(), y, z);
        }

    std::vector<Coord> neighbors = {
        Coord(1,0,0), Coord(-1,0,0),
        Coord(0,1,0), Coord(0,-1,0),
        Coord(0,0,1), Coord(0,0,-1)
    };

    while (!queue.empty()) {
        Coord c = queue.front(); queue.pop();
        for (const Coord& offset : neighbors) {
            Coord n = c + offset;
            if (bbox.isInside(n) && accessor.getValue(n) == 0.0f && visited.insert(n).second) {
                queue.push(n);
            }
        }
    }

    // Parallel fill 
    tbb::parallel_for(
        tbb::blocked_range3d<int>(min.x(), max.x() + 1,
            min.y(), max.y() + 1,
            min.z(), max.z() + 1),
        [&](const tbb::blocked_range3d<int>& r) {
            GridT::Accessor localAccessor = output->getAccessor();  // thread-safe copy
            for (int x = r.pages().begin(); x < r.pages().end(); ++x)
                for (int y = r.rows().begin(); y < r.rows().end(); ++y)
                    for (int z = r.cols().begin(); z < r.cols().end(); ++z) {
                        Coord c(x, y, z);
                        if (localAccessor.getValue(c) == 0.0f &&
                            visited.find(c) == visited.end()) {
                            localAccessor.setValue(c, 255.0f);
                        }
                    }
        }
    );

    return output;
}

openvdb::FloatGrid::Ptr fill_holes_openvdb(openvdb::FloatGrid::Ptr& input_grid) {
    using GridT = openvdb::FloatGrid;
    using Coord = openvdb::Coord;

    // Step 1: Deep copy input to output
    openvdb::FloatGrid::Ptr output = input_grid->deepCopy();

    // Step 2: Get accessors
    GridT::Accessor accessor = output->getAccessor();


    openvdb::CoordBBox bbox = output->evalActiveVoxelBoundingBox();  // ✅ Correct

    if (bbox.empty()) return output;

    Coord min = bbox.min();
    Coord max = bbox.max();

    // Step 4: Flood-fill from 0-valued boundary voxels
    std::queue<Coord> queue;
    std::set<Coord> visited; // fallback if Coord::Hash not available

    auto try_enqueue = [&](int x, int y, int z) {
        Coord c(x, y, z);
        if (accessor.getValue(c) == 0.0f && visited.insert(c).second) {
            queue.push(c);
        }
        };

    // Add all 6 faces of the bounding box
    for (int x = min.x(); x <= max.x(); ++x) {
        for (int y = min.y(); y <= max.y(); ++y) {
            try_enqueue(x, y, min.z());
            try_enqueue(x, y, max.z());
        }
    }
    for (int x = min.x(); x <= max.x(); ++x) {
        for (int z = min.z(); z <= max.z(); ++z) {
            try_enqueue(x, min.y(), z);
            try_enqueue(x, max.y(), z);
        }
    }
    for (int y = min.y(); y <= max.y(); ++y) {
        for (int z = min.z(); z <= max.z(); ++z) {
            try_enqueue(min.x(), y, z);
            try_enqueue(max.x(), y, z);
        }
    }

    // 6-connected neighborhood
    std::vector<Coord> neighbors = {
        Coord(1,0,0), Coord(-1,0,0),
        Coord(0,1,0), Coord(0,-1,0),
        Coord(0,0,1), Coord(0,0,-1)
    };

    // Step 5: BFS flood-fill from boundaries
    while (!queue.empty()) {
        Coord c = queue.front(); queue.pop();
        for (const Coord& offset : neighbors) {
            Coord n = c + offset;
            if (bbox.isInside(n) && accessor.getValue(n) == 0.0f && visited.insert(n).second) {
                queue.push(n);
            }
        }
    }

    // Step 6: Fill all unvisited 0-valued voxels (they are holes)
    for (int x = min.x(); x <= max.x(); ++x) {
        for (int y = min.y(); y <= max.y(); ++y) {
            for (int z = min.z(); z <= max.z(); ++z) {
                Coord c(x, y, z);
                if (accessor.getValue(c) == 0.0f && visited.find(c) == visited.end()) {
                    accessor.setValue(c, 1.0f); // Fill hole
                }
            }
        }
    }

    return output;
}


void cleanup_thin_result(openvdb::FloatGrid::Ptr& grid) {
    std::vector<openvdb::Coord> zero_voxels;

    // First pass: gather all coords to be deactivated
    for (auto iter = grid->cbeginValueOn(); iter; ++iter) {
        if (iter.getValue() == 0.0f) {
            zero_voxels.push_back(iter.getCoord());
        }
    }

    // Second pass: parallel deactivation
    tbb::parallel_for(size_t(0), zero_voxels.size(), [&](size_t i) {
        const openvdb::Coord& c = zero_voxels[i];
        grid->tree().setValueOff(c);
        });

    grid->tree().prune();  // shrink tree memory

}

int main() {
    // Initialize 
    openvdb::initialize();

    // input VDB 
    std::string input_filename = "C:/openvdb_drop/bin/phi_400.vdb";
    openvdb::io::File file_sdf(input_filename);
    file_sdf.open();
   

    openvdb::GridBase::Ptr baseGrid = file_sdf.readGrid(file_sdf.beginName().gridName());
    file_sdf.close();

    // Cast to FloatGrid
    //openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    
    openvdb::FloatGrid::Ptr input_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (!input_grid) {
        std::cerr << "Error: Failed to cast grid to FloatGrid." << std::endl;
        return 1;
    }

    prepare_data_full(input_grid);

    std::cout << "done prepare" << std::endl;
    input_grid = fill_holes_openvdb(input_grid);

    std::cout << "done fill holes" << std::endl;
   
   // grid->tree().prune();           // remove zero-valued active voxels

    // Lee thinning 
    openvdb::tools::changeBackground(input_grid->tree(), 0);
    input_grid->tree().prune();           // remove zero-valued active voxels
    input_grid->pruneGrid(0.0f);  // Removes all inactive voxels with the background value of 0.0

    computeThinImage(input_grid);
    
    /*tira::volume<float> T2(400, 400, 400);
   vdb2img3D(*grid, T2);
   T2.save_npy("C:/Users/meher/spyder/HD13.npy");*/
   //exit(1);
   
    //cleanup_thin_result(input_grid);

    // Save the output VDB 
    std::string output_filename = "C:/openvdb_drop/bin/lee_thin_400.vdb";
    openvdb::io::File output_file(output_filename);
    openvdb::GridPtrVec grids;
    grids.push_back(input_grid);

    try {
        output_file.write(grids);
        output_file.close();
        std::cout << "Successfully saved thinned VDB to: " << output_filename << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving VDB file: " << e.what() << std::endl;
        return 1;
    }

    return 0;

}

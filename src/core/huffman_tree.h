#pragma once

#include <vector>
#include <string>
#include <array>
#include <memory>
#include <cstdint>

namespace gpu_huffman {

struct HuffmanNode {
    unsigned char ch;
    uint32_t freq;
    HuffmanNode *left, *right;

    HuffmanNode(unsigned char character, uint32_t frequency)
        : ch(character), freq(frequency), left(nullptr), right(nullptr) {}
};

struct CompareNodes {
    bool operator()(HuffmanNode* l, HuffmanNode* r) {
        return l->freq > r->freq;
    }
};

class HuffmanTree {
public:
    HuffmanTree();
    ~HuffmanTree();

    void build(const uint32_t frequencies[256]);
    std::array<std::string, 256> generate_codes();
    
    struct CompressionStats {
        size_t original_size_bits;
        size_t compressed_size_bits;
        double savings_percent;
    };

    CompressionStats compute_stats(const uint32_t frequencies[256]);

private:
    HuffmanNode* root_;
    void delete_tree(HuffmanNode* node);
    void encode(HuffmanNode* node, std::string str, std::array<std::string, 256>& huffman_code);
};

} // namespace gpu_huffman

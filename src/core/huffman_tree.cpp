#include "huffman_tree.h"
#include <queue>

namespace gpu_huffman {

HuffmanTree::HuffmanTree() : root_(nullptr) {}

HuffmanTree::~HuffmanTree() {
    delete_tree(root_);
}

void HuffmanTree::delete_tree(HuffmanNode* node) {
    if (!node) return;
    delete_tree(node->left);
    delete_tree(node->right);
    delete(node);
}

void HuffmanTree::build(const uint32_t frequencies[256]) {
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNodes> pq;

    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            pq.push(new HuffmanNode(static_cast<unsigned char>(i), frequencies[i]));
        }
    }

    if (pq.empty()) return;

    while (pq.size() != 1) {
        HuffmanNode *left = pq.top(); pq.pop();
        HuffmanNode *right = pq.top(); pq.pop();

        uint32_t sum = left->freq + right->freq;
        HuffmanNode* node = new HuffmanNode('\0', sum);
        node->left = left;
        node->right = right;
        pq.push(node);
    }

    root_ = pq.top();
}

void HuffmanTree::encode(HuffmanNode* node, std::string str, std::array<std::string, 256>& huffman_code) {
    if (node == nullptr) return;

    if (!node->left && !node->right) {
        huffman_code[node->ch] = str;
    }

    encode(node->left, str + "0", huffman_code);
    encode(node->right, str + "1", huffman_code);
}

std::array<std::string, 256> HuffmanTree::generate_codes() {
    std::array<std::string, 256> huffman_code;
    if (root_ && !root_->left && !root_->right) {
        // Special case: single character
        huffman_code[root_->ch] = "0";
    } else {
        encode(root_, "", huffman_code);
    }
    return huffman_code;
}

HuffmanTree::CompressionStats HuffmanTree::compute_stats(const uint32_t frequencies[256]) {
    auto codes = generate_codes();
    size_t original_bits = 0;
    size_t compressed_bits = 0;

    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            original_bits += static_cast<size_t>(frequencies[i]) * 8;
            compressed_bits += static_cast<size_t>(frequencies[i]) * codes[static_cast<unsigned char>(i)].size();
        }
    }

    double savings = (original_bits > 0) ? (1.0 - (double)compressed_bits / original_bits) * 100.0 : 0.0;
    
    return { original_bits, compressed_bits, savings };
}

} // namespace gpu_huffman

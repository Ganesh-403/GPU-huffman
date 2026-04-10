// ============================================================
// huffman_cpu.cpp
// CPU implementation of:
//   1. Character frequency counting  (single-threaded)
//   2. Huffman Tree construction     (priority queue / min-heap)
//   3. Huffman Code generation       (recursive tree traversal)
// ============================================================

#include "huffman_cpu.h"
#include <cstring>   // memset
#include <stdexcept>

// ------------------------------------------------------------
// cpuCountFrequency
// Iterates through every character in `text` and increments
// the matching slot in the freq[256] array.
// Time Complexity: O(n), where n = number of characters
// ------------------------------------------------------------
void cpuCountFrequency(const std::string& text, int freq[256]) {
    // Zero-initialise the entire frequency table
    memset(freq, 0, 256 * sizeof(int));

    // Single-threaded sequential scan — this is what we compare
    // against the GPU parallel version
    for (unsigned char c : text) {
        freq[c]++;          // Increment count for this ASCII value
    }
}

// ------------------------------------------------------------
// buildHuffmanTree
// Constructs the Huffman Tree using a min-heap (priority queue).
//
// Algorithm:
//   1. Insert each character (freq > 0) as a leaf node.
//   2. Repeatedly extract the two nodes with the lowest frequency.
//   3. Merge them into a new internal node whose frequency is
//      the sum of the two children.
//   4. Repeat until only one node remains — that is the root.
// ------------------------------------------------------------
HuffmanNode* buildHuffmanTree(int freq[256]) {
    // Min-heap: smallest frequency is at the top
    std::priority_queue<HuffmanNode*,
                        std::vector<HuffmanNode*>,
                        Compare> pq;

    // Step 1: create a leaf node for every character that appears
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            pq.push(new HuffmanNode(static_cast<char>(i), freq[i]));
        }
    }

    // Edge case: only one distinct character in the file
    if (pq.size() == 1) {
        HuffmanNode* only = pq.top(); pq.pop();
        // Create a root with the single leaf as its left child
        return new HuffmanNode(only->freq, only, nullptr);
    }

    // Step 2-4: merge nodes until we have a single root
    while (pq.size() > 1) {
        // Extract the two nodes with the lowest frequencies
        HuffmanNode* left  = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();

        // Create a new internal node combining both frequencies
        HuffmanNode* parent = new HuffmanNode(
            left->freq + right->freq,
            left,
            right
        );

        // Push the merged node back into the priority queue
        pq.push(parent);
    }

    // The last remaining node is the root of the Huffman Tree
    return pq.top();
}

// ------------------------------------------------------------
// generateCodes
// Traverses the Huffman Tree recursively.
//   - Going LEFT  appends '0' to the current code string.
//   - Going RIGHT appends '1' to the current code string.
//   - At a LEAF node, the accumulated string is the Huffman code
//     for that character.
// ------------------------------------------------------------
void generateCodes(HuffmanNode* root,
                   const std::string& code,
                   std::unordered_map<char, std::string>& codes) {
    if (!root) return;

    // Leaf node: assign the accumulated code to this character
    if (!root->left && !root->right) {
        // If tree has only one node, assign "0" as its code
        codes[root->ch] = code.empty() ? "0" : code;
        return;
    }

    // Recurse left  → append '0'
    generateCodes(root->left,  code + "0", codes);
    // Recurse right → append '1'
    generateCodes(root->right, code + "1", codes);
}

// ------------------------------------------------------------
// freeTree
// Post-order traversal to delete all heap-allocated nodes.
// ------------------------------------------------------------
void freeTree(HuffmanNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

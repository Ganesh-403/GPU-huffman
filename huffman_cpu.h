#pragma once

// ============================================================
// huffman_cpu.h
// Declares all data structures and functions for the CPU-based
// Huffman encoding implementation.
// ============================================================

#include <string>
#include <unordered_map>
#include <queue>
#include <vector>

// ------------------------------------------------------------
// HuffmanNode: represents a single node in the Huffman Tree.
// Leaf nodes store a real character; internal nodes store '\0'.
// ------------------------------------------------------------
struct HuffmanNode {
    char    ch;     // Character (valid only at leaf nodes)
    int     freq;   // Frequency of the character (or combined freq for internal nodes)
    HuffmanNode* left;   // Left child  (represents bit '0')
    HuffmanNode* right;  // Right child (represents bit '1')

    // Constructor for leaf nodes (actual characters)
    HuffmanNode(char c, int f)
        : ch(c), freq(f), left(nullptr), right(nullptr) {}

    // Constructor for internal nodes (combined frequency)
    HuffmanNode(int f, HuffmanNode* l, HuffmanNode* r)
        : ch('\0'), freq(f), left(l), right(r) {}
};

// ------------------------------------------------------------
// Compare: custom comparator for the min-heap priority queue.
// Nodes with LOWER frequency have HIGHER priority (min-heap).
// ------------------------------------------------------------
struct Compare {
    bool operator()(HuffmanNode* a, HuffmanNode* b) {
        return a->freq > b->freq; // greater freq = lower priority
    }
};

// ------------------------------------------------------------
// Function Declarations
// ------------------------------------------------------------

// Count character frequencies using CPU (single-threaded loop)
void cpuCountFrequency(const std::string& text, int freq[256]);

// Build the Huffman Tree from a frequency table
// Returns pointer to root node (caller must free with freeTree)
HuffmanNode* buildHuffmanTree(int freq[256]);

// Recursively generate binary Huffman codes for each character
// Fills the 'codes' map: char -> binary string (e.g., 'a' -> "101")
void generateCodes(HuffmanNode* root,
                   const std::string& code,
                   std::unordered_map<char, std::string>& codes);

// Recursively free all nodes in the Huffman Tree
void freeTree(HuffmanNode* root);

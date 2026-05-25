package com.graph.method;

import java.util.*;

public class FrequentItemsetDetection {

    public static List<BitSet> convertToBitVectors(boolean[][] adjacencyMatrix) {
        List<BitSet> bitVectors = new ArrayList<>();
        for (boolean[] row : adjacencyMatrix) {
            BitSet bitSet = new BitSet(row.length);
            for (int i = 0; i < row.length; i++) {
                if (row[i]) {
                    bitSet.set(i);
                }
            }
            bitVectors.add(bitSet);
        }
        return bitVectors;
    }

    public static Set<BitSet> findFrequentItemsets(List<BitSet> bitVectors, double minSupport) {
        int numNodes = bitVectors.size();
        Set<BitSet> frequentItemsets = new HashSet<>();
        Map<BitSet, Integer> itemsetCounts = new HashMap<>();

        for (int i = 0; i < numNodes; i++) {
            BitSet singleItemset = new BitSet(numNodes);
            singleItemset.set(i);
            itemsetCounts.put(singleItemset, 0);
        }

        while (!itemsetCounts.isEmpty()) {
            Set<BitSet> newItemsets = new HashSet<>();

            for (BitSet itemset : itemsetCounts.keySet()) {
                int count = 0;
                for (BitSet bitVector : bitVectors) {
                    if (containsItemset(bitVector, itemset)) {
                        count++;
                    }
                }
                if ((double) count / numNodes >= minSupport) {
                    frequentItemsets.add(itemset);
                    newItemsets.add(itemset);
                }
            }

            itemsetCounts.clear();
            for (BitSet itemset1 : newItemsets) {
                for (BitSet itemset2 : newItemsets) {
                    BitSet newItemset = joinItemsets(itemset1, itemset2);
                    if (newItemset != null && !itemsetCounts.containsKey(newItemset)) {
                        itemsetCounts.put(newItemset, 0);
                    }
                }
            }
        }

        return frequentItemsets;
    }

    public static boolean containsItemset(BitSet bitVector, BitSet itemset) {
        BitSet clone = (BitSet) bitVector.clone();
        clone.and(itemset);
        return clone.equals(itemset);
    }

    private static BitSet joinItemsets(BitSet itemset1, BitSet itemset2) {
        if (itemset1.equals(itemset2)) {
            return null;
        }
        BitSet joined = (BitSet) itemset1.clone();
        joined.or(itemset2);
        if (joined.cardinality() == itemset1.cardinality() + 1) {
            return joined;
        }
        return null;
    }

    public static void printFrequentItemsets(Set<BitSet> frequentItemsets) {
        System.out.println("Frequent Itemsets:");
        for (BitSet itemset : frequentItemsets) {
            System.out.print("{");
            for (int i = itemset.nextSetBit(0); i >= 0; i = itemset.nextSetBit(i+1)) {
                System.out.print(i);
                if (itemset.nextSetBit(i+1) >= 0) {
                    System.out.print(", ");
                }
            }
            System.out.println("}");
        }
    }

    public static void main(String[] args) {
        boolean[][] adjacencyMatrix = new boolean[10][10];
        Random rand = new Random(42);

        for (int i = 0; i < 10; i++) {
            for (int j = i + 1; j < 10; j++) {
                boolean connected = rand.nextDouble() < 0.3;
                adjacencyMatrix[i][j] = connected;
                adjacencyMatrix[j][i] = connected;
            }
        }

        List<BitSet> bitVectors = convertToBitVectors(adjacencyMatrix);
        double minSupport = 0.3;

        Set<BitSet> frequentItemsets = findFrequentItemsets(bitVectors, minSupport);
        printFrequentItemsets(frequentItemsets);

        System.out.println("\nNode Frequent Itemset Counts:");
        for (int i = 0; i < bitVectors.size(); i++) {
            BitSet nodeVector = bitVectors.get(i);
            int count = 0;
            for (BitSet itemset : frequentItemsets) {
                if (containsItemset(nodeVector, itemset)) {
                    count++;
                }
            }
            System.out.println("Node " + i + ": " + count);
        }
    }
}
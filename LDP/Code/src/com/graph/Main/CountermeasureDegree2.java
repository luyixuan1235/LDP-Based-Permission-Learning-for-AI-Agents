package com.graph.Main;

import com.graph.data.Data;
import com.graph.method.FrequentItemsetDetection;
import com.graph.metric.Tools;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.*;

public class CountermeasureDegree2 {
    public static final long SEED = 44;
    public boolean[][] mat;

    public int dataset;
    // 1: facebook_combined.txt
    // 2: Email-Enron.txt
    // 3: CA-AstroPh-transform.txt
    // 4: Brightkite_edges.txt (not used)
    // 5: twitter_combined_transform.txt (not used)
    // 6: gplus_combined_transform.txt

    static int metric = 1;
    // 1: local clustering coefficient

    public static double epsilon = 1.0;
    public static double percentageForMatrix = 0.9;
    public static double percentage = 0.9;

    public int totalNum;
    public int attackerNum;
    public int realUserNum;

    public int attackerEdgeNum;

    public double beta;
    public double gamma;

    public int targetNum;

    public static String attackType;

    public int[] targetNodes;
    public int[] fakeNodes;

    private static Map<Double, int[]> betaToFakeNodes = new HashMap<>();
    private static Map<Double, int[]> gammaToTargetNodes = new HashMap<>();

    public static int type = 8;

    private double edgeDensity;
    private int maxAttackEdges;


    public CountermeasureDegree2(int dataset, double beta, double gamma, double per, double[] epsilon_all, String attackType) throws Exception {
        this.dataset = dataset;
        this.beta = beta;
        this.gamma = gamma;
        CountermeasureDegree2.attackType = attackType;

        CountermeasureDegree2.percentageForMatrix = Tools.optimalPercentage[dataset - 1][metric - 1][(int) epsilon - 1];

        String filename = Tools.inputFilename[dataset - 1] + ".txt";
        this.totalNum = Tools.inputFileUserNum[dataset - 1];
        this.targetNum = (int) (totalNum * gamma);

        attackerNum = (int) (totalNum * beta);
        realUserNum = totalNum - attackerNum;
        System.out.println("realUserNum: " + realUserNum + ", attackerNum: " + attackerNum);

        mat = new boolean[totalNum][totalNum];

        if (!betaToFakeNodes.containsKey(beta)) {
            fakeNodes = generateFakeNodes(totalNum, attackerNum);
            betaToFakeNodes.put(beta, fakeNodes);
        } else {
            fakeNodes = betaToFakeNodes.get(beta);
        }

        if (!gammaToTargetNodes.containsKey(gamma)) {
            Set<Integer> fakeNodesSet = new HashSet<>();
            for (int node : fakeNodes) {
                fakeNodesSet.add(node);
            }
            targetNodes = generateTargetNodes(totalNum, targetNum, fakeNodesSet);
            gammaToTargetNodes.put(gamma, targetNodes);
        } else {
            targetNodes = gammaToTargetNodes.get(gamma);
        }

        Data.readGraph(filename, mat);

        int edgeCount = 0;

        for (int i = 0; i < totalNum; i++) {
            for (int j = i + 1; j < totalNum; j++) {
                if (mat[i][j]) {
                    edgeCount++;
                }
            }
        }
        System.out.println("Number of edges: " + edgeCount);

        attackerEdgeNum = (int) (edgeCount * gamma);

        calculateEdgeDensity();
        calculateMaxAttackEdges();

        writeRealDegreeToFile();
        writeParametersToFile(targetNodes);
        testDegreeEstimation(mat, type, epsilon_all);
    }

    private void calculateEdgeDensity() {
        int edgeCount = 0;
        int possibleEdges = totalNum * (totalNum - 1) / 2;

        for (int i = 0; i < totalNum; i++) {
            for (int j = i + 1; j < totalNum; j++) {
                if (mat[i][j]) {
                    edgeCount++;
                }
            }
        }

        edgeDensity = (double) edgeCount / possibleEdges;
        System.out.println("Edge density: " + edgeDensity);
    }

    private void calculateMaxAttackEdges() {
        int possibleAttackEdges = (attackerNum * (totalNum - attackerNum)) / 2;
        maxAttackEdges = (int) (possibleAttackEdges * edgeDensity);
        System.out.println("Maximum attack edges: " + maxAttackEdges);
    }

    private int[] generateFakeNodes(int totalNum, int attackerNum) {
        Random rand0 = new Random(SEED);
        HashSet<Integer> selectedNodes = new HashSet<>();
        while (selectedNodes.size() < attackerNum) {
            int randomIndex = rand0.nextInt(totalNum);
            selectedNodes.add(randomIndex);
        }
        return selectedNodes.stream().mapToInt(Integer::intValue).toArray();
    }

    private int[] generateTargetNodes(int totalNum, int targetNum, Set<Integer> excludeNodes) {
        Random rand0 = new Random(SEED + 1);
        HashSet<Integer> targetNodesSet = new HashSet<>();
        while (targetNodesSet.size() < targetNum) {
            int randomIndex = rand0.nextInt(totalNum);
            if (!excludeNodes.contains(randomIndex) && !targetNodesSet.contains(randomIndex)) {
                targetNodesSet.add(randomIndex);
            }
        }
        return targetNodesSet.stream().mapToInt(Integer::intValue).toArray();
    }


    public void writeRealDegreeToFile() throws Exception {
        double[] degrees = Tools.getDegree(this.mat);

        String filename = "output/degree/real/RealDegree_" + this.dataset + ".txt";

        Files.createDirectories(Paths.get("output/degree/real"));

        try (PrintStream ps = new PrintStream(new File(filename))) {
            for (double degree : degrees) {
                ps.println((int) degree);
            }
        }

        System.out.println("Real degrees have been written to " + filename);
    }

    public void writePerturbedDegreeToFile(double[] degrees, double epsilon) throws IOException {
        String baseDir = "output" + File.separator + "degree" + File.separator + "perturbed" + File.separator + dataset;
        File dir = new File(baseDir);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Failed to create directory: " + baseDir);
        }

        String filename = baseDir + File.separator + epsilon + ".txt";

        double[] noisyDegrees = Tools.addLaplaceNoise_Degree(degrees, epsilon * (1.0 - percentage));

        try (PrintStream ps = new PrintStream(new File(filename))) {
            for (double v : noisyDegrees) {
                ps.println((int) Math.round(v));
            }
        }

        System.out.println("Perturbed degrees have been written to " + filename);
    }

    public void testDegreeEstimation(boolean[][] mat, int type, double[] epsilon_all) throws Exception {
        for (int i = 0; i < epsilon_all.length; i++) {
            double ep = epsilon_all[i];

            System.out.println("dataset=" + dataset + "\t beta=" + beta + "\t gamma=" + gamma + "\t epsilon=" + ep);
            writePerturbedDegreeToFile(Tools.getDegree(mat), ep);
            testDegreeEstimation_list(mat, ep, type);
        }
    }

    public void testDegreeEstimation_list(boolean[][] mat, double epsilon, int type) throws Exception {
        boolean[][] realMat = new boolean[totalNum][totalNum];
        boolean[][] fakeMat = new boolean[totalNum][totalNum];
        boolean[][] mergedMat = new boolean[totalNum][totalNum];

        boolean[] isFakeNode = new boolean[totalNum];
        boolean[] isTargetNode = new boolean[totalNum];

        for (int node : fakeNodes) {
            isFakeNode[node] = true;
        }
        for (int node : targetNodes) {
            isTargetNode[node] = true;
        }

        for (int i = 0; i < totalNum; i++) {
            for (int j = 0; j < totalNum; j++) {
                if (isFakeNode[i] || isFakeNode[j]) {
                    fakeMat[i][j] = mat[i][j];
                } else {
                    realMat[i][j] = mat[i][j];
                }
            }
        }

        if (type == 8) {
            int connectedTargetNodes = 0;

            switch (attackType) {

                case "randomValueAttack":
                    Random randomValue = new Random(SEED);
                    int edgesPerFakeNode = readAverageDegree(dataset, epsilon);
                    int totalAddedEdges = 0;
                    System.out.println("Using " + edgesPerFakeNode + " edges per fake node for randomValueAttack");

                    for (int fakeNode : fakeNodes) {
                        List<Integer> possibleTargets = new ArrayList<>();
                        for (int j = 0; j < totalNum; j++) {
                            if (fakeNode != j && !fakeMat[fakeNode][j]) {
                                possibleTargets.add(j);
                            }
                        }

                        Collections.shuffle(possibleTargets, randomValue);

                        int addedEdges = 0;
                        for (int targetNode : possibleTargets) {
                            if (addedEdges >= edgesPerFakeNode) break;
                            fakeMat[fakeNode][targetNode] = true;
                            fakeMat[targetNode][fakeNode] = true;
                            addedEdges++;
                            totalAddedEdges++;
                            if (isTargetNode[targetNode]) {
                                connectedTargetNodes++;
                            }
                        }
                    }

                    System.out.println("RandomValueAttack: Added " + totalAddedEdges + " edges (" + edgesPerFakeNode + " per fake node)");
                    System.out.println("RandomValueAttack: Connected to " + connectedTargetNodes + " target nodes");

                    break;
                case "randomNodeAttack":

                    Random randomNode = new Random(SEED);
                    int addedNodeEdges = 0;
                    for (int node : fakeNodes) {
                        int targetNode = targetNodes[randomNode.nextInt(targetNodes.length)];
                        fakeMat[node][targetNode] = true;
                        fakeMat[targetNode][node] = true;
                        addedNodeEdges++;
                        connectedTargetNodes++;
                    }
                    System.out.println("RandomNodeAttack: Added " + addedNodeEdges + " edges");
                    System.out.println("RandomNodeAttack: Connected to " + connectedTargetNodes + " target nodes");
                    break;
                case "maximumGainAttack":
                    Random random = new Random(SEED);
                    int edgesPerFakeGainNode = readAverageDegree(dataset, epsilon);
                    int totalAddedGainEdges = 0;
                    int totalPossibleTargets = 0;

                    for (int fakeNode : fakeNodes) {
                        List<Integer> possibleTargets = new ArrayList<>();

                        for (int targetNode : targetNodes) {
                            if (fakeNode != targetNode && !fakeMat[fakeNode][targetNode]) {
                                possibleTargets.add(targetNode);
                            }
                        }
                        for (int otherFakeNode : fakeNodes) {
                            if (fakeNode != otherFakeNode && !fakeMat[fakeNode][otherFakeNode]) {
                                possibleTargets.add(otherFakeNode);
                            }
                        }

                        totalPossibleTargets += possibleTargets.size();
                        Collections.shuffle(possibleTargets, random);

                        int addedEdges = 0;

                        for (int targetNode : possibleTargets) {
                            if (addedEdges >= edgesPerFakeGainNode) break;
                            fakeMat[fakeNode][targetNode] = true;
                            fakeMat[targetNode][fakeNode] = true;
                            addedEdges++;
                            totalAddedGainEdges++;
                            if (isTargetNode[targetNode]) {
                                connectedTargetNodes++;
                            }
                        }


                        if (addedEdges < edgesPerFakeGainNode) {
                            List<Integer> remainingNodes = new ArrayList<>();
                            for (int i = 0; i < totalNum; i++) {
                                if (i != fakeNode && !fakeMat[fakeNode][i]) {
                                    remainingNodes.add(i);
                                }
                            }
                            Collections.shuffle(remainingNodes, random);

                            for (int remainRandomNode : remainingNodes) {
                                if (addedEdges >= edgesPerFakeGainNode) break;
                                fakeMat[fakeNode][remainRandomNode] = true;
                                fakeMat[remainRandomNode][fakeNode] = true;
                                addedEdges++;
                                totalAddedGainEdges++;
                                if (isTargetNode[remainRandomNode]) {
                                    connectedTargetNodes++;
                                }
                            }
                        }
                    }

                    System.out.println("MaximumGainAttack: Added " + totalAddedGainEdges + " edges (" + edgesPerFakeGainNode + " per fake node)");
                    System.out.println("MaximumGainAttack: Connected to " + connectedTargetNodes + " target nodes");
                    break;
            }
        }

        for (int i = 0; i < totalNum; i++) {
            for (int j = 0; j < totalNum; j++) {
                mergedMat[i][j] = realMat[i][j] || fakeMat[i][j];
            }
        }

        double[] noiseDegree = Tools.getDegree(mergedMat);
        noiseDegree = Tools.addLaplaceNoise_Degree(noiseDegree, epsilon * (1.0 - percentage));

        double minSupport = 0.05;
        int frequentItemsetThreshold = 150;
        identifyAndCompareFakeNodes(mergedMat, noiseDegree, minSupport, frequentItemsetThreshold, isTargetNode);

        String baseDir = "output" + File.separator + "counter" + File.separator + attackType + File.separator + dataset;
        File dir = new File(baseDir);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new Exception("Failed to create directory: " + baseDir);
        }

        String filename = baseDir + File.separator + frequentItemsetThreshold + ".txt";

        try (PrintStream ps = new PrintStream(new File(filename))) {
            for (double v : noiseDegree) {
                ps.println((int) Math.round(v));
            }
        }
    }

    private void identifyAndCompareFakeNodes(boolean[][] perturbedMat, double[] noisyDegree, double minSupport, int frequentItemsetThreshold, boolean[] isTargetNode) {
        List<BitSet> bitVectors = FrequentItemsetDetection.convertToBitVectors(perturbedMat);
        Set<BitSet> frequentItemsets = FrequentItemsetDetection.findFrequentItemsets(bitVectors, minSupport);

        List<Integer> potentialFakeNodes = new ArrayList<>();
        for (int i = 0; i < bitVectors.size(); i++) {
            BitSet nodeVector = bitVectors.get(i);
            int containedFrequentItemsets = 0;
            for (BitSet itemset : frequentItemsets) {
                if (FrequentItemsetDetection.containsItemset(nodeVector, itemset)) {
                    containedFrequentItemsets++;
                }
            }
            if (containedFrequentItemsets >= frequentItemsetThreshold) {
                potentialFakeNodes.add(i);
            }
        }

        Set<Integer> actualFakeNodesSet = new HashSet<>();
        for (int fakeNode : fakeNodes) {
            actualFakeNodesSet.add(fakeNode);
        }

        Set<Integer> detectedFakeNodesSet = new HashSet<>(potentialFakeNodes);

        for (int targetNode : targetNodes) {
            int fakeConnections = 0;
            for (int detectedFakeNode : detectedFakeNodesSet) {
                if (perturbedMat[targetNode][detectedFakeNode]) {
                    fakeConnections++;
                }
            }
            noisyDegree[targetNode] = Math.max(0, noisyDegree[targetNode] - fakeConnections);
        }

        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        for (int node : detectedFakeNodesSet) {
            if (actualFakeNodesSet.contains(node)) {
                truePositives++;
            } else {
                falsePositives++;
            }
        }

        for (int node : actualFakeNodesSet) {
            if (!detectedFakeNodesSet.contains(node)) {
                falseNegatives++;
            }
        }

        double precision = (double) truePositives / (truePositives + falsePositives);
        double recall = (double) truePositives / (truePositives + falseNegatives);
        double f1Score = 2 * (precision * recall) / (precision + recall);
    }

    private int readAverageDegree(int dataset, double epsilon) {
        String filename = "output/avg_degree/" + dataset + "/" + epsilon + ".txt";
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String line = reader.readLine();
            reader.close();
            return (int) Math.round(Double.parseDouble(line));
        } catch (IOException e) {
            System.err.println("Error reading average degree file: " + e.getMessage());
            return 46;
        } catch (NumberFormatException e) {
            System.err.println("Error parsing average degree: " + e.getMessage());
            return 46;
        }
    }

    public void writeParametersToFile(int[] targetNodes) throws Exception {
        String dirPath = "output" + File.separator + "degree" + File.separator + "config" + File.separator + dataset;
        File dir = new File(dirPath);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new Exception("Failed to create directory: " + dirPath);
        }

        String filename = dirPath + File.separator + "parameters_" + beta + "_" + gamma + ".txt";

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("totalNum: " + totalNum);
            writer.newLine();
            writer.write("attackerNum: " + attackerNum);
            writer.newLine();
            writer.write("targetNum: " + targetNum);
            writer.newLine();

            for (int i = 0; i < targetNodes.length; i++) {
                writer.write("targetNode " + (i + 1) + ": " + targetNodes[i]);
                writer.newLine();
            }

            for (int i = 0; i < fakeNodes.length; i++) {
                writer.write("fakeNode " + (i + 1) + ": " + fakeNodes[i]);
                writer.newLine();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        int dataset = 1;

        double fixPercentage = 0.3;
        double[] fixedEpsilon = {4.0};
        double[] epsilonAll = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};

        double defaultBeta = 0.05;
        double defaultGamma = 0.05;

        String[] attackTypes = {"randomValueAttack", "randomNodeAttack", "maximumGainAttack"};

        double[] beta_values = {0.001, 0.005, 0.01, 0.1, 0.15};
        double[] gamma_values = {0.001, 0.005, 0.01, 0.1, 0.15};

        new CountermeasureDegree2(dataset, defaultBeta, defaultGamma, fixPercentage, fixedEpsilon, attackTypes[2]);
    }
}

package com.graph.Main;

import com.graph.data.Data;
import com.graph.data.NeighborListRandomization;
import com.graph.method.LCCEstimation;
import com.graph.metric.ClusteringCoefficient;
import com.graph.metric.Tools;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.*;

public class CountermeasureLCC1 {
    public static final long SEED = 43;
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

    public CountermeasureLCC1(int dataset, double beta, double gamma, double per, double[] epsilon_all, String attackType) throws Exception {
        this.dataset = dataset;
        this.beta = beta;
        this.gamma = gamma;
        CountermeasureLCC1.attackType = attackType;

        CountermeasureLCC1.percentageForMatrix = Tools.optimalPercentage[dataset - 1][metric - 1][(int) epsilon - 1];

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
        writeRealCoefficientToFile();
        writeParametersToFile(targetNodes);
        testLCCEstimation(mat, type, epsilon_all, per, true);
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

    public void writeRealCoefficientToFile() throws Exception {
        double[] coefficient = ClusteringCoefficient.getLocalClusteringCoefficientList(this.mat);
        String filename = "output/real/RealClusteringCoefficient_" + this.dataset + ".txt";
        Files.createDirectories(Paths.get("output/real"));

        try (PrintStream ps = new PrintStream(new File(filename))) {
            for (int i = 0; i < coefficient.length; i++) {
                ps.println(coefficient[i]);
            }
        }
    }

    public void testLCCEstimation(boolean[][] mat, int type, double[] epsilon_all, double per, boolean optimal) throws Exception {
        for (int i = 0; i < epsilon_all.length; i++) {
            double ep = epsilon_all[i];
            if (optimal) {
                per = Tools.optimalPercentage[dataset - 1][metric - 1][(int) ep - 1];
            }
            System.out.println("dataset=" + dataset + "\t beta=" + beta + "\t gamma=" + gamma + "\t epsilon=" + ep + "\t percentage=" + per);
            testLCCEstimation_list(mat, ep, per, type);
        }
    }

    public void testLCCEstimation_list(boolean[][] mat, double epsilon, double percentage, int type) throws Exception {
        boolean[][] realMat = new boolean[totalNum][totalNum];
        boolean[][] fakeMat = new boolean[totalNum][totalNum];
        boolean[][] perturbedRealMat = new boolean[totalNum][totalNum];
        boolean[][] perturbedFakeMat = new boolean[totalNum][totalNum];
        boolean[][] emptyFakeMat = new boolean[totalNum][totalNum];
        boolean[][] mergedMat = new boolean[totalNum][totalNum];
        boolean[][] perturbedMat = new boolean[totalNum][totalNum];

        double[] perturbed_coefficient = null;

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
                            addedEdges++;
                            totalAddedEdges++;
                        }
                    }

                    System.out.println("RandomValueAttack: Added " + totalAddedEdges + " edges (" + edgesPerFakeNode + " per fake node)");

                    break;
                case "randomNodeAttack":

                    Random randomNode = new Random(SEED);
                    int addedNodeEdges = 0;
                    for (int node : fakeNodes) {
                        int targetNode = targetNodes[randomNode.nextInt(targetNodes.length)];
                        fakeMat[node][targetNode] = true;
                        fakeMat[targetNode][node] = true;
                        addedNodeEdges++;
                    }
                    System.out.println("RandomNodeAttack: Added " + addedNodeEdges + " edges");
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
                            }
                        }
                    }

                    System.out.println("MaximumGainAttack: Added " + totalAddedGainEdges + " edges (" + edgesPerFakeGainNode + " per fake node)");
                    System.out.println("Average possibleTargets size: " + (totalPossibleTargets / fakeNodes.length));
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

        int maxDegree = 0;
        for (double degree : noiseDegree) {
            maxDegree = Math.max(maxDegree, (int) degree);
        }

        switch (attackType) {
            case "randomValueAttack":
                for (int node : fakeNodes) {
                    noiseDegree[node] = (int) (Math.random() * (totalNum));
                }
        }

        perturbedRealMat = NeighborListRandomization.randomize_half(realMat, epsilon * percentage);

        switch (attackType) {
            case "randomValueAttack":
                perturbedFakeMat = fakeMat;
                break;
            case "randomNodeAttack":
                perturbedFakeMat = NeighborListRandomization.randomize_half(fakeMat, epsilon * percentage);
                break;
            case "maximumGainAttack":
                perturbedFakeMat = fakeMat;
                break;
        }

        for (int i = 0; i < totalNum; i++) {
            for (int j = 0; j < totalNum; j++) {
                perturbedMat[i][j] = perturbedRealMat[i][j] || perturbedFakeMat[i][j];
            }
        }

        double[] perturbedVectorDegree = Tools.getDegree(perturbedMat);
        detectAndEvaluateFakeNodes(perturbedMat, perturbedVectorDegree, noiseDegree, isFakeNode, attackerNum, percentage, epsilon, isTargetNode);

        for (int i = 0; i < totalNum; i++) {
            for (int j = i + 1; j < totalNum; j++) {
                perturbedMat[i][j] = perturbedMat[j][i];
            }
        }

        double[] noisyDegreeFromVector = NeighborListRandomization.calibrate_randomize_all_degree(Tools.getDegree(perturbedMat), epsilon * percentage);

        double variance1 = 8.0 / (epsilon * (1.0 - percentage) * epsilon * (1.0 - percentage));
        double[] variance2 = Tools.getVariance_list(noiseDegree, epsilon * percentage);

        double[] noisyDegree = Tools.mergeTwoNoisyDegree_list(noiseDegree, variance1, noisyDegreeFromVector, variance2);

        perturbed_coefficient = LCCEstimation.LCC_SixCases_list(perturbedMat, noisyDegree, epsilon, percentage);

        String baseDir = "output" + File.separator + "counter_metric_degree" + File.separator + attackType + File.separator + dataset;
        File dir = new File(baseDir);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new Exception("Failed to create directory: " + baseDir);
        }

        String filename;
        if (percentage == 0.3 || percentage == 0.5 || percentage == 0.7 || percentage == 0.9) {
            int flag = (int) ((percentage) * 100 + (1 - percentage) * 10);
            filename = baseDir + File.separator + flag + "_" + epsilon + ".txt";
        } else {
            filename = baseDir + File.separator + epsilon + "_" + beta + "_" + gamma + ".txt";
        }

        PrintStream ps = new PrintStream(new File(filename));
        for (int i = 0; i < mat[0].length; i++) {
            ps.println(perturbed_coefficient[i]);
        }
        ps.close();
    }

    private void detectAndEvaluateFakeNodes(boolean[][] perturbedMat, double[] perturbedVectorDegree, double[] noiseDegree, boolean[] isFakeNode, int actualFakeNodesCount, double percentage, double epsilon, boolean[] isTargetNode) {
        boolean[] potentialFakeNodes = new boolean[totalNum];
        int potentialFakeNodeCount = 0;

        double laplaceScale = 2.0 / (epsilon * (1.0 - percentage));
        double threshold = 3 * Math.sqrt(2) * laplaceScale;

        double maxPerturbedDegree = Double.MIN_VALUE;
        for (double degree : perturbedVectorDegree) {
            if (degree > maxPerturbedDegree) {
                maxPerturbedDegree = degree;
            }
        }

        double detectionThreshold = maxPerturbedDegree + threshold;

        System.out.println(maxPerturbedDegree);

        for (int i = 0; i < totalNum; i++) {
            if (Math.abs(perturbedVectorDegree[i] - noiseDegree[i]) > detectionThreshold && !isTargetNode[i]) {
                potentialFakeNodes[i] = true;
                potentialFakeNodeCount++;
            }
        }

        Map<Integer, BitSet> bitVectors = new HashMap<>();
        for (int i = 0; i < totalNum; i++) {
            BitSet vector = new BitSet(totalNum);
            for (int j = 0; j < totalNum; j++) {
                if (perturbedMat[i][j]) {
                    vector.set(j);
                }
            }
            bitVectors.put(i, vector);
        }

        for (int i = 0; i < totalNum; i++) {
            if (potentialFakeNodes[i]) {
                BitSet fakeNodeVector = bitVectors.get(i);
                for (int j = 0; j < totalNum; j++) {
                    if (i != j) {
                        BitSet otherNodeVector = bitVectors.get(j);
                        boolean connectionInFakeVector = fakeNodeVector.get(j);
                        boolean connectionInOtherVector = otherNodeVector.get(i);

                        perturbedMat[i][j] = connectionInOtherVector;
                        perturbedMat[j][i] = connectionInOtherVector;

                        if (connectionInOtherVector) {
                            if (!connectionInFakeVector) {
                                noiseDegree[i]++;
                            }
                        } else {
                            if (connectionInFakeVector) {
                                noiseDegree[i]--;
                            }
                        }
                    }
                }
            }
        }

        int correctlyIdentified = 0;
        int falsePositives = 0;

        for (int i = 0; i < totalNum; i++) {
            if (potentialFakeNodes[i]) {
                if (isFakeNode[i]) {
                    correctlyIdentified++;
                } else {
                    falsePositives++;
                }
            }
        }

        int falseNegatives = actualFakeNodesCount - correctlyIdentified;
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
        String dirPath = "output" + File.separator + "config" + File.separator + dataset;
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

        for (double beta : beta_values) {
            System.out.println("Running experiment with beta = " + beta);
            new CountermeasureLCC1(dataset, beta, defaultGamma, fixPercentage, fixedEpsilon, attackTypes[0]);
        }
    }
}

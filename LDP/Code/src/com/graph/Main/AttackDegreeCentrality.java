package com.graph.Main;

import com.graph.data.Data;
import com.graph.metric.Tools;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class AttackDegreeCentrality {
    public static final long SEED = 44;
    public boolean[][] mat;

    public int dataset;
    static int metric = 1;
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

    // 移除静态缓存，避免跨实验的数据污染
    // private static Map<Double, int[]> betaToFakeNodes = new HashMap<>();
    // private static Map<Double, int[]> gammaToTargetNodes = new HashMap<>();
    // private static Set<Double> writtenEpsilons = new HashSet<>();

    public static int type = 8;

    private double edgeDensity;
    private int maxAttackEdges;

    public AttackDegreeCentrality(int dataset, double beta, double gamma, double per, double[] epsilon_all, String attackType) throws Exception {
        this.dataset = dataset;
        this.beta = beta;
        this.gamma = gamma;
        AttackDegreeCentrality.attackType = attackType;

        AttackDegreeCentrality.percentageForMatrix = Tools.optimalPercentage[dataset - 1][metric - 1][(int) epsilon - 1];

        String filename = Tools.inputFilename[dataset - 1] + ".txt";
        this.totalNum = Tools.inputFileUserNum[dataset - 1];
        this.targetNum = (int) (totalNum * gamma);

        attackerNum = (int) (totalNum * beta);
        realUserNum = totalNum - attackerNum;
        System.out.println("realUserNum: " + realUserNum + ", attackerNum: " + attackerNum);

        mat = new boolean[totalNum][totalNum];

        // 每次都重新生成fakeNodes，确保实验独立性
        fakeNodes = generateFakeNodes(totalNum, attackerNum);

        // 每次都重新生成targetNodes，确保基于当前的fakeNodes
        Set<Integer> fakeNodesSet = new HashSet<>();
        for (int node : fakeNodes) {
            fakeNodesSet.add(node);
        }
        targetNodes = generateTargetNodes(totalNum, targetNum, fakeNodesSet);

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
            for (int i = 0; i < degrees.length; i++) {
                ps.println((int) degrees[i]);
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

    private int[] readTargetNodesFromConfigFile() throws IOException {
        String configFile = "output" + File.separator + "degree" + File.separator + "config" +
                File.separator + dataset + File.separator + "parameters_" + beta + "_" + gamma + ".txt";
        List<Integer> targetNodeList = new ArrayList<>();
        boolean readingTargetNodes = false;
        try (BufferedReader br = new BufferedReader(new FileReader(configFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.equals("Target Nodes:")) {
                    readingTargetNodes = true;
                    continue;
                }
                if (line.equals("Fake Nodes:")) break;
                if (readingTargetNodes && line.startsWith("targetNode")) {
                    String[] parts = line.split(":");
                    if (parts.length == 2) {
                        targetNodeList.add(Integer.parseInt(parts[1].trim()));
                    }
                }
            }
        }
        return targetNodeList.stream().mapToInt(Integer::intValue).toArray();
    }

    public void writeTargetNodeDegreesAfterAttack(boolean[][] mergedMat, double epsilon) throws IOException {
        double[] degreesAfterAttack = Tools.getDegree(mergedMat);
        double[] noisyDegrees = Tools.addLaplaceNoise_Degree(degreesAfterAttack, epsilon * (1.0 - percentage));
        int[] targetNodeIds = readTargetNodesFromConfigFile();
        String baseDir = "output" + File.separator + "degree" + File.separator + "attackedtargeted" +
                File.separator + attackType + File.separator + dataset;
        File dir = new File(baseDir);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Failed to create directory: " + baseDir);
        }
        String filename = baseDir + File.separator + epsilon + "_" + beta + "_" + gamma + ".txt";
        try (PrintStream ps = new PrintStream(new File(filename))) {
            for (int targetNodeId : targetNodeIds) {
                if (targetNodeId >= 0 && targetNodeId < noisyDegrees.length) {
                    ps.println((int) Math.round(noisyDegrees[targetNodeId]));
                }
            }
        }
        System.out.println("Target node perturbed degrees after " + attackType + " attack have been written to " + filename);
    }

    public void writeAllNodeDegreesAfterAttack(boolean[][] mergedMat, double epsilon) throws IOException {
        double[] degreesAfterAttack = Tools.getDegree(mergedMat);
        double[] noisyDegrees = Tools.addLaplaceNoise_Degree(degreesAfterAttack, epsilon * (1.0 - percentage));
        String baseDir = "output" + File.separator + "degree" + File.separator + "attackedtargeted" +
                File.separator + attackType + File.separator + dataset;
        File dir = new File(baseDir);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Failed to create directory: " + baseDir);
        }
        String filename = baseDir + File.separator + epsilon + "_" + beta + "_" + gamma + "_.txt";
        try (PrintStream ps = new PrintStream(new File(filename))) {
            for (int i = 0; i < noisyDegrees.length; i++) {
                ps.println((int) Math.round(noisyDegrees[i]));
            }
        }
        System.out.println("All node perturbed degrees after " + attackType + " attack have been written to " + filename);
    }

    public void writeTargetNodePerturbedDegrees(boolean[][] originalMat, double epsilon) throws IOException {
        double[] degreesBeforeAttack = Tools.getDegree(originalMat);
        int[] targetNodeIds = readTargetNodesFromConfigFile();
        String baseDir = "output" + File.separator + "degree" + File.separator + "perturbtargeted";
        File dir = new File(baseDir);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Failed to create directory: " + baseDir);
        }
        String filename = baseDir + File.separator + epsilon + ".txt";
        double[] noisyDegrees = Tools.addLaplaceNoise_Degree(degreesBeforeAttack, epsilon * (1.0 - percentage));
        try (PrintStream ps = new PrintStream(new File(filename))) {
            for (int targetNodeId : targetNodeIds) {
                if (targetNodeId >= 0 && targetNodeId < noisyDegrees.length) {
                    ps.println((int) Math.round(noisyDegrees[targetNodeId]));
                }
            }
        }
        System.out.println("Perturbed degrees for epsilon " + epsilon + " have been written to " + filename);
    }

    public void testDegreeEstimation(boolean[][] mat, int type, double[] epsilon_all) throws Exception {
        for (double ep : epsilon_all) {
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

        for (int node : fakeNodes) isFakeNode[node] = true;
        for (int node : targetNodes) isTargetNode[node] = true;

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
                            if (isTargetNode[targetNode]) connectedTargetNodes++;
                        }
                    }
                    System.out.println("RandomValueAttack: Added " + totalAddedEdges + " edges");
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
                    break;
                case "maximumGainAttack":
                    Random random = new Random(SEED);
                    int edgesPerFakeGainNode = readAverageDegree(dataset, epsilon);
                    int totalAddedGainEdges = 0;
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
                        Collections.shuffle(possibleTargets, random);
                        int addedEdges = 0;
                        for (int targetNode : possibleTargets) {
                            if (addedEdges >= edgesPerFakeGainNode) break;
                            fakeMat[fakeNode][targetNode] = true;
                            fakeMat[targetNode][fakeNode] = true;
                            addedEdges++;
                            totalAddedGainEdges++;
                            if (isTargetNode[targetNode]) connectedTargetNodes++;
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
                                if (isTargetNode[remainRandomNode]) connectedTargetNodes++;
                            }
                        }
                    }
                    System.out.println("MaximumGainAttack: Added " + totalAddedGainEdges + " edges");
                    break;
                case "untargetedAttacked":
                    System.out.println("Using Maximum Attack Strategy: Fake nodes connect to all other nodes");
                    int totalEdgesAdded = 0;
                    for (int fakeNode : fakeNodes) {
                        for (int otherNode = 0; otherNode < totalNum; otherNode++) {
                            if (fakeNode != otherNode && !fakeMat[fakeNode][otherNode]) {
                                fakeMat[fakeNode][otherNode] = true;
                                fakeMat[otherNode][fakeNode] = true;
                                totalEdgesAdded++;
                            }
                        }
                    }
                    System.out.println("Maximum Attack: Added " + totalEdgesAdded + " edges");
                    break;
            }
        }

        for (int i = 0; i < totalNum; i++) {
            for (int j = 0; j < totalNum; j++) {
                mergedMat[i][j] = realMat[i][j] || fakeMat[i][j];
            }
        }

        if (attackType.equals("untargetedAttacked")) {
            writeAllNodeDegreesForUntargetedAttack(mat, mergedMat, epsilon);
        } else {
            writeAllNodeDegreesAfterAttack(mergedMat, epsilon);
        }
        writeTargetNodePerturbedDegrees(mat, epsilon);
    }

    public void writeAllNodeDegreesForUntargetedAttack(boolean[][] originalMat, boolean[][] mergedMat, double epsilon) throws IOException {
        double[] degreesBeforeAttack = Tools.getDegree(originalMat);
        double[] degreesAfterAttack = Tools.getDegree(mergedMat);
        double[] noisyDegreesBefore = Tools.addLaplaceNoise_Degree(degreesBeforeAttack, epsilon * (1.0 - percentage));
        double[] noisyDegreesAfter = Tools.addLaplaceNoise_Degree(degreesAfterAttack, epsilon * (1.0 - percentage));

        String baseDir = "output" + File.separator + "degree" + File.separator + "untargeted_all_nodes" + File.separator + dataset;
        File dir = new File(baseDir);
        if (!dir.exists() && !dir.mkdirs()) throw new IOException("Failed to create directory: " + baseDir);

        String filenameBefore = baseDir + File.separator + "before_attack_" + epsilon + "_" + beta + "_" + gamma + ".txt";
        try (PrintStream ps = new PrintStream(new File(filenameBefore))) {
            for (double v : noisyDegreesBefore) ps.println((int) Math.round(v));
        }

        String filenameAfter = baseDir + File.separator + "after_attack_" + epsilon + "_" + beta + "_" + gamma + ".txt";
        try (PrintStream ps = new PrintStream(new File(filenameAfter))) {
            for (double v : noisyDegreesAfter) ps.println((int) Math.round(v));
        }

        String filenameChange = baseDir + File.separator + "degree_change_" + epsilon + "_" + beta + "_" + gamma + ".txt";
        try (PrintStream ps = new PrintStream(new File(filenameChange))) {
            for (int i = 0; i < noisyDegreesAfter.length; i++) {
                ps.println((int) Math.round(noisyDegreesAfter[i]) - (int) Math.round(noisyDegreesBefore[i]));
            }
        }

        String filenameDetailed = baseDir + File.separator + "detailed_" + epsilon + "_" + beta + "_" + gamma + ".txt";
        try (PrintStream ps = new PrintStream(new File(filenameDetailed))) {
            ps.println("NodeID,DegreeBefore,DegreeAfter,DegreeChange,NodeType");
            Set<Integer> fakeNodesSet = new HashSet<>();
            for (int node : fakeNodes) fakeNodesSet.add(node);
            Set<Integer> targetNodesSet = new HashSet<>();
            for (int node : targetNodes) targetNodesSet.add(node);
            for (int i = 0; i < totalNum; i++) {
                String type = fakeNodesSet.contains(i) ? "Fake" : (targetNodesSet.contains(i) ? "Target" : "Real");
                int before = (int) Math.round(noisyDegreesBefore[i]);
                int after = (int) Math.round(noisyDegreesAfter[i]);
                ps.println(i + "," + before + "," + after + "," + (after - before) + "," + type);
            }
        }

        System.out.println("Untargeted attack files written:\n  Before: " + filenameBefore + "\n  After: " + filenameAfter + "\n  Change: " + filenameChange + "\n  Detailed: " + filenameDetailed);
    }

    private int readAverageDegree(int dataset, double epsilon) {
        String filename = "output/avg_degree/" + dataset + "/" + epsilon + ".txt";
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            return (int) Math.round(Double.parseDouble(reader.readLine()));
        } catch (IOException | NumberFormatException e) {
            System.err.println("Error reading average degree file: " + e.getMessage());
            return 46;
        }
    }

    public void writeParametersToFile(int[] targetNodes) throws Exception {
        String dirPath = "output" + File.separator + "degree" + File.separator + "config" + File.separator + dataset;
        File dir = new File(dirPath);
        if (!dir.exists() && !dir.mkdirs()) throw new Exception("Failed to create directory: " + dirPath);
        String filename = dirPath + File.separator + "parameters_" + beta + "_" + gamma + ".txt";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("totalNum: " + totalNum);
            writer.newLine();
            writer.write("attackerNum: " + attackerNum);
            writer.newLine();
            writer.write("targetNum: " + targetNum);
            writer.newLine();
            writer.write("attackType: " + attackType);
            writer.newLine();
            writer.write("beta: " + beta);
            writer.newLine();
            writer.write("gamma: " + gamma);
            writer.newLine();
            writer.write("Target Nodes:");
            writer.newLine();
            for (int i = 0; i < targetNodes.length; i++) {
                writer.write("targetNode " + (i + 1) + ": " + targetNodes[i]);
                writer.newLine();
            }
            writer.write("Fake Nodes:");
            writer.newLine();
            for (int i = 0; i < fakeNodes.length; i++) {
                writer.write("fakeNode " + (i + 1) + ": " + fakeNodes[i]);
                writer.newLine();
            }
        }
        System.out.println("Parameters have been written to " + filename);
    }

    public static void main(String[] args) throws Exception {
        int dataset = 1;
        double fixPercentage = 0.3;
        double[] fixedEpsilon = {4.0};
        double[] epsilonAll = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
        double defaultBeta = 0.05;
        double defaultGamma = 0.05;
        String[] attackTypes = {"randomValueAttack", "randomNodeAttack", "maximumGainAttack", "untargetedAttacked"};
        double[] beta_values = {0.001, 0.005, 0.01, 0.1, 0.15};
        double[] gamma_values = {0.001, 0.005, 0.01, 0.1, 0.15};

        for (String attackType : attackTypes) {
            System.out.println("===== Running experiments for attack type: " + attackType + " =====");

            System.out.println("----- Testing different epsilon values -----");
            for (double epsilon : epsilonAll) {
                System.out.println("Running experiment with epsilon = " + epsilon);
                new AttackDegreeCentrality(dataset, defaultBeta, defaultGamma, fixPercentage, new double[]{epsilon}, attackType);
            }

            System.out.println("----- Testing different beta values -----");
            for (double beta : beta_values) {
                System.out.println("Running experiment with beta = " + beta);
                new AttackDegreeCentrality(dataset, beta, defaultGamma, fixPercentage, fixedEpsilon, attackType);
            }

            System.out.println("----- Testing different gamma values -----");
            for (double gamma : gamma_values) {
                System.out.println("Running experiment with gamma = " + gamma);
                new AttackDegreeCentrality(dataset, defaultBeta, gamma, fixPercentage, fixedEpsilon, attackType);
            }
        }
        System.out.println("===== All experiments completed =====");
    }
}

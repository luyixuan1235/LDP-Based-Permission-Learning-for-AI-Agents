package com.graph.Main;

import com.graph.data.Data;
import com.graph.data.NeighborListRandomization;
import com.graph.method.LCCEstimation;
import com.graph.metric.ClusteringCoefficient;
import com.graph.metric.Tools;

import java.io.File;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class PerturbedLCC {
    public int userNum;
    public boolean[][] mat;

    public int dataset;
    static int metric = 1;

    public static double epsilon = 1.0;
    public static double percentageForMatrix = 0.9;

    public static int type = 8;

    public PerturbedLCC(int dataset, double per, double[] epsilon_all) throws Exception {
        this.dataset = dataset;
        PerturbedLCC.percentageForMatrix = Tools.optimalPercentage[dataset - 1][metric - 1][(int) epsilon - 1];
        String filename = Tools.inputFilename[dataset - 1] + ".txt";
        this.userNum = Tools.inputFileUserNum[dataset - 1];
        mat = new boolean[userNum][userNum];

        Data.readGraph(filename, mat);
        writeRealCoefficientToFile();
        testLCCEstimation(mat, type, epsilon_all, per, true);
    }

    public void writeRealCoefficientToFile() throws Exception {
        double[] coefficient = ClusteringCoefficient.getLocalClusteringCoefficientList(this.mat);
        String filename = "output/real/RealClusteringCoefficient_" + this.dataset + ".txt";
        PrintStream ps = new PrintStream(new File(filename));
        for (int i = 0; i < coefficient.length; i++)
            ps.println(coefficient[i]);
        ps.close();
    }

    public void testLCCEstimation(boolean[][] mat, int type, double[] epsilon_all, double per, boolean optimal) throws Exception {
        for (int i = 0; i < epsilon_all.length; i++) {
            double ep = epsilon_all[i];
            if (optimal)
                per = Tools.optimalPercentage[dataset - 1][metric - 1][(int) ep - 1];

            System.out.println("dataset=" + dataset + "\t epsilon=" + ep + "\t percentage=" + per);
            testLCCEstimation_list(mat, ep, per, type);
        }
    }

    public void testLCCEstimation_list(boolean[][] mat, double epsilon, double percentage, int type) throws Exception {
        double[] degree = Tools.getDegree(mat);
        double[] noisyDegree1 = Tools.addLaplaceNoise_Degree(degree, epsilon * (1.0 - percentage));

        boolean[][] perturbedMat = null;
        double[] perturbed_coefficient = null;

        if (type == 8) {
            perturbedMat = NeighborListRandomization.randomize_half(mat, epsilon * percentage);
            double[] rawDegree = Tools.getDegree(perturbedMat);
            double avgDegree = Arrays.stream(rawDegree).average().orElse(Double.NaN);
            File dirAvgDeg = new File("output/avg_degree/" + dataset);
            if (!dirAvgDeg.exists()) {
                dirAvgDeg.mkdirs();
            }

            String avgDegreeFilename = "output/avg_degree/" + dataset + "/" + epsilon + ".txt";
            try (PrintStream ps = new PrintStream(new File(avgDegreeFilename))) {
                ps.println(avgDegree);
            }

            double[] noisyDegreeFromVector = NeighborListRandomization.calibrate_randomize_all_degree(Tools.getDegree(perturbedMat), epsilon * percentage);
            double variance1 = 8.0 / (epsilon * (1.0 - percentage) * epsilon * (1.0 - percentage));
            double[] variance2 = Tools.getVariance_list(noisyDegree1, epsilon * percentage);
            double[] noisyDegree = Tools.mergeTwoNoisyDegree_list(noisyDegree1, variance1, noisyDegreeFromVector, variance2);
            perturbed_coefficient = LCCEstimation.LCC_SixCases_list(perturbedMat, noisyDegree, epsilon, percentage);
        }

        File dir = new File("output/perturbed/");
        if (!dir.exists()) {
            dir.mkdirs();
        }

        String filename = "";
        Path filePath;

        if (percentage == 0.3 || percentage == 0.5 || percentage == 0.7 || percentage == 0.9) {
            int flag = (int) ((percentage) * 100 + (1 - percentage) * 10);
            filename = "output/perturbed/" + flag + "_" + dataset + "_" + epsilon + ".txt";
        } else {
            filename = "output/perturbed/" + dataset + "/" + epsilon + ".txt";
        }

        filePath = Paths.get(filename);

        Files.createDirectories(filePath.getParent());

        if (!Files.exists(filePath)) {
            Files.createFile(filePath);
        }

        PrintStream ps = new PrintStream(new File(filename));
        for (int i = 0; i < mat[0].length; i++) {
            ps.println(perturbed_coefficient[i]);
        }
        ps.close();
    }

    public double[] baseline_oneNode(boolean[][] mat, double epsilon, int type, boolean calibrated) {
        boolean[][] perturbedMat = null;
        double[] perturbed_coefficient = new double[mat[0].length];
        double[] noisyDegree;
        if (type == 8) {
            perturbedMat = NeighborListRandomization.randomize_half(mat, epsilon);
            noisyDegree = Tools.getDegree(perturbedMat);

            if (calibrated) {
                for (int i = 0; i < noisyDegree.length; i++)
                    noisyDegree[i] = NeighborListRandomization.calibrate_randomize(noisyDegree[i], noisyDegree.length, epsilon);
            }
            perturbed_coefficient = LCCEstimation.LCC_SixCases_list(perturbedMat, noisyDegree, epsilon, 1.0);
        }
        return perturbed_coefficient;
    }

    public static void main(String[] args) throws Exception {
        int dataset = 1;
        double fixPercentage = 0.3;
        double[] epsilon_all = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};

        new PerturbedLCC(dataset, fixPercentage, epsilon_all);
    }
}

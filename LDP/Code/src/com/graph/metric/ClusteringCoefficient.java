package com.graph.metric;

public class ClusteringCoefficient {
    public static double[] getLocalClusteringCoefficientList(boolean[][] mat) {
        int userNum = mat[0].length;
        double[] lcc = new double[userNum];
        double[] degree = Tools.getDegree(mat);
        double[] triangle = getTriangle(mat);
        for (int i = 0; i < userNum; i++) {
            lcc[i] = oneLocalCoefficient(triangle[i], degree[i]);
        }
        return lcc;
    }

    public static double oneLocalCoefficient(double triangle, double degree) {
        if (degree == 0 || degree == 1) return 0;
        return 2.0 * triangle / (degree * (degree - 1));
    }

    public static double[] getTriangle(boolean[][] mat) {
        int userNum = mat[0].length;
        double[] triangle = new double[userNum];
        for (int i = 0; i < userNum; i++) {
            if (i % 10000 == 0 && i != 0)
                System.out.print(i + " ");
            int[] neighbors = new int[userNum];
            int cnt = 0;
            for (int j = 0; j < userNum; j++) {
                if (mat[i][j])
                    neighbors[cnt++] = j;
            }
            for (int j = 0; j < cnt; j++) {
                for (int k = j + 1; k < cnt; k++)
                    if (mat[neighbors[j]][neighbors[k]])
                        triangle[i]++;
            }
        }
        return triangle;
    }

}

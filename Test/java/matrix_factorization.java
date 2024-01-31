import java.util.Random;

public class matrix_factorization {

    static double[] getColumn(int column , double[][] A){
        double[] result = new double[A.length];
        for(int i = 0 ; i < A.length; i++){
            result[i] = A[i][column];
        }
        return result;
    }

    static double[][] randMatrixGen(int n, int m){
        Random rand = new Random();

        double[][] matrix = new double[n][m];

        for(int i = 0 ; i< n ; i++){
            for(int j = 0 ; j< m ; j++){
                matrix[i][j] = rand.nextDouble();
            }
        }

        return matrix;
    }

    static void display(double[] A){
        for(int i = 0 ; i<A.length ; i++){
            System.out.print(A[i]+" ");
        }
    }

    static void display2d(double[][] A){
        for(int i = 0 ; i<A.length ; i++){
            for(int j = 0 ; j<A[0].length ; j++){
                System.out.print(A[i][j]+" ");
            }
            System.out.println();
        }
        System.out.println();
    }

    static double[][] dotProductMatrix(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;

        double[][] result = new double[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }


    static double dot(double[] A, double[] B){
        double sum = 0;
        for(int i = 0 ; i<A.length ; i++ ){
            sum += A[i]*B[i];
        }
        return sum;
    }
    
    //transposes a matrix
    static double[][] transpose(double[][] A){

        int column = A[0].length;
        int row = A.length;

        double[][] tranpose = new double[column][row];
        for(int i = 0 ; i < row ; i++){
            for(int j = 0 ; j < column ; j++){
                tranpose[j][i] = A[i][j];
            }
        }
        return tranpose;
    }

    public static PQ factorizeMatrix(double[][] R , double[][] P , double[][] Q, int K , int max_it , double alpha, double beta){
        //R is the rating matrix
        //P and Q are factorized matrices
        //K is the # features
        //steps are the iterations
        //alpha learning rate
        //beta regularization parameter
        Q = transpose(Q);
        int M = R.length;
        int N = R[0].length;

        while(max_it>0){

            for(int i = 0 ; i<R.length ; i++){
                for(int j = 0 ; j<R[i].length ; j++){

                    //for the non-zero values
                    if (R[i][j] > 0){

                        //calculate error of P and Q
                        double eij = R[i][j] - dot(P[i],getColumn(j,Q));

                        for(int k=0 ; k<K ; k++){
                            // Calculate gradient with alpha and beta parameters
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k]);
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j]);
                        }
                    }
                }
            }

            //Iteration Matrix
            double[][] eR = dotProductMatrix(P,Q);
            display2d(eR);
            double e = 0;

            //Calculating error
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    if (R[i][j] > 0) {
                        e = e + Math.pow(R[i][j] - dot(P[i], getColumn(j,Q)), 2);

                        for (int k = 0; k < K; k++) {
                            e = e + (beta / 2) * (Math.pow(P[i][k], 2) + Math.pow(Q[k][j], 2));
                        }
                    }
                }
            }

            // 0.001: local minimum, convergence check
            if (e < 0.001) {
                break;
            }
        

            max_it--;
            if(max_it == 0){
                System.out.println("Max Iterations reached");
            }
        }
        Q = transpose(Q);
        PQ result  = new PQ(P,Q);
        return result;
    }

    public static void main(String[] args){
        
        double[][] R = {
            {5,3,0,1},
            {4,0,0,1},
            {1,1,0,5},
            {1,0,0,4},
            {0,1,5,4},
            {2,1,3,0}
            };
        //N is number of users
        int N = R.length;
        //M is number of movies
        int M = R[0].length;
        //K is the number of features
        int K = 3;
        
        //user features matrix
        double[][] P = randMatrixGen(N,K);      
        //movie features matrix
        double[][] Q = randMatrixGen(M,K);

        
        int max_it = 5000;
        double alpha = 0.0002;
        double beta = 0.02;

        PQ PQfactorization = factorizeMatrix(R,P,Q,K,max_it,alpha,beta);

        double[][] Pnew = PQfactorization.getP();
        double[][] Qnew = PQfactorization.getQ();

        System.out.println("P: ");
        display2d(Pnew);
        System.out.println("Q: ");
        display2d(Qnew);

        double[][] Rc = dotProductMatrix(P,transpose(Qnew));
        System.out.println("Rc: ");
        display2d(Rc);


    }

}

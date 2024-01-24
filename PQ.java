public class PQ {

    private double[][] P;
    private double[][] Q;


    public PQ (double[][] P , double[][]Q){
        this.P = P;
        this.Q = Q;
    }

    double[][] getP(){
        return this.P;
    }

    double[][] getQ(){
        return this.Q;
    }

}
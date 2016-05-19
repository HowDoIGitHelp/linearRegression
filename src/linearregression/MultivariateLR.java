/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author DCS
 */
package linearregression;

import Jama.Matrix;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implements multivariate linear regression. 
 * @author DCS
 */
public class MultivariateLR {
    double alpha = 0.1;
    int numIterations = 400;
    
    /**
     *FEATURENORMALIZE Normalizes the features in X 
    *   FEATURENORMALIZE(X) returns a normalized version of X where
    *   the mean value of each feature is 0 and the standard deviation
    *   is 1. This is often a good preprocessing step to do when
    *   working with learning algorithms.
     * working with learning algorithms.
     * @param X the matrix to be normalized
     * @return the object the contains the matrix values of X, mu and sigma
     */
    FeatureNormalizationValues featureNormalize(Matrix X){
        //Write equivalent Java code for the Octave code below.
        // You need to set these values correctly. 
        //Octave: X_norm = X;
        
        //Octave: mu = zeros(1, size(X, 2));
        
        //Octave: sigma = zeros(1, size(X, 2));
        

        // ====================== YOUR CODE HERE ======================
        // Instructions: First, for each feature dimension, compute the mean
        //               of the feature and subtract it from the dataset,
        //               storing the mean value in mu. Next, compute the 
        //               standard deviation of each feature and divide
        //               each feature by it's standard deviation, storing
        //               the standard deviation in sigma. 
        //
        //               Note that X is a matrix where each column is a 
        //               feature and each row is an example. You need 
        //               to perform the normalization separately for 
        //               each feature. 
        //
        // Hint: You might find the 'mean' and 'std' functions useful.
        //       
        int n=X.getColumnDimension();
        int m=X.getRowDimension();
        FeatureNormalizationValues fNV = new FeatureNormalizationValues();
        fNV.mu=new Matrix(1,n,0);
        fNV.sigma=new Matrix(1,n,0);
        fNV.X=new Matrix(m,n);
        for(double[] example:X.getArray()){
            double[][] TDExample=new double[1][n];
            TDExample[0]=example;
            fNV.mu.plusEquals(new Matrix(TDExample));
            
        }
        fNV.mu.timesEquals(1.d/m);
        System.out.print("MEAN:");
        for(double d:fNV.mu.getArray()[0])
            System.out.print(d+",");
        System.out.println("");
        double sigma=0;
        for(double[] row: X.getArray()){
            for(int i=0;i<n;i++){
                fNV.sigma.set(0,i,fNV.sigma.get(0,i)+Math.pow(row[i]-fNV.mu.get(0, i),2));
            }
        }
        fNV.sigma.timesEquals(1.d/(m-1));
        for(int i=0;i<n;i++){
            fNV.sigma.set(0, i, Math.sqrt(fNV.sigma.get(0, i)));
        }        System.out.print("SIGMA:");
        for(double d:fNV.sigma.getArray()[0])
            System.out.print(d+",");
        System.out.println("");
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                fNV.X.set(i, j,(X.get(i, j)-fNV.mu.get(0, j))/fNV.sigma.get(0, j));
            }
        }
//        Matrix expandedFeatures=new Matrix(m,n+1,1);
//        for(int i=0;i<m;i++){
//            for(int j=1;j<=n;j++){
//                //System.out.print(i+","+j+":"+fNV.X.get(i, j-1));
//                expandedFeatures.set(i,j,fNV.X.get(i,j-1));
//            }
//            //System.out.println();
//        }
//        fNV.X=expandedFeatures;
        return fNV;
    }
    /**
     * GRADIENTDESCENTMULTI Performs gradient descent to learn theta
        theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
     * @param X
     * @param y
     * @param theta
     * @param alpha
     * @param numIterations
     * @return 
     */
    GradientDescentValues gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int numIterations){
        //Write equivalent Java code for the Octave code below.
        
        //Initialize some useful values.
        //Octave: m = length(y); % number of training examples

        //create a matrix that stores cost history
        //Octave: J_history = zeros(num_iters, 1);

        //Loop thru numIterations
        //Octave:for iter = 1:num_iters
        
        // ====================== YOUR CODE HERE ======================
        // Instructions: Perform a single gradient step on the parameter vector
        //               theta. 
        //
        // Hint: While debugging, it can be useful to print out the values
        //       of the cost function (computeCostMulti) and gradient here.
        //
        
        GradientDescentValues gDV = new GradientDescentValues();
        
        int m=X.getRowDimension();
        int n=X.getColumnDimension();
        gDV.theta=theta;
        gDV.costHistory=new Matrix(numIterations,n+1);
        for(int i=0;i<numIterations;i++){
            gDV.costHistory.set(i,0,computeCostMulti(X,y,gDV.theta).get(0, 0));
            for(int cHRow=0;cHRow<numIterations;cHRow++){
                for(int cHCol=1;cHCol<=n;cHCol++){
                    gDV.costHistory.set(cHRow,cHCol,gDV.theta.get(cHCol-1,0));
                }
            }
            Matrix newTheta=new Matrix(n,1);
//            for(int featureIndex=0;featureIndex<n;featureIndex++){
//                double[][] f=new double[m][1];
//                for(int exampleIndex=0;exampleIndex<m;exampleIndex++){
//                    f[exampleIndex][0]=X.get(exampleIndex, featureIndex);
//                }
//                Matrix featureValues = new Matrix(f);
////                System.out.println("X is a "+m+"x"+n+" matrix");
////                System.out.println("theta is a "+gDV.theta.getRowDimension()+"x"+gDV.theta.getColumnDimension()+" matrix");
//                Matrix errors=(solveHypothesis(X,gDV.theta).minus(y));
//                for(int k=0;k<m;k++){
//                    errors.set(k,0,errors.get(k,0)*featureValues.get(k, 0));
//                }
//                double t=alpha*(columnSum(errors,0)/m);
//                newTheta.set(featureIndex,0,gDV.theta.get(featureIndex, 0)-t);
//            }
            gDV.theta=gDV.theta.minus(X.transpose().times(X.times(gDV.theta).minus(y)).times(alpha/m));
           
            
//            int thetaCount=0;
//            for(double[] row:gDV.theta.getArray()){
//                System.out.print("theta"+thetaCount+++" "+row[0]+",");
//            }
            
            //System.out.println(i);
            
        }
        
        
        
        
        
        
        
        
        
        // Save the cost J in every iteration    
        //Octave: J_history(iter) = computeCostMulti(X, y, theta);
        return gDV;
    }
    /**
     *COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    *   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    *   parameter for linear regression to fit the data points in X and y
     * @param X
     * @param y
     * @param theta
     * @return 
     */
    Matrix computeCostMulti(Matrix X, Matrix y, Matrix theta){
        //Write equivalent Java code for the Octave code below.
        // Initialize some useful values
        //Octave: m = length(y); % number of training examples

        // You need to return the following variables correctly 
        //Octave: J = 0;

        // ====================== YOUR CODE HERE ======================
        // Instructions: Compute the cost of a particular choice of theta
        //                You should set J to the cost.

        //return ((X.arrayTimes(theta).minus(y)).transpose()).times(X.arrayTimes(theta).minus(y));
        int m=X.getRowDimension();
        int n=X.getColumnDimension();
//        Matrix errors = new Matrix(m,1);
//        errors=solveHypothesis(X,theta).minus(y);
//        for(int i=0;i<m;i++){
//            errors.set(i, 0,Math.pow(errors.get(i,0),2));
//        }
//        
        Matrix unsquared = X.times(theta).minus(y);
        return unsquared.transpose().times(1.d/(m+m)).times(unsquared);
        //return new Matrix(1,1,columnSum(errors,0)/(m+m));
        
        
    }
    /**
     NORMALEQN Computes the closed-form solution to linear regression 
       NORMALEQN(X,y) computes the closed-form solution to linear 
       regression using the normal equations.
     * @param X
     * @param y
     * @return 
     */
    
    Matrix normalEqn(Matrix X, Matrix y){
        //Write equivalent Java code for the Octave code below.

        //Octave: theta = zeros(size(X, 2), 1);

        // ====================== YOUR CODE HERE ======================
        //        Instructions: Complete the code to compute the closed form solution
        //               to linear regression and put the result in theta.

        return null;
    }
    private Matrix solveHypothesis(Matrix X,Matrix theta){
        return X.times(theta);
    }
    private double columnSum(Matrix m,int col){
        double sum=0;
        for(double[] row:m.getArray()){
            sum+=row[col];
        }
        return sum;
    }
    public Matrix thetaDenormalized(FeatureNormalizationValues X,FeatureNormalizationValues Y,Matrix theta){
        int m=X.X.getRowDimension();
        int n=X.X.getColumnDimension()+1;
        Matrix deTheta=new Matrix(n,1);
        deTheta.set(0,0,(theta.get(0,0)*Y.sigma.get(0, 0))/(Y.mu.get(0, 0)));
        for(int i=1;i<n;i++){
            deTheta.set(i,0,(theta.get(i,0)*X.mu.get(0,i-1)*Y.sigma.get(0, 0))/(X.sigma.get(0,i-1)*Y.mu.get(0, 0)));
        }
        return deTheta;
    }
    Matrix mapFeature(Matrix X1){
        int m=X1.getRowDimension();
        int n=X1.getColumnDimension();
        
        Matrix r=new Matrix(m,3,1);
        for(int curRow=0;curRow<m;curRow++){
            r.set(curRow,1,X1.get(curRow, 1));
            r.set(curRow,2,X1.get(curRow, 1)*X1.get(curRow, 1));
            //System.out.println("");
        }
        return r;
    }
}
class GradientDescentValues{
    Matrix theta;
    Matrix costHistory;

    public Matrix getTheta() {
        return theta;
    }

    public void setTheta(Matrix theta) {
        this.theta = theta;
    }

    public Matrix getCostHistory() {
        return costHistory;
    }

    public void setCostHistory(Matrix costHistory) {
        this.costHistory = costHistory;
    }
    
}
class FeatureNormalizationValues{
    Matrix X;
    Matrix mu;
    Matrix sigma;

    public Matrix getX() {
        return X;
    }

    public void setX(Matrix X) {
        this.X = X;
    }

    public Matrix getMu() {
        return mu;
    }

    public void setMu(Matrix mu) {
        this.mu = mu;
    }

    public Matrix getSigma() {
        return sigma;
    }

    public void setSigma(Matrix sigma) {
        this.sigma = sigma;
    }
    
}

class TestMLR{
    public static void main(String[] args) {
        try {
            MultivariateLR MLR=new MultivariateLR();
            //Write the corresponding Java code for the Octave code below between /**...*/
            
            long start = System.currentTimeMillis();
            
            System.out.println("Loading data...");
            //code for loading data
            /**
             * data = load('ex1data2.txt');
             * X = data(:, 1:2);
             * y = data(:, 3);
             * m = length(y);
             * %Print out some data points
             * fprintf('First 10 examples from the dataset: \n');
             * fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
             */
            //JAVA CODE HERE
            //==============
            ArrayList<String[]> examples=new ArrayList();
            Scanner sc=new Scanner(new File("noLOG.csv"));
            String[] features=new String[1];
            while(sc.hasNext()){
                features=sc.nextLine().split(",");
                examples.add(features);
            }
            double[][] rawMatrix = new double[examples.size()][features.length-1];
            double[][] rawYValues = new double[examples.size()][1];
            int rowCount=0;
            System.out.println(examples.size()+" training values with "+(features.length-1)+" features loaded.");
            for(String[] featureList:examples){
                //int colCount=0;
                for(int colCount=0;colCount<features.length-1;colCount++){
                    //System.out.print(feature+":"+colCount+",");
                    rawMatrix[rowCount][colCount]=Double.valueOf(featureList[colCount]);
                }
                rawYValues[rowCount][0]=Double.valueOf(featureList[features.length-1]);
                rowCount++;
                
            }
            Matrix X=new Matrix(rawMatrix);
            Matrix Y=new Matrix(rawYValues);
            //==============
            
            System.out.println("Normalizing features..");
            //code for normalizing data
            /**
             * [X mu sigma] = featureNormalize(X);
             * 
             * % Add intercept term to X
             * X = [ones(m, 1) X];
             */
            //JAVA CODE HERE
            //==============
            FeatureNormalizationValues  fNV=new FeatureNormalizationValues();
            fNV=MLR.featureNormalize(X);
//            for(double [] example:fNV.X.getArray()){
//                for(double feature:example){
//                    System.out.print(feature+",");
//                }
//                System.out.println("");
//            }
            int o=X.getRowDimension();
            int p=X.getColumnDimension();
            Matrix expandedFeatures=new Matrix(o,p+1,1);
            for(int i=0;i<o;i++){
                for(int j=1;j<=p;j++){
                    //System.out.print(i+","+j+":"+fNV.X.get(i, j-1));
                    expandedFeatures.set(i,j,fNV.X.get(i,j-1));
                }
                //System.out.println();
            }
            fNV.X=expandedFeatures;
            fNV.X=MLR.mapFeature(fNV.X);
             for(int i=0;i<o;i++){
                for(int j=0;j<3;j++){
                    System.out.print(fNV.X.get(i, j)+" ");
                }
                 System.out.println("");
             }
            o=X.getRowDimension();
            p=X.getColumnDimension();
            //==============
            FeatureNormalizationValues  fNVY=MLR.featureNormalize(Y);
            System.out.println("Running gradient descent...");
            //code for performing gradientDescent
            /**
             * % Choose some alpha value
             * alpha = 0.01;
             * num_iters = 400;
             * 
             * % Init Theta and Run Gradient Descent
             * theta = zeros(3, 1);
             * [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
             * 
             * % Plot the convergence graph
             * figure;
             * plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
             * xlabel('Number of iterations');
             * ylabel('Cost J');
             * 
             * % Display gradient descent's result
             * fprintf('Theta computed from gradient descent: \n');
             * fprintf(' %f \n', theta);
             */
            
            //JAVA CODE HERE
            //==============

            Matrix startTheta= new Matrix(fNV.X.getColumnDimension(),1);
            
            
            GradientDescentValues gDV=MLR.gradientDescent(fNV.X, Y, startTheta, MLR.alpha, MLR.numIterations);
//            for(double[] row:gDV.getCostHistory().getArray()){
//                System.out.print("COST:"+row[0]+"|");
//                for(int u=1;u<gDV.getCostHistory().getArray()[0].length;u++)
//                    System.out.print("THETA"+u+":"+row[u]+"|");
//                System.out.println("");
//            }
            double[] xData = new double[MLR.numIterations];
            for(int dataCount=0;dataCount<MLR.numIterations;dataCount++){
                xData[dataCount]=dataCount;
            }
            int aa=0;
            for(double[] row:gDV.theta.getArray()){
                System.out.println("theta "+aa+":"+row[0]);
                aa++;
            }
            double[] yData = gDV.getCostHistory().transpose().getArray()[0];
            Chart chart = QuickChart.getChart("Convergence of gradient descent with learning rate "+MLR.alpha, "Iterations", "Cost", "Cost", xData, yData);
            new SwingWrapper(chart).displayChart();
            
            
            
            
            
            
            
            
            //==============
            System.out.println("Estimating price...");
            //startTheta=MLR.thetaDenormalized(fNV, fNVY, gDV.theta);
            double[][] rawXVal=new double[3][1];
            rawXVal[0][0]=1;
            rawXVal[1][0]=(30233.7-fNV.mu.get(0, 0))/fNV.sigma.get(0, 0);
            //System.out.println(fNV.mu.get(0, 0));
            rawXVal[2][0]=(rawXVal[1][0]*rawXVal[1][0]);     
            //System.out.println(fNV.mu.get(0, 1));
            Matrix example=new Matrix(rawXVal);
            
            System.out.println("PREDICTION USING GRADIENT DESCENT:"+gDV.theta.transpose().times(example).get(0,0));
            /*
            % Estimate the price of a 1650 sq-ft, 3 br house
            % ====================== YOUR CODE HERE ======================
            % Recall that the first column of X is all-ones. Thus, it does
            % not need to be normalized.
            price = 0; % You should change this
            
            
            % ============================================================
            
            fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
            '(using gradient descent):\n $%f\n'], price);
            */
            //JAVA CODE HERE
            //==============
            
            
            
            
            //==============
            
            System.out.println("Solving with normal equations");
            /**
             * %% Load Data
             * data = csvread('ex1data2.txt');
             * X = data(:, 1:2);
             * y = data(:, 3);
             * m = length(y);
             * 
             * % Add intercept term to X
             * X = [ones(m, 1) X];
             * 
             * % Calculate the parameters from the normal equation
             * theta = normalEqn(X, y);
             * 
             * % Display normal equation's result
             * fprintf('Theta computed from the normal equations: \n');
             * fprintf(' %f \n', theta);
             * fprintf('\n');
             * 
             * 
             * % Estimate the price of a 1650 sq-ft, 3 br house
             * % ====================== YOUR CODE HERE ======================
             * price = 0; % You should change this
             * 
             * 
             * % ============================================================
             * 
             * fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
             * '(using normal equations):\n $%f\n'], price);
             */
            
            //JAVA CODE HERE
            //==============
            Matrix normalX=new Matrix(X.getRowDimension(),X.getColumnDimension()+1,1);
            for(int i=0;i<o;i++){
                for(int j=1;j<=p;j++){
                    //System.out.print(i+","+j+":"+fNV.X.get(i, j-1));
                    normalX.set(i,j,X.get(i,j-1));
                }
                //System.out.println();
            }
            X=normalX;
            
            example.set(1, 0, 7.480491299);
            example.set(2, 0, 55.95775008);
            Matrix normalTheta=(((X.transpose().times(X)).inverse()).times(X.transpose())).times(Y);
            for(double row: example.getArray()[0]){
                System.out.println(row);
            }
            //System.out.println("PREDICTION USING NORMAL EQUATION:"+normalTheta.transpose().times(example).get(0,0));
            
            
            
            
            
//            for (int i = 0; i < 1000000000; i++) {
//                double a = Math.sqrt((i+5.9)*(i*i));
//            }
            
            
            
            
            
            
            //==============
            long end = System.currentTimeMillis();
            
            long dif = end-start;
            if(dif>1000){
                dif = (end-start)/1000;
                System.out.println("Speed:"+dif+" seconds");
            }else{
                System.out.println("Speed:"+dif+" milliseconds");
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TestMLR.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
    }
    
}
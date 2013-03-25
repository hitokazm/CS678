package CS678.DBN;

import java.util.Arrays;
import java.util.Random;

import cs678.tools.Matrix;

public class RBM {

	Matrix inputs; // samples or original inputs
	double learningRate; // epsilon
	double[][] weights; // weights (dim of # hidden nodes x # input nodes)
	double[][] deltas; // delta weights for updates
	
	double[] b;  // input vector bias
	double[] c;  // hidden vector bias
	double[] h1; // h1 vector
	double[] h2; // h2 vector
	double[] x1; // x1 vector
	double[] x2; // x2 vector
	double[] Q1; // probability Q(h1|x1)
	double[] P;  // probability P(x2|h1)
	double[] Q2; // probability Q(h2|x2)
	
	int numHiddenNodes;
	int numVisibleNodes;
	int layerNum; 

	Random rand;
	
	public RBM(){
		this.rand = new Random(10000L);
		learningRate = 0.01;
	}
	
	public RBM(Matrix inputs){
		this();
		this.inputs = inputs;
	}
	
	public RBM(Matrix inputs, int numHiddenNodes, int numVisibleNodes){
		this(inputs);
		this.numHiddenNodes = numHiddenNodes;
		this.numVisibleNodes = numVisibleNodes;
	}
	
	public RBM(Matrix inputs, int numHiddenNodes, int numVisibleNodes, int layerNum){
		this(inputs, numHiddenNodes, numVisibleNodes);
		this.layerNum = layerNum;
		this.setInitialWeights();
		this.setVisibleBias();
		this.setHiddenBias();
		this.Q1 = new double[this.numHiddenNodes];
		this.P = new double[this.numVisibleNodes];
		this.Q2 = new double[this.numHiddenNodes];
	}
	
	
	public void setInitialWeights(){
		this.weights = new double[this.numHiddenNodes][this.numVisibleNodes];
		this.deltas = new double[this.numHiddenNodes][this.numVisibleNodes];
		for(int i = 0; i < this.weights.length; i++){
			this.setInitialWeights(this.weights[i]);
		}
	}
	
	public void setInitialWeights(double[] weights){
		for(int i = 0; i < weights.length; i++){
			double w;
			do{
				w = this.rand.nextGaussian();
			}while(w == 0.0 || w > 0.1 || w < -0.1);
			weights[i] = w;
		}
	}
	
	public void setVisibleBias(){
		this.b = new double[this.numVisibleNodes];
		this.setInitialWeights(this.b);
	}
	
	public void setHiddenBias(){
		this.c = new double[this.numHiddenNodes];
		this.setInitialWeights(this.c);
	}
	
	public double sigmoid(double bias, double[][] weights, int index, double[] vector, boolean row){
		if(row)
			return bias + this.dotProductOf(weights[index], vector); // for Q probability (row iteration)
		else{
			double sum = 0;
			for(int i = 0; i < weights.length; i++){
				sum += weights[i][index] * vector[i]; // for P probability (column iteration)
			}
			return bias + sum;
		}
	}
	
	
	public double dotProductOf(double[] vector1, double[] vector2){
		
		double sum = 0;
		for(int i = 0; i < vector1.length; i++){
			sum += vector1[i] * vector2[i];
		}
		return sum;
		
	}
	
	private double[] createInputFeatureVector(double[] row) {
		
		double[] inputFeautureVector = Arrays.copyOf(row, row.length);
		return inputFeautureVector;
		
	}
	
	public double sample(double probability){
		double p = this.rand.nextDouble();
		if(p < probability){
			return 1.0;
		}
		else{
			return 0.0;
		}
	}
	
	public double[] extractColumn(int col){
		double[] colArray = new double[this.weights.length];
		for(int i = 0; i < this.weights.length; i++){
			colArray[i] = this.weights[i][col];
		}
		return colArray;
	}
	
	public void update(){
		
		this.x1 = this.createInputFeatureVector(this.inputs.row(1));
		
		for(int i = 0; i < this.Q1.length; i++){
			this.Q1[i] = this.sigmoid(this.c[i], this.weights, i, this.x1, true);
			this.h1[i] = this.sample(this.Q1[i]);
		}
		
		for(int j = 0; j < this.P.length; j++){
			this.P[j] = this.sigmoid(this.b[j], weights, j, this.h1, false);
			this.x2[j] = this.sample(this.P[j]);
		}
		
		for(int i = 0; i < this.Q2.length; i++){
			this.Q2[i] = this.sigmoid(this.c[i], this.weights, i, this.x2, true);
		}
		
		this.updateWeights();
		
	}

	private void updateWeights() {
		
		for(int i = 0; i < this.numHiddenNodes; i++){
			for(int j = 0; j < this.numVisibleNodes; j++){
				this.deltas[i][j] = this.learningRate * (this.h1[i] * this.x1[j] - 
						this.Q2[i] * this.x2[j]);
			}
		}
		
		for(int i = 0; i < this.b.length; i++){
			this.b[i] += this.learningRate * (this.x1[i] - this.x2[i]);
		}
		
		for(int j = 0; j < this.c.length; j++){
			this.c[j] += this.learningRate * (this.h1[j] - this.Q2[j]);
		}
		
	}
	
}

package CS678.DBN;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Random;

import cs678.tools.Matrix;

public class RBM {
	
	static final boolean printout = false;
	static final boolean printout2 = true;
	
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
	
	double[][] bestWeights; // best weights
	
	int numHiddenNodes;
	int numVisibleNodes;
	int layerNum; 
	int maxSampleSize;
	int smallest;
	
	double stoppingCriteria;
	double threshold;
	
	Random rand;
	
	public RBM(){
		this.smallest = 1000000;
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
		this.h1 = new double[this.numHiddenNodes];
		this.h2 = new double[this.numHiddenNodes];
		this.x1 = new double[this.numVisibleNodes];
		this.x2 = new double[this.numVisibleNodes];
	}
	
	public RBM(Matrix inputs, int numHiddenNodes, int numVisibleNodes, int layerNum){
		this(inputs, numHiddenNodes, numVisibleNodes);
		this.layerNum = layerNum;
		this.setInitialWeights();
		this.setHiddenBias();
		this.Q1 = new double[this.numHiddenNodes];
		this.P = new double[this.numVisibleNodes];
		this.Q2 = new double[this.numHiddenNodes];
	}
	
	public RBM(Matrix inputs, int numHiddenNodes, int numVisibleNodes, int layerNum, 
			double stoppingCriteria){
		this(inputs, numHiddenNodes, numVisibleNodes, layerNum);
		this.stoppingCriteria = stoppingCriteria;
	}
	
	public RBM(Matrix inputs, int numHiddenNodes, int numVisibleNodes, int layerNum, 
			double stoppingCriteria, double threshold){
		this(inputs, numHiddenNodes, numVisibleNodes, layerNum, stoppingCriteria);
		this.threshold = threshold;
	}
	
	public RBM(Matrix inputs, int numHiddenNodes, int numVisibleNodes, int layerNum, 
			double stoppingCriteria, double threshold, int maxSampleSize){
		this(inputs, numHiddenNodes, numVisibleNodes, layerNum, stoppingCriteria, threshold);
		this.maxSampleSize = maxSampleSize;
	}
	
	public void setInitialWeights(){
		this.weights = new double[this.numHiddenNodes][this.numVisibleNodes];
		this.deltas = new double[this.numHiddenNodes][this.numVisibleNodes];
		
		this.bestWeights = new double[this.numHiddenNodes][this.numVisibleNodes];
		
		for(int i = 0; i < this.weights.length; i++){
			this.setInitialWeights(this.weights[i]);
			this.setInitialWeights(this.bestWeights[i]);
		}
		if(printout){
			for(int i = 0; i < this.weights.length; i++){
				System.out.printf("w%d: ", i);
				for(int j = 0; j < this.weights[i].length; j++){
					System.out.printf("%.3f ", this.weights[i][j]);
				}
				System.out.println();
			}
			System.out.println();
		}
	}
	
	public void setInitialWeights(double[] weights){
		for(int i = 0; i < weights.length; i++){
			double w;
			do{
				w = this.rand.nextGaussian();
			}while(w == 0.0 || w > 0.01 || w < -0.01);
			weights[i] = w;
		}
	}
	
	public void setVisibleBias(){
		this.b = new double[this.numVisibleNodes];
		
		double[] counts = new double[this.numVisibleNodes];
		
		for(int row = 0; row < this.inputs.rows(); row++){
			for(int col = 0; col < this.inputs.row(row).length - 1; col++){
				if(this.inputs.row(row)[col] > 0){
					counts[col] += 1.0;
				}
			}
		}
		
		double maximum = 0;
		for(int i = 0; i < counts.length; i++){
			if(maximum < counts[i]){
				maximum = counts[i];
			}
		}
		
		for(int i = 0; i < counts.length; i++){
			this.b[i] = counts[i] / maximum * 0.1;
		}
		
		this.setInitialWeights(this.b);
		if(printout){
			System.out.print("b: ");
			for(double w : this.b){
				System.out.printf("%.3f ", w);
			}
			System.out.println();
		}
	}
	
	public void setHiddenBias(){
		this.c = new double[this.numHiddenNodes];
		for(int i = 0; i < this.c.length; i++){
			this.c[i] = 0;
		}
		if(printout){
			System.out.print("c: ");
			for(double w : this.c){
				System.out.printf("%.3f ", w);
			}
			System.out.println("\n");
		}
	}
	
	public double sigmoid(double bias, double[][] weights, int index, double[] vector, boolean row) throws Exception{
		
		double net;
		if(row){
			net = bias + this.dotProductOf(weights[index], vector); // for Q probability (row iteration)
		}
		else{
			net = 0;
			for(int i = 0; i < weights.length; i++){
				net += weights[i][index] * vector[i]; // for P probability (column iteration)
			}
			net += bias;
		}
		return 1 / (1 + Math.exp(-1.0 * net));
	}
	
	
	public double dotProductOf (double[] vector1, double[] vector2) throws Exception{
		
		if(vector1.length != vector2.length)
			throw new Exception("Vector1: " + vector1.length + " Vector 2: " + vector2.length + "   Lengths are different.");
		
		double sum = 0;
		for(int i = 0; i < vector1.length; i++){
			sum += vector1[i] * vector2[i];
		}
		return sum;
		
	}
	
	private double[] createInputFeatureVector(double[] row) {
		
		double[] inputFeautureVector = Arrays.copyOf(row, row.length-1);
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
	
	public void CD1(boolean sampling, boolean useProbability, double[] features) throws Exception{

		for(int i = 0; i < this.Q1.length; i++){
			this.Q1[i] = this.sigmoid(this.c[i], this.weights, i, this.x1, true);
			this.h1[i] = this.sample(this.Q1[i]);
			if(printout)
				System.out.printf("Q1[%d]: %.3f   Sample: %.0f\n\n", i, this.Q1[i], h1[i]);
		}
		
		for(int j = 0; j < this.P.length; j++){
			this.P[j] = this.sigmoid(this.b[j], this.weights, j, this.h1, false);
			this.x2[j] = this.sample(this.P[j]);
			if(printout)
				System.out.printf("P[%d]: %.3f   Sample: %.0f\n\n", j, this.P[j], x2[j]);
		}
		
		for(int i = 0; i < this.Q2.length; i++){
			this.Q2[i] = this.sigmoid(this.c[i], this.weights, i, this.x2, true);
			if(sampling){
				if(useProbability){
					features[i] = this.Q2[i];//this.sample(this.Q2[i]);
				}
				else{
					features[i] = this.sample(this.Q2[i]);
				}
			}
			if(printout)
				System.out.printf("Q2[%d]: %.3f\n\n", i, this.Q2[i]);
		}

	}
	
	public void update() throws Exception{

		this.setVisibleBias();
		
		for(int row = 0; row < this.inputs.rows(); row++){
			this.x1 = this.createInputFeatureVector(this.inputs.row(row));
			this.CD1(false, false, null);
			
			if(this.updateWeights()){
				if(printout2){
					System.out.printf("No more updates needed. Done at row %d.\n\n", (row+1));
				}
				break;
			}
			else{
				if(printout)
					System.out.println("More updates needed. Continue.\n");
			}
		}
		this.weights = this.bestWeights;
	}

	public Matrix nextInputs() throws Exception{

		Matrix next = this.createNewDataset(this.inputs, false);
				
		return next;
	}
	
	public Matrix convertTestSet(Matrix testset) throws Exception{

		Matrix newTest = this.createNewDataset(testset, true);
		
		return newTest;
		
	}
	
	private Matrix createNewDataset (Matrix original, boolean testset) throws Exception{

		if(testset){
			double[] max = new double[this.inputs.cols()];
			double[] min = new double[this.inputs.cols()];
			for(int col = 0; col < max.length; col++){
				max[col] = this.inputs.columnMax(col);
				min[col] = this.inputs.columnMin(col);
			}
			
			original.normalize(min, max);			
		}
		
		Matrix dataset = new Matrix(0, this.Q2.length);
		dataset.setOutputClass("class", 9);

		String outFile = "";
		
		if(testset){
			System.out.println("Create New Testset.");
			int maxRow = original.rows();
			for(int row = 0; row < maxRow; row++){
				this.x1 = this.createInputFeatureVector(original.row(row));
				double[] features = new double[this.Q2.length+1];
				this.CD1(true, true, features);
				features[this.Q2.length] = original.get(row, this.inputs.cols()-1);
				dataset.addRow(features);
			}
			System.out.println("Export New Testset.");
			outFile = "testset" + this.layerNum + ".matrix";
		}
		else{
			System.out.println("Create Next Input.");
			int maxRow = maxSampleSize;
			int datumCount = 0;
			while(datumCount < maxRow){
				original.shuffle(this.rand);
				int row = this.rand.nextInt(original.rows());
				this.x1 = this.createInputFeatureVector(this.inputs.row(row));
				double[] features = new double[this.Q2.length+1];				
				this.CD1(true, true, features);
				features[this.Q2.length] = this.inputs.get(row, this.inputs.cols()-1);
				dataset.addRow(features);
				datumCount++;
			}
			System.out.println("Export Next Input.");			
			outFile = "rbm" + this.layerNum + ".matrix";
		}

		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("data/" + outFile));
		oos.writeObject(dataset);
		oos.close();
		System.out.println("Finished Exporting.");

		if(!testset)
			this.inputs = null;

		return dataset;
	}
	
	private boolean updateWeights() {
		
		boolean stop = false;
		int counter = 0;
		
		for(int i = 0; i < this.numHiddenNodes; i++){
			for(int j = 0; j < this.numVisibleNodes; j++){
				this.deltas[i][j] = this.learningRate * (this.h1[i] * this.x1[j] - 
						this.Q2[i] * this.x2[j]);
				this.weights[i][j] += this.deltas[i][j];
				if(Math.abs(this.deltas[i][j]) > this.stoppingCriteria){
					counter++;
					//if(stop){
					if(printout){
						if(Math.abs(this.deltas[i][j]) == .01 && printout2){
							System.out.printf("h1[%d]: %.2f  " +
									"x1[%d]: %.2f  Q2[%d]: %.2f  x2[%d]: %.2f\n", 
									i, h1[i], j, x1[j], i, Q2[i], j, x2[j]);
						}
						System.out.printf("dw[%d][%d]: %f > %f\n\n", i,j,
								Math.abs(this.deltas[i][j]), 
								this.stoppingCriteria);
					}
					//stop = false;
					//}
				}
				if(printout)
					System.out.printf("dw[%d][%d]: %.3f ", i, j, this.deltas[i][j]);
			}
			if(printout)
				System.out.println("\n");
		}
		
		if(printout)
			System.out.print("b: ");
		for(int i = 0; i < this.b.length; i++){
			this.b[i] += this.learningRate * (this.x1[i] - this.x2[i]);			
			if(printout)
				System.out.printf("%.3f ", b[i]);
		}
		if(printout)
			System.out.println("\n");
		
		if(printout)
			System.out.print("c: ");
		for(int j = 0; j < this.c.length; j++){
			this.c[j] += this.learningRate * (this.h1[j] - this.Q2[j]);
			if(printout)
				System.out.printf("%.3f ", c[j]);
		}
		if(printout)
			System.out.println("\n");
		
		double percent = 0;
		if(this.smallest > counter){
			this.smallest = counter;
			percent = (double) this.smallest / 
					(double) (this.numHiddenNodes * this.numVisibleNodes) * 100;
			for(int i = 0; i < this.weights.length; i++)
				this.bestWeights[i] = this.weights[i].clone();
			if(printout2){
				System.out.println("< Threshold Count: " + this.smallest + " (" + 
						percent +  "%)");
		}
			if(percent < this.threshold){
				stop = true;
			}
		}
		
		return stop;
		
	}
	
}

package CS678.PA;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Formatter;
import java.util.List;
import java.util.Random;

import cs678.tools.Matrix;
import CS678.MLP.SupervisedLearner;

public class PA extends SupervisedLearner {

	private double[][] weights;
	private Matrix traininingFeatures;
	private Matrix trainingLabels; 
	private Random rand; // random number generator

	private double C; // constant for PA calculation
	private int numLoop; // number of loop you want (should use validation?)
	
	private int numClasses; // number of classes 
	
	//private double trainAccuracy; // accuracy rate for training data
	//private double validationAccuracy; // accuracy rate for validation data
	
	public PA(){
		this.rand = new Random(1000000L);
	}
	
	public PA(int numLoop){
		this();
		this.numLoop = numLoop;
	}
	
	public PA(int numLoop, double C){
		this(numLoop);
		this.C = C;
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		double[] instance = new double[features.length+1];		
		for(int i = 0; i < features.length; i++){
			instance[i] = features[i];
		}
		instance[features.length] = 1.0;
		
		double[] yh = new double[this.numClasses]; // instantiate y^hat_t (for prediction)
		for(int t = 0; t < this.numClasses; t++){
			yh[t] = dot(this.weights[t], instance); //get dot products without argmax (i.e., not getting the prediction label) in Figure 2 in the paper
		}

		double argmax = rand.nextInt(this.numClasses);
		double yt = Double.NEGATIVE_INFINITY;
		
		for(int t = 0; t < this.numClasses; t++)
			if(yt < yh[t]){
				yt = yh[t]; // get argmax_s not in Y of yh (see line 5 in Figure 2 on page 571)
				argmax = (double) t;
			}
		labels[0] = argmax;
	}

	public double PA0(double loss, double xn){
		return loss / xn;
	}
	
	public double PA1(double C, double loss, double xn){
		return Math.min(C, loss/xn);
	}
	
	public double PA2(double C, double loss, double xn){
		return loss/(xn+1.0/(2.0*C));
	}

	public void exportCSV(Formatter formatter, int i){
		String fileName = "results" + (i+1) + ".csv";
		
		try{
			File file = new File("results/" + fileName);
			FileWriter filewriter = new FileWriter(file);
			BufferedWriter bw = new BufferedWriter(filewriter);
			PrintWriter pw = new PrintWriter(bw);
			pw.write(formatter.toString());
			pw.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {		
		Formatter formatter = new Formatter(new StringBuilder());
		
		// get number of classes for this dataset
		this.numClasses = labels.valueCount(0);
		
		//set initial weights to zeros
		this.setWeights(this.numClasses, features);
		
		// copy of weights for sse
		double[][] wd = new double[this.weights.length][this.weights[0].length];
		double sse = 0;
		
		Matrix flatLabels = this.customizeLabels(labels, 0, labels.rows()); // contain -1s and 1s horizontally

		List<Integer> rows = new ArrayList<Integer>();
		for(int row = 0; row < features.rows(); row++)
			rows.add(row);
		
		int rowCount=0;
		
		for(int iteration = 0; iteration < this.numLoop; iteration++){
			
			Collections.shuffle(rows);
			for(Integer row : rows){
				rowCount++;
//			for(int row = 0; row < features.rows(); row++){
				double[] instance = new double[features.row(row).length+1];
				for(int col = 0; col < features.row(row).length; col++){
					instance[col] = features.row(row)[col];
				}
				instance[instance.length-1] = 1.0; // bias input
				
//				System.out.print("instance: ");
//				for(double feature : instance){
//					System.out.printf("%.2f ", feature);
//				}
//				System.out.println();
				
				double[] yh = new double[flatLabels.cols()]; // instantiate y^hat_t (for prediction)
				for(int i = 0; i < yh.length; i++)
					yh[i] = 0.0; // initialize y_hat to zeros
				
				int yr_index = -1; // correct label variable
				for(int t = 0; t < this.numClasses; t++){
//					System.out.print("weights: ");
//					for(double w : this.weights[t]){
//						System.out.printf("%.2f ", w);
//					}
//					System.out.println();
					yh[t] = dot(this.weights[t], instance); //get dot products without argmax (i.e., not getting the prediction label) in Figure 2 in the paper
					if(flatLabels.get(row, t) == 1.0)
						yr_index = t; // get index of the correct label in the current label
				}
//				System.out.println("correct class: " + yr_index);
//				System.out.print("yh: ");
//				for(double y : yh){
//					System.out.printf("%.2f ", y);
//				}
//				System.out.println();
				
				if(yr_index == -1){ // no correct label is found (when does this happen??)
					double yt = Double.NEGATIVE_INFINITY;
					for(int t = 0; t < flatLabels.cols(); t++)
						if(flatLabels.row(row)[t] < 1)
							if(yt < yh[t])
								yt = yh[t]; // get argmax_s not in Y of yh (see line 5 in Figure 2 on page 571)
					double L = -1.0 * yt; // 0 (yr) - yt
					if(L < 1){
						double loss = 1.0 - L; // loss value on (40) on page 571
						double xn = this.dot(instance, instance); // ||x_t||^2 on page 571 (last line)
						double tau = this.PA2(this.C, loss, xn);
						for(int t = 0; t < flatLabels.cols(); t++){
							if(yt == yh[t])
								for(int st = 0; st < this.weights[t].length; st++){
									this.weights[t][st] = this.weights[t][st] - tau * instance[st]; // w^st_t+1 = w^st_t+1 - tau_t x_t (40) on pg 571
								}
						}
					}
					break;
				}
				
				double yr = yh[yr_index]; // output at the correct position
				double yt = Double.NEGATIVE_INFINITY;
				int yt_index = -1;
				for(int t = 0; t < flatLabels.cols(); t++){
					if(t != yr_index){
						if(yt < yh[t]){
							yt = yh[t]; // maximum wrong output
							yt_index = t;
						}
					}
				}
//				System.out.println("yr_index: " + yr_index + " yt_index: " + yt_index);
//				System.out.println("yr: " + yr + " yt: " + yt);

				double L = yr - yt; // compute w^r_t_t dot x - w^s_t_t dot x
//				System.out.println("L: " + L);
				
//				System.out.println("before update weights: ");
//				for(int i = 0; i < this.weights.length; i++){
//					for(int j = 0; j < this.weights[i].length; j++)
//						System.out.printf("%.2f ", this.weights[i][j]);
//					System.out.println();
//				}
				if(L < 1.0){
					double loss = 1.0 - L; // see (41)
//					System.out.println("loss: " + loss);
					double xn = 2.0 * this.dot(instance, instance); // 2*||x_t||^2 on page 571 (last line)
//					System.out.println("xn: " + xn);
					double tau = this.PA2(this.C, loss, xn); // PA2
//					System.out.println("tau: " + tau);
					for(int k = 0; k < flatLabels.cols(); k++){
						if(yt_index == k){
							for(int st = 0; st < this.weights[k].length; st++){
								this.weights[k][st] = this.weights[k][st] - tau * instance[st]; // w^st_t+1 = w^st_t+1 - tau_t x_t (40) on pg 571 
							}
						}
						else if(yr_index == k){
							for(int rt = 0; rt < this.weights[k].length; rt++){
								this.weights[k][rt] = this.weights[k][rt] + tau * instance[rt]; // w^rt_t+1 = w^rt_t+1 + tau_t x_t (40) on pg 571
							}
						}
					}
				}
//				System.out.println("after update weights: ");
//				for(int i = 0; i < this.weights.length; i++){
//					for(int j = 0; j < this.weights[i].length; j++)
//						System.out.printf("%.2f ", this.weights[i][j]);
//					System.out.println();
//				}
				double acc = super.measureAccuracy(features, labels, null);
				System.out.println("instance: " + (row+1) + " Accuracy: " + acc);
				formatter.format("%d,%f\n", rowCount, acc);
			}
			
//			System.out.printf("Iteration %d\n", (iteration+1));
//			for(int i = 0; i < this.weights.length; i++){
//				for(int j = 0; j < this.weights[i].length; j++){
//					System.out.printf("%.2f ", this.weights[i][j]);
//				}
//				System.out.println();
//			}
//			if(iteration > 0){
//				sse = 0;
//				for(int i = 0; i < this.weights.length; i++)
//					for(int j = 0; j < this.weights[0].length; j++)
//						sse += Math.pow(this.weights[i][j] - wd[i][j], 2.0);
//				System.out.println("Iteration: " + (iteration+1) + " SSE: " + sse);
//				for(int i = 0; i < this.weights.length; i++)
//					wd[i] = Arrays.copyOf(this.weights[i], this.weights[i].length);
//			}
		}
		this.exportCSV(formatter, 1);
	}
	
//	private void setGramMatrix(Matrix features) {
//		int rows = features.rows();
//		this.gramMatrix = new double[rows][rows];
//		for(int row1 = 0; row1 < rows; row1++){
//			for(int row2 = 0; row2 < rows; row2++){
//				this.gramMatrix[row1][row2] = this.dot(features.row(row1), features.row(row2)) + 1;
//			}
//		}
//	}

	private double dot(Matrix features, int t){
		return 0.0;
	}
	
	private double dot(double[] weights, double[] instance) {
		double sum = 0.0;
		
		for(int i = 0; i < weights.length; i++)
			sum += weights[i] * instance[i];
		
		return sum;
	}

	private void setWeights(int numClasses, Matrix features) {

		this.weights = new double[numClasses][features.cols()+1]; // row: # classes, col: # features + 1 (bias)
		
		for(int row = 0; row < this.weights.length; row++)
			for(int col = 0; col < this.weights[0].length; col++){
				this.weights[row][col] = 0.0;
			}
		}
	
	private Matrix customizeLabels(Matrix labels, int rowStart, int trainSize) {
		
		int numClass = labels.valueCount(0); 
		Matrix fixedLabels;
		
		if(numClass == 0){ // if # class is just one (continuous) or binary
			fixedLabels = new Matrix(labels, rowStart, 0, trainSize, labels.cols()); // just copy the labels as it is
		}
		else{ // if there are more than 2 classes
			fixedLabels = new Matrix();
			fixedLabels.setSize(trainSize, labels.valueCount(0));
			for(int row = 0; row < trainSize; row++){
				for(int col = 0; col < fixedLabels.cols(); col++){
					//System.out.println("Original Output: " + labels.row(row)[0]);
					if((int) labels.row(row)[0] == col)
						fixedLabels.set(row, col, 1.0);
					else
						fixedLabels.set(row, col, -1.0);
				}
			}
		}
		//labels.print();
		//fixedLabels.print();
		// TODO does this really contain 1s and 0s?? check this!
		return fixedLabels;
	}

	private void shuffleData(Matrix features, Matrix labels) {
		this.rand = new Random(1L); // set the random seed to 1L
		features.shuffle(rand);
		this.rand = new Random(1L); // set the random seed to 1L
		labels.shuffle(rand);		
	}

}

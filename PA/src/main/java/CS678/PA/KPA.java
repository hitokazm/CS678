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
import edu.byu.nlp.util.CounterMap;
import edu.byu.nlp.util.Pair;
import CS678.MLP.SupervisedLearner;

public class KPA extends SupervisedLearner {

	private CounterMap<Integer, Integer> kernelMatrix;
	private List<List<Pair<Double, Integer>>> weights;
	private Matrix traininingFeatures;
	private Matrix trainingLabels;
	private Random rand; // random number generator

	final static int B = 10000;
	private int[] lowestRows; 
	private List<List<Double>> kernels; 
	
	final static double power = 5.0; // power for polynomial kernel
	//final static double constant = 0.45; // consntant for polynomial kernel
	
	private double constant;
	private double C; // constant for PA calculation
	private int numLoop; // number of loop you want (should use validation?)
	
	private int numClasses; // number of classes 
	
	public KPA(){
		this.kernels = new ArrayList<List<Double>>();
		this.rand = new Random(1000000L);
		this.constant = 0.5;
	}
	
	public KPA(int numLoop){
		this();
		this.numLoop = numLoop;
	}
	
	public KPA(int numLoop, double C){
		this(numLoop);
		this.C = C;
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
//		double[] instance = new double[features.length+1];		
//		for(int i = 0; i < features.length; i++){
//			instance[i] = features[i];
//		}
//		instance[features.length] = 1.0;
		
		double[] yh = new double[this.numClasses]; // instantiate y^hat_t (for prediction)
		for(int cls = 0; cls < this.numClasses; cls++){
			yh[cls] = this.getOutput(features, cls);//dot(this.weights[t], instance); //get dot products without argmax (i.e., not getting the prediction label) in Figure 2 in the paper
		}

		double argmax = rand.nextInt(this.numClasses);
		double yt = Double.NEGATIVE_INFINITY;
		
		for(int t = 0; t < this.numClasses; t++){
			if(yt < yh[t]){
				yt = yh[t]; // get argmax_s not in Y of yh (see line 5 in Figure 2 on page 571)
				argmax = (double) t;
			}
		}
		
		labels[0] = argmax;
	}

	private double getOutput(double[] features, int cls){
		
		double sum = 0.0;
		for(Pair<Double, Integer> pair : this.weights.get(cls)){
			double alpha = pair.getFirst();
			int row = pair.getSecond();
			sum += alpha * this.polynomialKernel(this.traininingFeatures.row(row), 
					features, power, constant);
		}
		
		return sum;
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

	public double PA0(double loss, double xn){
		return loss / xn;
	}
	
	public double PA1(double C, double loss, double xn){
		return Math.min(C, loss/xn);
	}
	
	public double PA2(double C, double loss, double xn){
		return loss/(xn+1.0/(2.0*C));
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// save original features
		
		Formatter formatter = new Formatter(new StringBuilder());
		
		this.traininingFeatures = features;
		this.trainingLabels = labels;
		
		// get number of classes for this dataset
		this.numClasses = labels.valueCount(0);
		
		this.lowestRows = new int[this.numClasses];
		
		// set array of weights (update history) for kernelized PA
		this.weights = new ArrayList<List<Pair<Double,Integer>>>(this.numClasses); // y*tau -> row
		for(int i = 0; i < this.numClasses; i++){
			this.weights.add(new ArrayList<Pair<Double, Integer>>());
			this.kernels.add(new ArrayList<Double>());
			//System.out.println("weight " + i + "'s size: " + this.weights.get(this.weights.size()-1).size());
		}
		
		// set kernel matrix -> too memory-intensive!!
		//this.setKernelMatrix(features);
		this.kernelMatrix = new CounterMap<Integer, Integer>();
		
		//set initial weights to zeros
		//this.setWeights(this.numClasses, features);
		
		Matrix flatLabels = this.customizeLabels(labels, 0, labels.rows()); // contain -1s and 1s horizontally
		
//		for(int row = 0; row < labels.rows(); row++){
//			System.out.print("original: " + labels.get(row, 0) + " converted: ");
//			for(int col = 0; col < flatLabels.cols(); col++)
//				System.out.print(flatLabels.get(row, col) + " ");
//			System.out.println();
//		}
		
		
		List<Integer> rows = new ArrayList<Integer>();
		for(int row = 0; row < features.rows(); row++)
			rows.add(row);
		
		for(int iteration = 0; iteration < this.numLoop; iteration++){
			//for(int row = 0; row < features.rows(); row++){
			Collections.shuffle(rows);
			int rowCount = 0;
			for(Integer row : rows){
				rowCount++;
				double[] instance = features.row(row); //new double[features.row(row).length+1]; 
//				for(int col = 0; col < features.row(row).length; col++){
//					instance[col] = features.row(row)[col];
//				}
//				instance[instance.length-1] = 1.0; // bias input

//				System.out.print("instance: ");
//				for(double feature : instance){
//					System.out.printf("%.2f ", feature);
//				}
//				System.out.println();
				
				double[] yh = new double[flatLabels.cols()]; // instantiate y^hat_t (for prediction)
				
				int yr_index = -1; // correct label variable
				for(int cls = 0; cls < this.numClasses; cls++){
					yh[cls] = this.getKernelizedDot(instance, row, cls); //dot(this.weights[t], instance); //get dot products without argmax (i.e., not getting the prediction label) in Figure 2 in the paper
					if(flatLabels.get(row, cls) == 1.0)
						yr_index = cls; // get index of the correct label in the current label
				}
				
//				System.out.println("correct class: " + yr_index);
//				System.out.print("yh: ");
//				for(double y : yh){
//					System.out.printf("%.2f ", y);
//				}
//				System.out.println();

				if(yr_index == -1){ // no correct label is found (when does this happen??)
					double yt = Double.NEGATIVE_INFINITY;
					int yt_index = -1;
					for(int t = 0; t < flatLabels.cols(); t++){
//						System.out.println("SVs: ");
//						System.out.println(this.weights.toString());
						if(flatLabels.row(row)[t] < 1)
							if(yt < yh[t]){
								yt = yh[t]; // get argmax_s not in Y of yh (see line 5 in Figure 2 on page 571)
								yt_index = t;
							}
					}
					
					double L = -1.0 * yt; // 0 (yr) - yt
					if(L < 1){
						double loss = 1.0 - L; // loss value on (40) on page 571
						double xn = this.dot(instance, instance); // ||x_t||^2 on page 571 (last line)
						double tau = this.PA2(this.C, loss, xn);
						this.weights.get(yt_index).add(new Pair<Double, Integer>(-1.0 * tau, row));
//						for(int cls = 0; cls < flatLabels.cols(); cls++){
//							if(yt == yh[cls]){
//								this.weights.get(cls)
//								for(int st = 0; st < this.weights[t].length; st++){
//									this.weights[t][st] = this.weights[t][st] - tau * instance[st]; // w^st_t+1 = w^st_t+1 - tau_t x_t (40) on pg 571
//								}
//							}
//						}
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
//				System.out.println(this.weights.toString());

				if(L < 1.0){
					double loss = 1.0 - L; // see (41)
//					System.out.println("loss: " + loss);
					double xn = 2.0 * this.polynomialKernel(instance, instance, power, constant); //(this.dot(instance, instance) + 1); // 2*||x_t||^2 on page 571 (last line)
//					System.out.println("xn: " + xn);
					double tau = this.PA2(this.C, loss, xn); // PA2
//					System.out.println("tau: " + tau);
					if(this.weights.get(yt_index).size() + 1 > B){
						this.weights.get(yt_index).remove(lowestRows[yt_index]);
					}
					if(this.weights.get(yr_index).size() + 1 > B){
						this.weights.get(yr_index).remove(lowestRows[yr_index]);
					}					
					this.weights.get(yt_index).add(new Pair<Double, Integer>(-1.0 * tau, row));
					this.weights.get(yr_index).add(new Pair<Double, Integer>(1.0 * tau, row));

//					for(int k = 0; k < flatLabels.cols(); k++){
//						if(yt_index == k){
//							for(int st = 0; st < this.weights[k].length; st++){
//								this.weights[k][st] = this.weights[k][st] - tau * instance[st]; // w^st_t+1 = w^st_t+1 - tau_t x_t (40) on pg 571 
//							}
//						}
//						else if(yr_index == k){
//							for(int rt = 0; rt < this.weights[k].length; rt++){
//								this.weights[k][rt] = this.weights[k][rt] + tau * instance[rt]; // w^rt_t+1 = w^rt_t+1 + tau_t x_t (40) on pg 571
//							}
//						}
//					}
					int clsCount = 0;
					for(List<Pair<Double, Integer>> weight : this.weights){
						System.out.println("Class: " + clsCount + "  SV Count: " + weight.size());
						clsCount++;
					}
				}
				System.out.println("Insntance " + rowCount);
//				System.out.println();
//				double acc = super.measureAccuracy(super.getTestFeatures(), super.getTestLabels(), null);
//				System.out.println("T=" + rowCount + "  Test set accuracy: " + acc);
//				System.out.println();
//				formatter.format("%d", rowCount);
//				formatter.format(",");
//				formatter.format("%f\n", acc);
//				System.out.println("after update weights: ");
//				System.out.println(this.weights.toString());
			}
			
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
		//this.exportCSV(formatter, (int) power);
	}
	
	private double getKernelizedDot(double[] instance, int row, int cls) {

		
		double sum = 0.0;
		
		if(this.weights.get(cls).size() == 0)
			return sum;
		
		if(this.kernels.get(cls).size() + 1 > B){
			//this.kernels.get(cls).clear();

			double min = Double.POSITIVE_INFINITY;
			int sv = -1;
			int rowPosition = 0;

			for(Pair<Double, Integer> pair : this.weights.get(cls)){
				double tau_y = pair.getFirst(); // tau_i * y_i (-1.0 or 1.0)
				int row2 = pair.getSecond(); // rows contained in the update history
				
				double dw = tau_y * this.polynomialKernel(this.traininingFeatures.row(row), 
						this.traininingFeatures.row(row2), power, constant);
				
				if(Math.abs(dw) < min){
					min = Math.abs(dw);
					sv = rowPosition; // get the row position in the weight sv vector
				}
				rowPosition++;
				
				sum += dw;
			}
			
			this.lowestRows[cls] = sv;
			
			return sum;
		}
		
		for(Pair<Double, Integer> pair : this.weights.get(cls)){
			double tau_y = pair.getFirst(); // tau_i * y_i (-1.0 or 1.0)
			int row2 = pair.getSecond(); // rows contained in the update history
			
			sum += tau_y * this.polynomialKernel(this.traininingFeatures.row(row), 
					this.traininingFeatures.row(row2), power, constant);
			
//			if(row <= row2){
//				if(!this.kernelMatrix.containsKey(row)){
//					this.setKernelMatrix(this.traininingFeatures, row, row2, power, constant);					
//				}
//				else if (!this.kernelMatrix.getCounter(row).containsKey(row2)){
//					this.setKernelMatrix(this.traininingFeatures, row, row2, power, constant);										
//				}
//				sum += tau_y * this.kernelMatrix.getCount(row, row2);
//			}
//			else{
//				if(!this.kernelMatrix.containsKey(row2)){
//					this.setKernelMatrix(this.traininingFeatures, row2, row, power, constant);					
//				}
//				else if (!this.kernelMatrix.getCounter(row2).containsKey(row)){
//					this.setKernelMatrix(this.traininingFeatures, row2, row, power, constant);										
//				}
//				sum += tau_y * this.kernelMatrix.getCount(row2, row);
//			}
		}
		
//		System.out.println("matrix size: " + this.kernelMatrix.totalSize());
		
		return sum;
	}

	private void setKernelMatrix(Matrix features, int row1, int row2, double power, double constant){
		this.kernelMatrix.setCount(row1, row2, this.polynomialKernel(features.row(row1), 
				features.row(row2), power, constant));
	}
	
	private void setKernelMatrix(Matrix features) {
		int rows = features.rows();
		//System.out.println("feature count: " + rows);
		this.kernelMatrix = new CounterMap<Integer, Integer>();
		for(int row1 = 0; row1 < rows; row1++){
			for(int row2 = 0; row2 < rows; row2++){
				if(row1 <= row2){
					//System.out.println("pair: " + row1 + ", " + row2);
					this.kernelMatrix.setCount(row1, row2, this.polynomialKernel(features.row(row1), 
							features.row(row2), power, constant));
					//this.setKernelMatrix(features, row1, row2, power, constant);
				}
			}
		}
		//System.out.println(this.kernelMatrix.toString());
		//System.out.println("Total Pair: " + this.kernelMatrix.totalSize());
	}

	private double polynomialKernel(double[] feature1, double[] feature2, double power, double c){
//		System.out.print("feature1: [");
//		for(double feature : feature1)
//			System.out.printf("%.2f ", feature);
//		System.out.println("]");
//		System.out.print("feature2: [");
//		for(double feature : feature2)
//			System.out.printf("%.2f ", feature);
//		System.out.println("]");
//		System.out.println("power: " + power);
//		System.out.println("dot product: " + (c+this.dot(feature1, feature2)));
//		System.out.printf("Kernel: %.2f\n", Math.pow(c+this.dot(feature1, feature2), power));
		
//		System.out.print("feature1: [");
//		for(double f : feature1)
//			System.out.printf("%.2f ", f);
//		System.out.println("]");

//		double[] feature = Arrays.copyOf(feature1, feature1.length+1);
//		feature[feature1.length] = 1.0;
//		feature1 = feature;

//		System.out.print("new feature1: [");
//		for(double f : feature1)
//			System.out.printf("%.2f ", f);
//		System.out.println("]");
		
//		feature = Arrays.copyOf(feature2, feature2.length+1);
//		feature[feature2.length] = 1.0;
//		feature2 = feature;
		
//		for(int i = 0; i < feature1.length; i++){
//			System.out.printf("%.2f * %.2f ", feature1[i], feature2[i]);
//			if(i < feature1.length - 1)
//				System.out.print("+ ");
//			else
//				System.out.print("= ");
//		}
//		System.out.println(this.dot(feature1, feature2));
//		
//		System.out.println("(" + this.dot(feature1, feature2) + "+" 
//		+ c + ")" + "^" + power + "=" + Math.pow(c+this.dot(feature1, feature2), power));
		
		double kernel = Math.pow(c+this.dot(feature1, feature2), power);
		
//		System.out.println("Kernel: " + kernel);
		
		return kernel;
	}
	
	private double dot(double[] weights, double[] instance) {
		double sum = 0.0;
		
		for(int i = 0; i < weights.length; i++)
			sum += weights[i] * instance[i];
		
		return sum;
	}

//	private void setWeights(int numClasses, Matrix features) {
//
//		this.weights = new double[numClasses][features.cols()+1]; // row: # classes, col: # features + 1 (bias)
//
//		for(int row = 0; row < this.weights.length; row++)
//			for(int col = 0; col < this.weights[0].length; col++){
//				this.weights[row][col] = 0.0;
//			}
//	}
	
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

}

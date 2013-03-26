package cs678.bptt;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Formatter;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import cs678.tools.Matrix;
import cs678.tools.SupervisedLearner;
import edu.byu.nlp.trees.Tree;
import edu.byu.nlp.util.Pair;

public class BPTT extends SupervisedLearner {

	final static double trainingPercent = 0.9; // amount of training data
	final static long seed = 1L; // random number seed
	final static int maxInteration = 100; // # training iterations
	final static boolean useValidation = false; // option for validation data check
	
	
	private double learningRate; // learning rate (eta)
	private double momentum; // momentum (alpha)
	private double trainAccuracy; // training data accuracy
	private double validationAccuracy; // validation set accuracy
	private Random random; // random number generator

	private int numHiddenNodes; // # hidden nodes, don't need to be array because BPTT has a single layer
	private int numOutputNodes; // # output nodes
	
	private int k; // size of history
	
	private final static int accuracyTrial = 100; // count of trials for accuracy measure 
	
	private Pair<List<double[]>, double[]> currentlySelectedFeatures; // currently selected feature series; for accuracy computation
	
	private Layer[] layers; // hidden and output layers
	private Matrix trainFeatures; // training features
	private Matrix trainLabels; // training labels (multiple columns)
	private Matrix trainTestLabels; // training labels (single columns)
	private Matrix validationFeatures; // validation features
	private Matrix validationLabels; // validation labels (multiple columns)
	private Matrix validationTestLabels; // validation labels (single columns)
	
	private final static Logger logger = Main.logger;
	
	/**
	 * constructor
	 */
	public BPTT(){
		this.learningRate = 0.5; // default value
		this.momentum = 0.9; // default value 
		this.random = new Random();
		this.numHiddenNodes = 0;
		this.numOutputNodes = 0;
	}
	
	public BPTT(int k){
		this();
		this.k = k;
	}
	
	/**
	 * constructor
	 * @param learningRate: specified learning rate (eta)
	 */
	public BPTT(double learningRate){
		this();
		this.setLearningRate(learningRate);
	}
		
	/**
	 * set specified learning rate (eta)
	 * @param learningRate: learning rate (double)
	 */
	public void setLearningRate(double learningRate){
		this.learningRate = learningRate;
	}
	
	/**
	 * set # hidden nodes
	 * @param numHiddenNodes (int)
	 */
	public void setNumHiddenNodes(int numHiddenNodes){
		this.numHiddenNodes = numHiddenNodes;
	}
	
	/**
	 * set # hidden nodes based on the number of features in the instances (if it's not pre-specified).
	 * @param featureInstances: # attributes in the matrix * 2 + 1(bias) (int)
	 */
	private void setNumHiddenNodes(Matrix featureInstances){
		if (this.numHiddenNodes == 0)
			this.numHiddenNodes = featureInstances.cols() * 2 + 1;
	}

	private void setNumOutputNodes(Matrix labels) {
		/*
		 * 5 cases
		 * 1) output feature が一個でcontinuous -- one node
		 * 2) output feature が一個でclassがbinary -- one node
		 * 3) output feature が一個でclassがtertiary以上 -- two or more nodes according to the # classes
		 * 4) output feature が複数で全部continuous -- nodes with # cols in the label matrix
		 * 5) output feature が複数で全部がbinary -- nodes with # cols in the label matrix 
		 * 今回はケースが(3)のみ。なのでoutput nodesの形式は一通りでいい、と思う。
		 */
		if(this.numOutputNodes == 0){ // if # output nodes are not specified yet
			if(labels.cols() == 1){ // if # output col is 1 (in this lab)
				if(labels.valueCount(0) == 2) // if case (2)
					this.numOutputNodes = 1; 
				else if(labels.valueCount(0) > 2) // if case (3)
					this.numOutputNodes = labels.valueCount(0); // set # output nodes based on # outputs
				else // if the output value is continuous
					this.numOutputNodes = 1; // # output is 1
			}
			else{ // if # output cols are more than 1
				// ここは今回はkick inしないはず。とりあえずこのままで。
				this.numOutputNodes = labels.cols(); // set # output nodes to # cols
			}
		}		
	}
	
	
	/**
	 * set # output nodes
	 * @param numOutputNodes (int)
	 */
	private void setNumOutputNodes(int numOutputNodes) {
		this.numOutputNodes = numOutputNodes;
		
	}

	/**
	 * export Map data to CSV file.
	 * @param map sorted map structure
	 * @param fileName file name (String)
	 * @throws Exception
	 */
	private void exportCSV(Map<Integer, Double> map, String fileName) throws Exception{

		File file = new File("data/" + fileName);
		FileWriter filewriter = new FileWriter(file);
		BufferedWriter bw = new BufferedWriter(filewriter);
		PrintWriter pw = new PrintWriter(bw);
		
		for(Map.Entry<Integer, Double> entry : map.entrySet()){
			pw.printf("%d,%1.9f\n", entry.getKey(), entry.getValue()); // sample size & accuracy
		}
		pw.close();
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// for data collection
		Map<Integer, Double> accuracy = new TreeMap<Integer, Double>();
		Map<Integer, Double> kestimate = new TreeMap<Integer, Double>();
		double bestAcc = 0;
		int bestK = this.k;
		
		int noImprovement = 0;
		
		// pre-process data sets
		
		this.createDataset(features, labels);
		
		for(int j = 1; j < 100; j++){
			// create hidden layer
			int inputSize = features.cols()*3+1; // #features (actual input size) + # hidden neurons (2 * #features) + 1 (bias)
			this.numHiddenNodes = features.cols() * 2; // # features * 2
			Layer hiddenLayer = new HiddenLayer(this.numHiddenNodes, this.learningRate, random, this.momentum, this.k, inputSize);

			// create output layer
			inputSize = features.cols() * 2 + 1; // # hidden neurons (2 * #features) + 1 (bias)
			this.setNumOutputNodes(labels); // compute # output nodes
			Layer outputLayer = new OutputLayer(this.numOutputNodes, this.learningRate, this.random, this.momentum, inputSize);
			((OutputLayer) outputLayer).setNumHiddenNeurons(hiddenLayer.getNumNeurons());

			// put layers in an array (very redundant. fix this in the future)
			this.layers = new Layer[2];
			layers[0] = hiddenLayer;
			layers[1] = outputLayer;

			for(int i = 0; i < 100; i++){
				feedforward();
				backpropagation();
				//double acc = this.measureAccuracy(this.trainFeatures, this.trainTestLabels, this.trainLabels, null);
				//accuracy.put(i+1, acc);
				//if(i % 2 == 0){
				//	System.out.printf("Sample Size: %d    Accuracy: %.2f\n", i+1, acc);
			}
			double acc = this.averageAccuracy();
			if(bestAcc < acc){
				bestAcc = acc;
				bestK = this.k;
				kestimate.put(j, (double) bestK);
			}
			else{
				kestimate.put(j, (double) bestK);
				this.k++;
				//noImprovement++;
				//if(noImprovement > 100)
				//	break;
			}
			System.out.printf("iteration: " + j + " current best K: " + bestK + " current best acc: " + bestAcc);
			System.out.println();
			
		}
		String filename = "estimate-k.csv";
		this.exportCSV(kestimate, filename);
		//logger.info("Training Accuracy: " + this.measureAccuracy(this.trainFeatures, this.trainTestLabels, this.trainLabels, null));
	}

	private double averageAccuracy() throws Exception {
		double sum = 0;
		double acc;
		int max = 20;
		for(int i = 0; i < max; i++){
			acc = this.measureAccuracy(this.trainFeatures, this.trainTestLabels, this.trainLabels, null);
			sum += acc;
		}
		return ((double) sum / (double) max);
	}
	
	/**
	 * feed-forward process.
	 * @return a output value provided by the output layer.
	 */
	private double[] feedforward() throws Exception {
		// feed forward 1: create inputs with size k and a target 
		Pair<List<double[]>, double[]> selectedFeatures = this.getInputs(this.trainFeatures, this.trainLabels);
		
		// feed forward 2: set inputs to the hidden layer
		this.layers[0].setInput(selectedFeatures.getFirst());

		// feed forward 3: set target values
		this.layers[1].setTargets(selectedFeatures.getSecond());
		
		// feed forward 4: set the final output from the hidden layer as the input to the output layer
		this.layers[1].setInput(Arrays.asList(layers[0].getOutput()));
		
		// get output value vector
		return this.layers[1].getOutput();
				
	}
	
	/**
	 * backpropagation process.
	 */
	private void backpropagation(){
		
		// backpropagation 1: do backpropagation in output layer
		this.layers[1].backpropagation();
		
		// backpropagation 2: pass output layer's errors to hidden layer
		((HiddenLayer) layers[0]).setBackpropagatedErrors(layers[1].getErrors(layers[0].getNumNeurons()));
		
		// backpropagation 3: do backpropagation in hidden layer
		layers[0].backpropagation();
	}
	
	/**
	 * create training and validation data. 
	 * @param features input features
	 * @param labels output feature(s)
	 */
	private void createDataset(Matrix features, Matrix labels) {
	
		// set the training data size
		int trainSize = (int)(trainingPercent * features.rows());

		// divide training and validation data
		if(useValidation){
			this.trainFeatures = new Matrix(features, 0, 0, trainSize, features.cols());
			this.trainTestLabels = new Matrix(labels, 0, 0, trainSize, labels.cols());
			this.trainLabels = customizeLabels(labels, 0, trainSize);
			this.validationFeatures = new Matrix(features, trainSize, 0, features.rows() - trainSize, features.cols());
			this.validationTestLabels = new Matrix(labels, trainSize, 0, labels.rows() - trainSize, labels.cols());
			this.validationLabels = customizeLabels(labels, trainSize, labels.rows() - trainSize);
		}
		else{
			this.trainFeatures = new Matrix(features, 0, 0, features.rows(), features.cols());
			this.trainTestLabels = new Matrix(labels, 0, 0, labels.rows(), labels.cols());
			this.trainLabels = customizeLabels(labels, 0, labels.rows());
		}
				
	}
		
	/**
	 * get collection of k input vectors from the dataset.
	 * @param features feature matrix (Matrix)
	 * @param labels label matrix (Matrix)
	 * @return k input vectors and a target (Pair<List<double[]>, double[]>)
	 */
	private Pair<List<double[]>, double[]> getInputs(Matrix features, Matrix labels){
		List<double[]> inputs = new ArrayList<double[]>(this.k);
		double[] target;
		int startIndex = this.random.nextInt(features.rows() - this.k);
		
		for(int row = startIndex; row < startIndex + k; row++){
			inputs.add(features.row(row));
		}
		target = labels.row(startIndex + k - 1);
	
		logger.info(this.getCurrentData(inputs, target));

		
		return new Pair<List<double[]>, double[]>(inputs, target);
		
	}
	
	/**
	 * print input feature sequence.
	 * @param inputs input sequence (List<double[]) 
	 * @param target target vector (double[])
	 * @return
	 */
	private String getCurrentData(List<double[]> inputs, double[] target) {
		StringBuilder sb = new StringBuilder();
		Formatter formatter = new Formatter(sb);
		
		formatter.format("Chosen Features:\n");
		for(double[] input : inputs){
			formatter.format(printArray(input));
		}
		formatter.format("Chosen Labels:\n");
		formatter.format(printArray(target));
		return formatter.toString();
	}

	/** 
	 * print array in a string format.
	 * @param array double[]
	 * @return [num, num, ...., num] format string
	 */
	public static String printArray(double[] array){
		StringBuilder sb = new StringBuilder();
		Formatter formatter = new Formatter(sb);
		formatter.format("[");
		for(int i = 0; i < array.length; i++){
			formatter.format("%.2f", array[i]);
			if(i == array.length - 1){
				formatter.format("]\n");
			}
			else{
				formatter.format(",");
			}
		}
		return formatter.toString();
		
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
	}

	/**
	 * predict output value in BPTT setting.
	 * @param features multiple consecutive features for time series prediction (List<double[]>)
	 * @param labels predicted answer (double[])
	 * @throws Exception 
	 */
	public void predict(List<double[]> features, double[] labels) throws Exception{
		
		for(int i = 0; i < labels.length; i++){
			labels[i] = this.getOutput(features);
		}
		
	}

	/**
	 * compute a single output value
	 * @param features feature time series
	 * @return a value (double)
	 * @throws Exception
	 */
	private double getOutput(List<double[]> features) throws Exception {
		double[] output = this.feedforward();

		if(output.length == 1){ // if there is only one node in the output layer
			return (output[0] > 0.5) ? 1.0 : 0.0; // return 1 or 0
		}
		else{ // if there are more than 2 nodes
			int index = 0;
			double max = output[index];

			logger.info("Output Array: " + printArray(output));

			for(int i = 1; i < output.length; i++){
				if(max < output[i]){
					max = output[i];
					index = i;
				}
			}
			return (double) index;
		}

	}
	
	
	/**
	 * convert single-column output labels with multiple classes 
	 * to multiple columns with 1's and 0's.
	 * this is just for labels with one column and not applicable for labels with multiple cols
	 * @param labels original label set
	 * @param rowStart start row index
	 * @param trainSize # of rows used to be contained
	 * @return modified label set
	 */
	private Matrix customizeLabels(Matrix labels, int rowStart, int trainSize) {
		
		int numClass = labels.valueCount(0); 
		Matrix fixedLabels;
		
		if(numClass == 0 || numClass == 2){ // if # class is just one (continuous) or binary
			fixedLabels = new Matrix(labels, rowStart, 0, trainSize, labels.cols()); // just copy the labels as it is
		}
		else{ // if there are more than 3 classes
			fixedLabels = new Matrix();
			fixedLabels.setSize(trainSize, labels.valueCount(0));
			for(int row = 0; row < trainSize; row++){
				fixedLabels.set(row, (int)labels.row(row)[0], 1.0);
			}
		}
		return fixedLabels;
	}

	/**
	 * get name of this learner (BPTT)
	 * @return learner name
	 */
	public String getName(){
		return "BPTT";
	}

	/**
	 * measure prediction accuracy especially for BPTT.
	 * @param features feature matrix (Matrix)
	 * @param labels label matrix (Matrix)
	 * @param confusion confusion matrix (Matrix)
	 */
	public double measureAccuracy(Matrix features, Matrix labels, Matrix modifiedLabels, Matrix confusion) throws Exception
	{
		
		if(features.rows() != labels.rows())
			throw(new Exception("Expected the features and labels to have the same number of rows"));
		if(labels.cols() != 1)
			throw(new Exception("Sorry, this method currently only supports one-dimensional labels"));
		if(features.rows() == 0)
			throw(new Exception("Expected at least one row"));

		int labelValues = labels.valueCount(0);
		logger.info("Value Count: " + labelValues);
		if(labelValues == 0) // If the label is continuous...
		{
			logger.info("continous accuracy measure");
			// The label is continuous, so measure root mean squared error
			double[] pred = new double[1];
			double sse = 0.0;
			for(int i = 0; i < accuracyTrial; i++)
			{
				this.currentlySelectedFeatures = this.getInputs(features, modifiedLabels);
				pred[0] = 0.0; // make sure the prediction is not biassed by a previous prediction
				predict(this.currentlySelectedFeatures.getFirst(), pred);
				double delta = this.currentlySelectedFeatures.getSecond()[0] - pred[0]; // target - prediction
				sse += (delta * delta);
			}
			return Math.sqrt(sse / accuracyTrial);
		}
		else
		{
			logger.info("nominal accuracy measure");
			// The label is nominal, so measure predictive accuracy
			if(confusion != null)
			{
				confusion.setSize(labelValues, labelValues);
				for(int i = 0; i < labelValues; i++)
					confusion.setAttrName(i, labels.attrValue(0, i));
			}
			int correctCount = 0;
			double[] prediction = new double[1];
			for(int i = 0; i < accuracyTrial; i++)
			{
				this.currentlySelectedFeatures = this.getInputs(features, modifiedLabels); // feature series
				int targ = this.getIndex(this.currentlySelectedFeatures.getSecond()); // target value
				if(targ >= labelValues)
					throw new Exception("The label is out of range");
				predict(this.currentlySelectedFeatures.getFirst(), prediction);
				int pred = (int)prediction[0];
				if(confusion != null)
					confusion.set(targ, pred, confusion.get(targ, pred) + 1);

				logger.info("Target: " + targ + "  Prediction: " + pred);
				
				if(pred == targ)
					correctCount++;
			}
			return (double)correctCount / accuracyTrial;
		}
	}
	
	/**
	 * reconvert output values converted by customizeData() method for accuracy measure.
	 * @param outputClasses converted output values
	 * @return index (int) which is the same as the output value
	 */
	private int getIndex(double[] outputClasses){
		int i = 0;
		logger.info("output class vector: " + printArray(outputClasses));
		for(i = 0; i < outputClasses.length; i++){
			if(outputClasses[i] == 1.0)
				return i;
		}
		return i;
	}
	
	
}

package CS678.MLP;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.omg.CosNaming._BindingIteratorImplBase;

import cs678.tools.Matrix;

public class NeuralNet extends SupervisedLearner{

	final static double trainPercent = 0.3; // training data size
	final static long seed = 1L; // random number seed
	final static int numNodeMultiple = 1; // # multiples for hidden nodes
	final static int maxIteration = 1; // max # of epochs
	final static int noImprovedmentLimit = 10; // the training stops if there is no improvement in this number of epochs after the last improvement
	final static boolean printout = true; // print out the log
	final static boolean useValidation = false; // option for validation 
	final static double defAlpha = 0.0; // default alpha value
	
	private Random rand; // random number generator
	private double eta; // learning rate
	private int numHidden; // # of hidden layers
	private int[] numHiddenNodes; // # of neurons in the hidden layers
	private int numOutputNodes; // # of neurons in the output layers
	private double alpha; // momentum
	private double trainAccuracy; // accuracy rate for training data
	private double validationAccuracy; // accuracy rate for validation data
	
	private Layer[] layers; // layers used in this MLP (# hidden + output)
	private Matrix trainFeatures; // training features
	private Matrix trainLabels; // training labels (multiple columns)
	private Matrix trainTestLabels; // train labels (single columns)
	private Matrix validationFeatures; // validation features
	private Matrix validationLabels; // validation labels (multiple columns)
	private Matrix validationTestLabels; // validation labels (single columns)
	
	public NeuralNet(){
		this.eta = 0.1; // learning rate
		this.numHidden = 1; // # hidden layer
		this.numHiddenNodes = new int[] {0}; // empty
		this.numOutputNodes = 0; // empty
		this.alpha = defAlpha; // no momentum
		this.trainAccuracy = 0; // no match
		this.validationAccuracy = 0; // no match
	}
	
	public NeuralNet(Random rand){
		this();
		this.rand = rand;
	}
	
	public NeuralNet(Random rand, double eta){
		this(rand);
		this.eta = eta;
	}
	
	public NeuralNet(Random rand, double eta, int numHidden){
		this(rand, eta);
		this.numHidden = numHidden;
	}
	
	public NeuralNet(Random rand, double eta, int numHidden, int numOutputNodes){
		this(rand, eta, numHidden);
		this.numOutputNodes = numOutputNodes;
	}
	
	public NeuralNet(Random rand, double eta, int numHidden, int numOutputNodes, double alpha){
		this(rand, eta, numHidden, numOutputNodes);
		this.alpha = alpha;
	}
	
	public NeuralNet(Random rand, double eta, int numHidden, int numOutputNodes, 
			double alpha, int ... numHiddenNodes){
		this(rand, eta, numHidden, numOutputNodes, alpha);
		setNumHiddenNodes(numHiddenNodes);
	}
	
	public NeuralNet(Random rand, int numHidden, int ... numHiddenNodes){
		this(rand, 0.1, numHidden);
		this.setNumHiddenNodes(numHiddenNodes);
	}
	
	public void setNumHiddenNodes(int[] numHiddenNodes){
		int[] hiddenNodes;
		if(numHiddenNodes.length > this.numHidden)
			hiddenNodes = Arrays.copyOf(this.numHiddenNodes, numHidden); // Truncate the excessive hidden nodes specification
		else if(numHiddenNodes.length < this.numHidden){
			hiddenNodes = Arrays.copyOf(this.numHiddenNodes, numHidden);
			for(int i = 0; i < hiddenNodes.length; i++){
				if(hiddenNodes[i] == 0){
					hiddenNodes[i] = hiddenNodes[i-1] * numNodeMultiple; // what if i = 0 is also 0??
				}
			}
		}
		else{
			hiddenNodes = numHiddenNodes;
		}
		this.numHiddenNodes = hiddenNodes;
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {

		for(int i = 0; i < labels.length; i++){
			labels[i] = getOutput(features);
		}
		
	}

	/**
	 * return the best guess
	 * @return
	 */
	private double getOutput(double[] features) throws Exception{
		
		double[] input = createInputFeatureVector(features);
		double[] output = feedForward(input);
		
		if(output.length == 1){ // if there is only one node in the output layer
			return (output[0] > 0.5) ? 1.0 : 0.0; // return 1 or 0
		}
		else{ // if there are more than 2 nodes
			int index = 0;
			double max = output[index];
			for(int i = 1; i < output.length; i++){
				if(max < output[i]){
					max = output[i];
					index = i;
				}
			}
			return (double) index;
		}
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {

		// variables for analysis
		Map<Integer, Double> mse = new HashMap<Integer, Double>(); // map for mean squared error
		Map<Integer, Double> misclassification = new HashMap<Integer, Double>(); // map for mis-classification rate
		/* 
		 * initialize layers (hidden + output)
		 */

		// set # hidden nodes if there is one hidden layer
		setNumHiddenNodes(features);
				
		// set # output nodes (# class = # nodes)
		setNumOutputNodes(labels);
		
		// instantiate layers
		instantiateLayers();

		// set weight vectors in the neurons
		setWeights(features);
				
		/*
		 * Training
		 */
		
		// divide the data into training and validation
		// default training : validation = 8 : 2
		createDataset(features, labels);

		System.out.println(features.columnMax(0));
		
		int noImprovementCount = 0; // counter to check if there is improvement 
		
		if(printout){
			System.out.println("\nEpoch: 0\tOriginal Accuracy: " + this.trainAccuracy);
		}
		
		int epoch = 0;
		
		for(epoch = 0; epoch < maxIteration; epoch++){
			
			for(int row = 0; row < this.trainFeatures.rows(); row++){ // for each input vector
				// set input vector (features + bias)
				double[] inputVector = createInputFeatureVector(trainFeatures.row(row));
				if(printout)
					System.out.println("Epoch: " + (epoch+1) + "   Data instance: " + (row+1));
				feedForward(inputVector);
				backpropagation(row); // do backprop for better weights				
					if(this.trainAccuracy == 1.0)
						break;
			}
			if(this.trainAccuracy == 1.0)
				break;
		
			// calculate mean squared error
			//computeMSE(mse, epoch+1);
			//computeMisclassification(misclassification, epoch+1);
			
			if(useValidation){
				if(isImproved("training")){
					if(printout){
						System.out.printf("Epoch: %d \tCurrent Training Set Auccracy: %1.10f", (epoch+1), this.trainAccuracy);
						System.out.printf("\tCurrent Validation Set Auccracy: %1.10f\n", this.validationAccuracy);
					}
					noImprovementCount = 0;
					if(!isImproved("validation")){
						noImprovementCount++;
					}
				}
//				if(noImprovementCount > noImprovedmentLimit){ // if no improvement count exceed the limit
//					break; // get out and end the training
//				}
//				else{
//					noImprovementCount++; // increment counter
//				}				
			}
			else{
//				if(isImproved("training")){
					//noImprovementCount = 0;
					if(printout){
						System.out.printf("Epoch: %d \tCurrent Training Set Auccracy: %1.10f\n", (epoch+1), this.trainAccuracy);
					}
//				}
//				if(noImprovementCount > noImprovedmentLimit){ // if no improvement count exceed the limit
//					break; // get out and end the training
//				}
//				else{
//					noImprovementCount++; // increment counter
//				}
			}
			this.shuffleData(trainFeatures, trainLabels);
		}
		
		System.out.println("Total Epochs: " + (epoch+1));
		//exportCSV(mse, "mse.csv");
		//exportCSV(misclassification, "misclassification.csv");
	}

	/**
	 * calculate misclassification rate
	 * @param misclassification	map for csv data
	 * @param epoch current epoch
	 */
	private void computeMisclassification (
			Map<Integer, Double> misclassification, int epoch) throws Exception {
				
		misclassification.put(epoch, (1.0 - this.trainAccuracy));
		
	}

	/**
	 * export MSE results to csv
	 * @param mse
	 */
	
	private void exportCSV(Map<Integer, Double> map, String fileName) throws Exception{

		File file = new File("data/" + fileName);
		FileWriter filewriter = new FileWriter(file);
		BufferedWriter bw = new BufferedWriter(filewriter);
		PrintWriter pw = new PrintWriter(bw);
		
		for(int index = 1; index < map.size(); index++){
			pw.printf("%d,%1.9f\n", index, map.get(index)); // epoch,MSEValue
		}
		pw.close();
	}

	/**
	 * calculate mean squared error 
	 * @param mse variable to store MSE (Epoch -> MSE)
	 * @param epoch current epoch
	 */
	private void computeMSE(Map<Integer, Double> mse, int epoch) throws Exception{
		
		double[] guess;
		double[] gold;
		
		double sum = 0.0;
		
		for(int row = 0; row < trainFeatures.rows(); row++){
			gold = trainLabels.row(row);
			guess = feedForward(createInputFeatureVector(trainFeatures.row(row)));
			for(int i = 0; i < gold.length; i++){
				sum += Math.pow((gold[i] - guess[i]), 2.0); // sum all squared error (SSE)
			}
		}
		
		mse.put(epoch, sum/trainFeatures.rows()); // set epoch -> MSE
		
	}

	private void setNumHiddenNodes(Matrix features) {
		if(this.numHiddenNodes.length == 1 && this.numHiddenNodes[0] == 0){ // if tehre is one hidden layer and the hidden layer nodes are yet unspecified
			this.numHiddenNodes[0] = (features.cols()) * numNodeMultiple + 1; // set the hidden layer nodes as the 2 times # inputs + 1 (bias)
			//this.numHiddenNodes[0] = 10;
		}
	}
	
	private void setNumOutputNodes(Matrix labels) {
		/*
		 * 5 cases
		 * 1) output feature が一個でcontinuous -- one node
		 * 2) output feature が一個でclassがbinary -- one node
		 * 3) output feature が一個でclassがtertiary以上 -- two or more nodes according to the # classes
		 * 4) output feature が複数で全部continuous -- nodes with # features
		 * 5) output feature が複数で全部がbinary -- nodes with # features 
		 * 6) output feature が複数で全部バラバラ(continuousとかbinaryとか) -- もう分からん。気にせん。
		 * 今回のlabはirisとvowel. つまりケースが(3)のみ。なのでoutput nodesの形式は一通りでいい、と思う。
		 * 今回はとりあえずclass別にnodeを形成。つまりMultiplePerceptronと同じようにしてみますよ。
		 */
		if(this.numOutputNodes == 0){ // if # output nodes are not specified
			if(labels.cols() == 1){ // if # output col is 1
				if(labels.valueCount(0) == 2) // if the output values are binary
					this.numOutputNodes = 1; 
				else if(labels.valueCount(0) > 2) // if the output values are more than binary
					this.numOutputNodes = labels.valueCount(0); // set # output nodes based on # outputs
				else // if the output value is continuous
					this.numOutputNodes = 1; // # output is 1
			}
			else{ // if # output cols are more than 1
				// ここは今回はkick inしないはず。とりあえずこのままで。
				this.numOutputNodes = labels.cols(); // set # output nodes to # of output features
			}
		}
		
		if(printout){
			System.out.println("# of output nodes: " + this.numOutputNodes);
		}
	}

	private void instantiateLayers() {
		this.layers = new Layer[this.numHidden + 1]; // # hidden layers + 1 output layer 
		for(int i = 0; i < layers.length - 1; i++){
			this.layers[i] = new HiddenLayer(this.numHiddenNodes[i], this.eta, this.rand, this.alpha); // set nodes in the hidden layers
			if(printout)
				System.out.println("# of hidden nodes: " + this.numHiddenNodes[i]);
		}
		this.layers[this.layers.length-1] = new OutputLayer(this.numOutputNodes-1, this.eta, this.rand, this.alpha); // initialize output layer		

		if(printout){
			System.out.println("# of layers: " + this.layers.length);
		}
	}

	private void setWeights(Matrix features) {
		
		for(int i = 0; i < layers.length; i++){
			if(printout){
				System.out.println("Layer " + (i+1) + " Weights: ");
			}
			if(i == 0){ // if the lowest layer
				layers[i].setWeights(features.cols() + 1); // # weights are the same as input size + 1 (bias)
			}
			else{
				layers[i].setWeights(layers[i-1].getNumNeurons()); // # weights are the same as # neurons in the lower layer
			}
		}
		
	}

	/**
	 * create training and validation data
	 * @param features input features
	 * @param labels output feature(s)
	 */
	private void createDataset(Matrix features, Matrix labels) {
	
		// shuffle dataset
		shuffleData(features, labels);

		// set the training data size
		int trainSize = (int)(trainPercent * features.rows());

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
	 * customize the labelsets for training & validation
	 * # of output class will be converted to # of cols
	 * this is just for labels with one column and not applicable for labels with multiple cols
	 * @param labels original label set
	 * @param rowStart start row index
	 * @param trainSize # of rows used to be contained
	 * @return modified label set
	 */
	private Matrix customizeLabels(Matrix labels, int rowStart, int trainSize) {
		
		int numClass = labels.valueCount(0); 
		Matrix fixedLabels;
		
		if(printout){
			System.out.println("# classes: " + numClass);
		}
		
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


	private void shuffleData(Matrix features, Matrix labels) {
		this.rand = new Random(seed); // set the random seed to 1L
		features.shuffle(rand);
		this.rand = new Random(seed); // set the random seed to 1L
		labels.shuffle(rand);		
	}

	/**
	 * create input vector (features + 1 for bias)
	 * @param row original input vector
	 * @return modified input vector (original inputs + bias)
	 */	
	private double[] createInputFeatureVector(double[] row) {
		
		double[] inputFeautureVector = Arrays.copyOf(row, row.length+1);
		inputFeautureVector[row.length] = 1.0;
		return inputFeautureVector;
		
	}

	/**
	 * feed-forward process
	 * pre-condition: the input vector should be the same length as the weight vector in the lowest layer
	 * post-condition: the output vector should be the same length as the target vector 
	 * @param inputVector the input vector with bias input (1.0)
	 * @return output vector to be compared to the target vector
	 * @throws Exception 
	 */
	private double[] feedForward(double[] inputVector) throws Exception{
		
		for(int layerNum = 0; layerNum < layers.length; layerNum++){
			if(layerNum == 0) {// if lowest layer
				layers[layerNum].setInput(inputVector); // feed actual input + bias
			}
			else { // if higher layer
				layers[layerNum].setInput(layers[layerNum-1].getOutput()); // feed the output vector from the lower layer
			}
		}
		
		return layers[layers.length-1].getOutput(); // output vector generated at the output layer
	}

	/**
	 * check if the current accuracy rate is better than the previous one
	 * this condition decides if the weights need to be updated or not
	 * @param type training: check with training data validation: check with validation data
	 * @return true if the current accuracy rate is better than the previous one; false otherwise
	 * @throws Exception
	 */
	private boolean isImproved(String type) throws Exception{
		
		double currentAccuracy;
		
		if(type.equalsIgnoreCase("training")){ // if checking on training data
			currentAccuracy = this.measureAccuracy(trainFeatures, trainTestLabels, null);
			if(printout){
				System.out.println("Current Accuracy: " + currentAccuracy);
			}
			if(this.trainAccuracy < currentAccuracy){
				this.trainAccuracy = currentAccuracy;
				return true;
			}
			else{
				return false;
			}
		}
		else { // if validation
			currentAccuracy = this.measureAccuracy(validationFeatures, validationTestLabels, null);
			if(this.validationAccuracy <= currentAccuracy){
				this.validationAccuracy = currentAccuracy;
				return true;
			}
			else {
				return false;
			}
		}		
	}

	private void backpropagation(int row) throws Exception{
		
		for(int layerNum = this.layers.length-1; layerNum >= 0; layerNum--){ // go backwards
			
			if(layerNum == layers.length-1){ // if the current layer is output layer
				((OutputLayer) this.layers[layerNum]).setTarget(trainLabels.row(row));
				this.layers[layerNum].computeCoefs(); // compute all (t_k - y_k) & all Sum_k wjk * delta_ok
				//this.layers[layerNum].computeDeltas(); // compute delta_ok = (t_k - y_k) y_k (1 - y_k)
			}
			else{ // if the current layer is lower ones
				((HiddenLayer) layers[layerNum]).setUpperCoefs(layers[layerNum+1].getCoefs()); // get all Sum_k w_jk * delta_ok from the upper layer
				layers[layerNum].computeDeltas(); // compute delta_hj for this layer
				if(layerNum > 0){
					layers[layerNum].computeCoefs(); // compute coefs for the lower hidden layer
				}
			}
			// update the weights
			layers[layerNum].updateWeights();
		}
		
	}

}

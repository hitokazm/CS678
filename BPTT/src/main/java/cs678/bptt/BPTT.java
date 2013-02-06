package cs678.bptt;

import java.util.Random;

import cs678.tools.Matrix;
import cs678.tools.SupervisedLearner;

public class BPTT extends SupervisedLearner {

	final static double trainingPercent = 0.7; // amount of training data
	final static long seed = 1L; // random number seed
	final static int maxInteration = 1000; // # training iterations
	final static boolean useValidation = false; // option for validation data check
	
	
	private double learningRate; // learning rate (eta)
	private double momentum; // momentum (alpha)
	private double trainAccuracy; // training data accuracy
	private double validationAccuracy; // validation set accuracy
	private Random random; // random number generator

	private int numHiddenNodes; // # hidden nodes, don't need to be array because BPTT has a single layer
	private int numOutputNodes; // # output nodes
	
	private int k; // size of history
	
	private Layer[] layers; // hidden and output layers
	private Matrix trainFeatures; // training features
	private Matrix trainLabels; // training labels (multiple columns)
	private Matrix trainTestLabels; // train labels (single columns)
	private Matrix validationFeatures; // validation features
	private Matrix validationLabels; // validation labels (multiple columns)
	private Matrix validationTestLabels; // validation labels (single columns)
	
	/**
	 * constructor
	 */
	public BPTT(){
		this.learningRate = 0.1; // default value
		this.momentum = 0.0; // default value 
		this.random = new Random(seed);
		this.numHiddenNodes = 0;
		this.numOutputNodes = 0;
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
	 * constructor
	 * @param numHiddenNodes: specified # hidden nodes
	 */
	public BPTT(int numHiddenNodes){
		this();
		this.setNumHiddenNodes(numHiddenNodes);
	}
	
	/**
	 * constructor
	 * @param numHiddenNodes: specified # hidden nodes (int)
	 * @param numOutputNodes: specified # output nodes (int)
	 */
	public BPTT(int numHiddenNodes, int numOutputNodes){
		this(numHiddenNodes);
		this.setNumOutputNodes(numOutputNodes);
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

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// create hidden layer
		int inputSize = features.cols()*3+1; // #features (actual input size) + # hidden neurons (2 * #features) + 1 (bias)
		Layer hiddenLayer = new HiddenLayer(features.cols()*2, this.learningRate, random, this.momentum, this.k, inputSize);
		
		// create output layer
		inputSize = features.cols() * 2 + 1; // # hidden neurons (2 * #features) + 1 (bias)
		this.setNumOutputNodes(labels); // compute # output nodes
		Layer outputLayer = new OutputLayer(this.numOutputNodes, this.learningRate, this.random, this.momentum);

	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
	}

}

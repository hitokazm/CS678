package cs678.bptt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Neuron {

	/*
	 * Definitions:
	 * net = dot product of input from the lower neurons and associated weights
	 * output = a single value based on the sigmoid function
	 */
	
	private double[] weights; // current weights between me and lower neurons
	private double[] input; // input values
	private double learningRate; // learning rate (eta)
	private double net; // net value
	private double output; // output value
	private double errorRate; // error rate (delta)
	private double momentum; // momentum (alpha)
	private Random random; // random number generator
	
	private int k; // size of history
	private List<double[]> deltaWeights; // update weight values. There might be multiple deltaWs' if this neuron is in the hidden layer. 
	
	/**
	 * constructor
	 */
	public Neuron(){
		this.momentum = 0.9; 
		this.random = new Random();
		this.deltaWeights = new ArrayList<double[]>();
	}
	
	/**
	 * constructor
	 * @param learningRate: learning rate (eta, double)
	 */
	public Neuron(double learningRate){
		this();
		this.setLearningRate(learningRate);
	}
	
	/**
	 * constructor
	 * @param eta: learning rate (double)
	 * @param random: random number generator (Random)
	 */
	public Neuron(double eta, Random random){
		this(eta);
		this.setRandom(random);
	}
	
	/**
	 * constructor
	 * @param eta (learning rate, double)
	 * @param random 
	 * @param alpha (momentum, double)
	 */
	public Neuron(double eta, Random random, double alpha){
		this(eta, random);
		this.setMomentum(alpha);
	}
	
	
	/**
	 * constructor
	 * @param eta (learning rate, double)
	 * @param random 
	 * @param alpha (momentum, double)
	 * @param k (size of history, int)
	 */
	public Neuron(double eta, Random random, double alpha, int k){
		this(eta, random, alpha);
		this.setK(k);
	}
	
	/**
	 * constructor
	 * @param eta (learning rate, double)
	 * @param random 
	 * @param alpha (momentum, double)
	 * @param k (size of history, int)
	 * @param inputSize (int feature size which determines the number of weights, int)
	 */
	public Neuron(double eta, Random random, double alpha, int k, int inputSize){
		this(eta, random, alpha, k);
		this.setWeights(inputSize);
	}
	
	/**
	 * set history size for BPTT.
	 * @param k (int)
	 */
	private void setK(int k) {
		this.k = k;
	}

	/**
	 * set the learning rate (eta)
	 * @param eta: learning rate (double)
	 */
	public void setLearningRate(double eta){
		this.learningRate = eta;
	}
	
	/**
	 * set momentum (alpha)
	 * @param alpha: momentum (double)
	 */
	public void setMomentum(double alpha){
		this.momentum = alpha;
	}
	
	/**
	 * set initial weights.
	 * @param size: # weights (i.e., # arrows from this neuron to the upper layer; int)
	 */
	protected void setWeights(int size){
		if(this.weights == null)
			this.weights = this.setInitialWeights(size);
	}

	/**
	 * print out weight values.
	 */
	public void printWeights(){
		StringBuilder sb = new StringBuilder();
		sb.append("Weight Values: [");
		for (double weight : this.weights){
			sb.append(weight);
			if(weight != this.weights[this.weights.length-1]){
				sb.append(",");
			}
		}
		sb.append("]\n");
		System.out.println(sb);
	}
	
	/**
	 * assign initial weights using Gaussian Distribution
	 * @param size: size of weight vector (int)
	 * @return initial weights -0.1 < each value < 0.1. 
	 */
	private double[] setInitialWeights(int size) {
		
		double[] initialWeights = new double[size];
		
		for(int i = 0; i < initialWeights.length; i++){
			double weight;
			do{
				weight = random.nextGaussian();
			}while(Math.abs(weight) > 0.1);
			initialWeights[i] = weight;
		}
		
		return initialWeights;
	}
	
	/**
	 * set pre-specified random number generator.
	 * @param random (Random)
	 */
	public void setRandom(Random random){
		this.random = random;
	}
	
	/**
	 * compute net value using dot product (Sigma feature dot weights)
	 * @param input: input values (double[])
	 */
	private void computeNetValue(double[] input){
		
		try{
			this.net = dotProduct(input, this.weights);
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}

	/**
	 * compute output value with sigmoid function
	 * @param input input vector (double[])
	 */
	private void computeOutputValue(double[] input){
		this.computeNetValue(input);
		this.output = 1.0 / (1.0 + Math.exp(-1.0 * this.net));
	}
	
	/**
	 * get output value.
	 * @return output (double)
	 */
	public double getOutput() throws Exception {
		
		if(this.input == null){
			throw new Exception("No input features.");
		}
		
		computeOutputValue(this.getInput());
		return this.output;
	}
	
	/**
	 * compute dot product.
	 * @param vector1 (double[])
	 * @param vector2 (double[])
	 * @return
	 * @throws Exception
	 */
	public static double dotProduct(double[] vector1, double[] vector2) throws Exception {
		
		double dotProduct = 0;
		
		if (vector1.length != vector2.length)
			throw new Exception("The feature and weight lengths are different.");
		else{
			for(int i = 0; i < vector1.length; i++){
				dotProduct += vector1[i] * vector2[i];
			}
		}
		
		return dotProduct;
	}
	
	/**
	 * set input values.
	 * @param input (double[])
	 */
	public void setInput(double[] input){
		this.input = input;
	}
	
	/**
	 * get input vector.
	 * @return input (double[])
	 */
	private double[] getInput(){
		return this.input;
	}
	
	/**
	 * get this neuron's error rate.
	 * @return delta (error rate, double)
	 */
	public double getErrorRate(){
		return this.errorRate;
	}
	
	/**
	 * get current weights between this neuron and the input vector.
	 * @return weights (double[])
	 */
	public double[] getWeights(){
		return Arrays.copyOf(this.weights, this.weights.length);
	}
	
	/**
	 * compute error rate (delta) of this neuron.
	 * @param update
	 * 		for output node, update = (t_j - o_j) (j: the neuron's index, t: target, o: output value)
	 * 		for hidden, update = Sum_k w_jk * delta_k (k: index for the upper neuron, 
	 * 		j: the neuron's own index; delta: upper neuron's error rate)
	 * f'(net) = output * ( 1 - output). 
	 */
	private void computeErrorRate(double error){
		double fPrimeNet = this.output * (1.0 - this.output); // f'(net)
		this.errorRate = fPrimeNet * error; // set delta
	}
	
	
	/**
	 * compute deltaW_ij.
	 * delta w_ij = eta (learning rate) * O_i (output of a child neuron i / input from a child neuron i) delta_j (error rate).
	 */
	private void computeDeltaWeights(){
		
		double[] deltaWeights = new double[this.weights.length];
		
		for(int i = 0; i < deltaWeights.length; i++){
			deltaWeights[i] = this.learningRate * this.input[i] * this.errorRate;
		}
		
		this.deltaWeights.add(deltaWeights);
		
	}
	
	/**
	 * receive the propagated error from the upper neuron(s) for weight updates.
	 * @param error: error value provided by the upper neurons(s) (double)
	 */
	public void backpropagation(double error){
		this.computeErrorRate(error);
		this.computeDeltaWeights();
	}
	
	/**
	 * update weights. If this neuron is in the hidden layer, the delta ws' will be summed and then update the weights.
	 */
	public void update() throws Exception{
		
		if(this.deltaWeights.size() != this.k){
			throw new Exception("The delta ws' are not properly set.");
		}
		
		this.updateWeights(this.sumDeltaWeights());
		
	}
	
	/**
	 * sum up the delta weights.
	 * @return summed delta weights (double[], same length as this.weights)
	 */
	private double[] sumDeltaWeights(){
		
		double[] totalDeltas = new double[this.weights.length];
		
		for(int i = 0; i < this.weights.length; i++){
			double deltaW = 0.0;
			for(double[] deltaWeights : this.deltaWeights){
				deltaW += deltaWeights[i];
			}
			totalDeltas[i] = deltaW;
		}
		
		return totalDeltas;
				
	}
	
	/**
	 * update weights with given delta w_ij's.
	 * @param summedDeltaWeights: delta ws'. If this neuron is in the hidden layer, all the delta ws' will be summed and then added to the current weights.
	 */
	private void updateWeights(double[] summedDeltaWeights){
		
		try{
			this.weights = add(this.weights, this.sumDeltaWeights());
		}
		catch(Exception e){
			e.printStackTrace();
		}
		this.clearDeltaWeights();
		
	}

	/**
	 * add two vectors in the same length.
	 * @param vector1
	 * @param vector2
	 * @return added vector (double[])
	 * @throws Exception: the length must be the same.
	 */
	public static double[] add(double[] vector1, double[] vector2) throws Exception {
				
		if(vector1.length != vector2.length){
			throw new Exception("The lengths of the given vectors are different.");
		}
		
		double[] vector = new double[vector1.length];
		
		for(int i = 0; i < vector.length; i++){
			double value = vector1[i] + vector2[i];
			vector[i] = value;
		}
		
		return vector;
		
	}
	
	/**
	 * clear delta weights for the next training phase.
	 */
	private void clearDeltaWeights(){
		this.deltaWeights.clear();
	}
	
}

package cs678.bptt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

public class OutputLayer extends Layer {

	private double[] input; // input vector. Since this is a output layer, it accepts only one input. 
	private double[] errors; // error (t_i - o_i) where i is the index of neuron i in this layer
	private double[] errorRates; // error rates (delta), computed by delta_j = (t_j - o_j) o_j (1 - o_j)
	private double[] targets; // target value vector; usually one value
	private int numLowerNeurons; // # of neurons in the lower hidden layer
	
	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 */
	public OutputLayer(int numNeurons){
		super(numNeurons);
	}
	
	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 */
	public OutputLayer(int numNeurons, double eta){
		super(numNeurons, eta);
	}
	
	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 * @param random: random number generator (Random)
	 */
	public OutputLayer(int numNeurons, double eta, Random random){
		super(numNeurons, eta, random);
	}
	
	/**
	 * constructor
	 * @param numNeurons # neurons in this layer (int)
	 * @param eta learning rate (double)
	 * @param random random number generator (Random)
	 * @param alpha momentum (double)
	 */
	public OutputLayer(int numNeurons, double eta, Random random, double alpha){
		super(numNeurons, eta, random, alpha, 1); // k is always 1 because no recurrence.
	}
	
	/**
	 * constructor
	 * @param numNeurons # neurons in this layer (int)
	 * @param eta learning rate (double)
	 * @param random random number generator (Random)
	 * @param alpha momentum (double)
	 * @param targets target value vector (double[])
	 */
	public OutputLayer(int numNeurons, double eta, Random random, double alpha, double[] targets){
		this(numNeurons, eta, random, alpha);
		this.setTargets(targets);
	}
	
	/**
	 * constructor
	 * @param numNeurons # neurons in this layer (int)
	 * @param eta learning rate (double)
	 * @param random random number generator (Random)
	 * @param alpha momentum (double)
	 * @param inputSize input size (int)
	 * @param targets target value vector (double[])
	 */
	public OutputLayer(int numNeurons, double eta, Random random, double alpha, int inputSize, double[] targets){
		super(numNeurons, eta, random, alpha, 1, inputSize);
		this.setTargets(targets);
	}

	/**
	 * set target values.
	 * @param targets target value vector (double[])
	 */
	public void setTargets(double[] targets) {
		this.targets = targets;		
	}

	/**
	 * set input vector.
	 * @param input list of input vectors (size: 1)
	 * @exception # input vector must be 1.
	 */
	public void setInput(List<double[]> input) throws Exception {
		
		if(input.size() != 1){
			throw new Exception("Ouput layer must have one input.");
		}
		
		this.input = input.get(0);
		
		for(Neuron neuron : super.getNeurons()){
			neuron.setInput(this.input);
		}
		
	}

	public void setNumHiddenNeurons(int numNeurons){
		this.numLowerNeurons = numNeurons;
	}
	
	/**
	 * compute error rates, based on delta_j = (t_j - o_j) o_j (1 - o_j).
	 * delta_j: error rate, which will be stored in errorRates vector (double[])
	 * (t_j - o_j): error, which is stored in errors vector (double[])
	 * o_j (1 - o_j): f'(net_j), which is provided by  
	 */
	public void backpropagation() {
		Neuron[] neurons = super.getNeurons(); // get all neurons
		this.errorRates = new double[neurons.length]; // get error rates
		this.computeErrors(); // compute errors (t - o) for all neurons
		for(int i = 0; i < neurons.length; i++){
			neurons[i].backpropagation(this.errors[i]); // compute error rate and delta w
			this.errorRates[i] = neurons[i].getErrorRate(); // set error rates (delta_j)
			try{
				neurons[i].update(); // update weights right after backprop because no recurrence
			}
			catch(Exception e){
				e.printStackTrace();
			}
		}
	}

	/**
	 * compute error values
	 * (t_i - o_i): t is a target value, o is an output of a neuron i.
	 */
	protected void computeErrors() {
		double[] errors = new double[this.targets.length];
		
		try{
			double[] outputs = super.getOutput();
			
			if(outputs == null){
				throw new Exception("Output vector must be non-null.");
			}
			
			if(errors.length != outputs.length){
				throw new Exception("Ouput and target vectors should be the same length.");
			}
			
			for(int i = 0; i < errors.length; i++){
				errors[i] = this.targets[i] - outputs[i];
			}
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
		this.errors = errors;
	}

	/**
	 * get error vector for the lower hidden layer. 
	 * The calculation is Sum_j (delta_j * w_ij), where j is the index of neurons in this layer 
	 * and i is the index of the neurons in the lower hidden layer.
	 * @return errors error vector size of # neurons in the lower hidden layer. (double[]).
	 */
	protected double[] getErrors(int numLowerNeurons) {
		
		List<Double> errors = new ArrayList<Double>(numLowerNeurons);
		int startIndex = this.getStartIndex(numLowerNeurons); //this.input.length - numLowerNeurons - 1;
		Neuron[] neurons = super.getNeurons();
	
		int max = startIndex + numLowerNeurons;
		
		for (int i = startIndex; i < max; i++) {
			double sum = 0.0;
			for (Neuron neuron : neurons) {
				double[] weights = neuron.getWeights();
				sum += neuron.getErrorRate() * weights[i]; // Sum_j (delta_j * w_ij)
			}
			errors.add(sum);
		}
		
		try{
			if(errors.size() > numLowerNeurons){
				throw new Exception("The number of errors should be the same as # of hidden neurons.");
			}
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
		// convert the ArrayList<Double> to double[]
		return ArrayUtils.toPrimitive(errors.toArray(new Double[errors.size()]));
	}
	
	protected int getStartIndex(int numLowerNeurons){
		return this.input.length - numLowerNeurons - 1;
	}
	
}

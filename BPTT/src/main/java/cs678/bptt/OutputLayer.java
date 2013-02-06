package cs678.bptt;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class OutputLayer extends Layer {

	private double[] input; // input vector. Since this is a output layer, it accepts only one input. 
	private double[] errors; // error (t_i - o_i) where i is the index of neuron i in this layer
	private double[] errorRates; // error rates (delta), computed by delta_j = (t_j - o_j) o_j (1 - o_j)
	private double[] targets; // target value vector
	
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
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 * @param random: random number generator (Random)
	 * @param alpha: momentum (double)
	 */
	public OutputLayer(int numNeurons, double eta, Random random, double alpha){
		super(numNeurons, eta, random, alpha, 1); // k is always 1 because no recurrence.
	}
	
	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 * @param random: random number generator (Random)
	 * @param alpha: momentum (double)
	 * @param targets: target value vector (double[])
	 */
	public OutputLayer(int numNeurons, double eta, Random random, double alpha, double[] targets){
		super(numNeurons, eta, random, alpha, 1);
		this.setTargets(targets);
	}
	
	/**
	 * set target values.
	 * @param targets: target value vector (double[])
	 */
	public void setTargets(double[] targets) {
		this.targets = targets;		
	}

	/**
	 * set input vector.
	 * @param input: list of input vectors (size: 1)
	 * @exception: # input vector must be 1.
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

	/**
	 * compute error rates, based on delta_j = (t_j - o_j) o_j (1 - o_j).
	 * delta_j: error rate, which will be stored in errorRates vector (double[])
	 * (t_j - o_j): error, which is stored in errors vector (double[])
	 * o_j (1 - o_j): f'(net_j), which is provided by  
	 */
	public void backpropagation() {
		Neuron[] neurons = super.getNeurons();
		this.errorRates = new double[neurons.length];
		for(int i = 0; i < neurons.length; i++){
			neurons[i].backpropagation(this.errors[i]); // compute error rate and delta w
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
	 * get error vector.
	 * @param errors: error vector (double[]).
	 */
	protected double[] getErrors() {
		return Arrays.copyOf(this.errors, this.errors.length);
	}

}

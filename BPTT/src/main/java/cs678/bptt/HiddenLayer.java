package cs678.bptt;

import java.util.ArrayList;
import java.util.Currency;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import cs678.tools.Matrix;

public class HiddenLayer extends Layer {
	
	private int k; // history size
	List<double[]> inputs; // series of inputs with size k (# rows)
	double[] errors; // error (Sum_j w_ij * delta_j) where i is the index of the neurons in this layer and j is the index of neurons in the upper layer
	//double[] input; // one instance from inputs + output from hidden layer (0.5s if at the beginning)
	
	private final static Logger logger = Main.logger;
	
	/**
	 * constructor
	 * @param numNeurons # neurons in this layer (int)
	 * @param eta learning rate (double)
	 * @param random random number generator (Random)
	 * @param alpha momentum (double)
	 * @param k history size (int)
	 */
	public HiddenLayer(int numNeurons, double eta, Random random, double alpha, int k){
		super(numNeurons, eta, random, alpha, k);
		this.setK(k);
	}
	
	/**
	 * constructor
	 * @param numNeurons # neurons in this layer (int)
	 * @param eta learning rate (double)
	 * @param random random number generator (Random)
	 * @param alpha momentum (double)
	 * @param k history size (int)
	 * @param inputSize input size (int)
	 */
	public HiddenLayer(int numNeurons, double eta, Random random, double alpha, int k, int inputSize){
		super(numNeurons, eta, random, alpha, k, inputSize);
		if (logger.getLevel().equals(Level.INFO))
			logger.info("Instantiate Hidden Layer --- " + "# Neurons: "
					+ numNeurons + "\tLearning Rate: " + eta + "\tK: " + k
					+ "\tInput Size: " + inputSize + "\n");
		this.setK(k);
	}
	
	/**
	 * set history size.
	 * @param k history size (int)
	 */
	private void setK(int k) {
		this.k = k;		
	}

	@Override
	public void setInput(List<double[]> inputs) throws Exception {
		
		this.inputs = new ArrayList<double[]>(k);
		
		double[] input = new double[1];  
		for(int i = 0; i < k; i++){
			input = new double[inputs.get(i).length + this.getNumNeurons() + 1]; // # input features + # neurons in this layer + 1 bias 
			int currentLength = inputs.get(i).length;
			for (int j = 0; j < currentLength; j++){
				input[j] = inputs.get(i)[j];
			}
			int j;
			if(i == 0){
				for(j = currentLength; j < currentLength + this.getNumNeurons(); j++){
					input[j] = 0.5;
				}
				input[j] = 1; // bias value
			}
			else{
				for(Neuron neuron : this.getNeurons()){
					neuron.setInput(this.inputs.get(this.inputs.size()-1));
				}
				double[] output = this.getOutput();
				int count = 0;
				for(j = currentLength; j < currentLength + this.getNumNeurons(); j++){
					input[j] = output[count];
					count++;
				}				
				input[j] = 1; // bias value
			}
			if(logger.getLevel().equals(Level.INFO))
				logger.info("Added Input: " + BPTT.printArray(input));
			this.inputs.add(input);
		}
		
		for(Neuron neuron : this.getNeurons()){
			neuron.setInput(input);
		}
		
		if(this.inputs.size() != this.k){
			throw new Exception("Input vector size and k are different.");
		}
		
	}

	/**
	 * get back-propageted errors from (1) the output layer or (2) the recurrent layer (virtual upper hidden layer).
	 * @param errors error vector passed from the upper layer (double[])
	 */
	public void setBackpropagatedErrors(double[] errors){
		this.errors = errors;
	}
	
	
	@Override
	public void backpropagation() {
		// the errors from the output layer should be stored before this process.
		Neuron[] neurons = super.getNeurons();
		int recurrent = 0;
		while(recurrent < k){ // backprop the errors based on the number k
			try {
				if (neurons.length != this.errors.length)
					throw new Exception("Neuron and error sizes are different."
							+ "#Neuron: " + neurons.length + " #Errors: "
							+ this.errors.length);
			} catch (Exception e) {
				e.printStackTrace();
			}
			for(int i = 0; i < neurons.length; i++){
				logger.info("backpropped error[" + i + "]: " + this.errors[i]);
				neurons[i].backpropagation(this.errors[i]);
			}
			this.errors = super.getErrors(neurons.length);
			recurrent++;
		}
		
		try{
			for(Neuron neuron : neurons){
				neuron.update();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void computeErrors() {
		
	}

	@Override
	protected int getStartIndex(int numLowerNeurons) {
		return this.inputs.get(0).length - numLowerNeurons - 1;
	}

}

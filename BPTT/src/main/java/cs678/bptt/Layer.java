package cs678.bptt;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public abstract class Layer {
	
	private int k; // size of history (time series)
	private Neuron[] neurons; // neurons in the layer
	private double[] output; // output values from this layer
	private double[] errorRates; // error rates (deltas) of the neurons in this layer
	
	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 */
	public Layer(int numNeurons){
		neurons = new Neuron[numNeurons]; // # for hidden, features + bias; for output, just the number
		this.output = new double[this.neurons.length]; // output vector (same size as neurons)
	}
	
	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 */
	public Layer(int numNeurons, double eta){
		this(numNeurons);
		for(int i = 0; i < neurons.length; i++){
			neurons[i] = new Neuron(eta);
		}
	}

	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 * @param random: random number generator (Random)
	 */
	public Layer(int numNeurons, double eta, Random random){
		this(numNeurons, eta);
		for(int i = 0; i < neurons.length; i++){
			neurons[i].setRandom(random);
		}
	}

	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 * @param random: random number generator (Random)
	 * @param alpha: momentum (double)
	 */
	public Layer(int numNeurons, double eta, Random random, double alpha){
		this(numNeurons, eta, random);
		for(int i = 0; i < neurons.length; i++){
			neurons[i].setMomentum(alpha);
		}
	}

	/**
	 * constructor
	 * @param numNeurons: # neurons in this layer (int)
	 * @param eta: learning rate (double)
	 * @param random: random number generator (Random)
	 * @param alpha: momentum (double)
	 * @param k: time series history size (int)
	 */
	public Layer(int numNeurons, double eta, Random random, double alpha, int k){
		this(numNeurons);
		for(int i = 0; i < neurons.length; i++){
			neurons[i] = new Neuron(eta, random, alpha, k);
		}
	}
	
	/**
	 * get output values of this layer.
	 * @return output vector (double[])
	 * @throws Exception: the input vector must be set in each neuron.
	 */
	public double[] getOutput() throws Exception {
		
		for(int i = 0; i < this.neurons.length; i++){
			output[i] = this.neurons[i].getOutput();
		}
		
		return this.output;
	}

	/**
	 * set input vector(s). If this is a hidden layer, 
	 * there are multiple input vectors according to the k value. 
	 * If this is an output layer, there is only one input vector.
	 * @param input: list of input vectors (output layer: 1 instance; 
	 * hidden layer: k instances)
	 * @exception: the number of input conforms to the specified number (output: 1, hidden: k).	
	 */
	public abstract void setInput(List<double[]> input) throws Exception;
	
	/**
	 * get # of neurons in this layer.
	 * @return # neurons (int)
	 */
	public int getNumNeurons(){
		return this.neurons.length;
	}
	
	/**
	 * set neurons' initial weights.
	 * @param size: vector size (int)
	 */
	private void setWeights(int size){
		for(Neuron neuron : this.neurons){
			neuron.setWeights(size);
		}
	}
	
	/**
	 * get neurons in this layer.
	 * @return neurons (Neuron[])
	 */
	protected Neuron[] getNeurons(){
		return this.neurons;
	}
	
	/**
	 * calculate error rates (deltas) and back-propagate the errors to the lower layer
	 * output layer: delta_j = (t_j - o_j) o_j (1 - o_j) (t: target, o: output value, j: index of the neurons/outputs, (t_j - o_j): errors of the neuron j, 
	 * o_j (1 - o_j): f'(net_j). This is done in the layer itself)
	 * hidden layer: delta_i = o_i (1 - o_i) Sum_j w_ij * delta_j (i: index of the neurons/outputs, j: index of the neurons in the upper layer, 
	 * w_ij: weights between the neuron i in this layer and neuron j in the upper layer; delta_j: error rate of the neuron j in the upper layer. 
	 * The summation part should be provided by the upper layer, passed by getErrors() method)
	 */
	public abstract void backpropagation();	

	/**
	 * get current error rates of the neurons in this layer.
	 * @return error rate vector (double[])
	 */
	public double[] getErrorRates(){
		return Arrays.copyOf(errorRates, this.errorRates.length);
	}
	
	/**
	 * calculate errors of the neurons in this layer for backpropagation.
	 *  for output layer, (t_j - y_j), where j is the index of the neurons in this layer and the corresponding output values 
	 *  for hidden layer, Sum_j (w_ij * delta_j), where j is the index of the neurons of the upper layer;
	 *  i is the index of the neurons in this layer.  
	 */
	protected abstract void computeErrors();
	
	/**
	 * get error vectors of this layer.
	 * @return error vector (double[])
	 */
	protected abstract double[] getErrors();
	
}

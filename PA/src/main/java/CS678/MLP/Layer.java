package CS678.MLP;

import java.util.Arrays;
import java.util.Random;

public abstract class Layer {

	final static boolean printout = false; 
	
	Neuron[] neurons; // neurons contained in this layer
	double[] input; // input values provided by the lower neurons or input vector
	double[] output; // output values to be passed up to the upper layer
	double[] deltas; // errors for each neuron in the layer
	
	/**
	 * constructor
	 * @param numNeurons Number of neurons in the layer
	 */
	public Layer(int numNeurons){
		neurons = new Neuron[numNeurons + 1]; // # features + bias
	}
	
	/**
	 * constructor
	 * @param numNeurons Number of neurons in the layer
	 * @param eta learning rate 
	 */
	public Layer(int numNeurons, double eta){
		this(numNeurons);
		for(int i = 0; i < neurons.length; i++){
			neurons[i] = new Neuron(eta);
		}
	}

	/**
	 * constructor
	 * @param numNeurons Number of neurons in the layer
	 * @param eta learning rate 
	 * @param rand random number generator 
	 */
	public Layer(int numNeurons, double eta, Random rand){
		this(numNeurons);
		for(int i = 0; i < neurons.length; i++){
			neurons[i] = new Neuron(eta, rand);
		}
	}

	/**
	 * constructor 
	 * @param numNeurons Number of neurons in the layer
	 * @param eta learning rate 
	 * @param rand random number generator 
	 * @param alpha momentum coefficient
	 */
	public Layer(int numNeurons, double eta, Random rand, double alpha){
		this(numNeurons);
		for(int i = 0; i < neurons.length; i++){
			neurons[i] = new Neuron(eta, rand, alpha);
		}
	}
	
	/**	
	 * return the output values to be passed up to the upper layer
	 * pre-condition: the input vector is already set 
	 * post-condition: provide a output vector based on the number of neurons in this layer
	 * @param input output from the lower layer
	 * @return output vector for the upper layer
	 */	
	public double[] getOutput() throws Exception{
		
		if(this.input == null){
			throw new Exception("The input vector is empty.");
		}
		
		this.output = new double[this.neurons.length]; 
		
		for(int i = 0; i < this.neurons.length; i++){
			output[i] = this.neurons[i].getOutput();
		}
		
		return this.output;
	}
	
   /**	 
 	* set the input values sent by the lower layer
	* @param output	output vector from the lower layer
	*/	
	public void setInput(double[] output){
		
		this.input = Arrays.copyOf(output, output.length);
		for(Neuron neuron : neurons){
			neuron.setInput(input);
		}
		
	}
	
	/**
	 * return the number of neurons in this layer
	 * @return number of neurons
	 */
	public int getNumNeurons(){
		return this.neurons.length;
	}
	
	/**
	 * update weights in each neuron
	 */
	public void updateWeights() throws Exception{
		for(int k = 0; k < this.deltas.length; k++){
			for(int i = 0; i < neurons.length; i++){
				neurons[i].updateWeights(this.neurons[k].weights, this.deltas[k], this.input);
			}
		}
	}
	
	/**
	 * set weight vectors in the neurons in the layer
	 * @param size # of input from the lower layer 
	 */
	public void setWeights(int size){
		
		for(int i = 0; i < neurons.length; i++){
			neurons[i].instantiateWeights(size);
		}
				
	}
	
	/**
	 * provide errors calculated in backprop process
	 * @return deltas (array)
	 * @throws if deltas is empty, throw exception
	 */
	public double[] getDeltas() throws Exception {
		if(this.deltas == null)
			throw new Exception("the delta is null.");
		
		return this.deltas;
	}

	/**
	 * undo the weight updates in each neuron of this layer
	 */
	public void undoWeightUpdate() {
		
		for(Neuron neuron : neurons){
			neuron.undoWeightUpdate();
		}
		
	}
	
	/**	
	 *  calculate the coefficient multiple for error (delta) and passed down to the lower layer
	 *  as update for method computeDelta in Neuron class
	 *  for output layer, (t_k - y_k)
	 *  for hidden layer, Sum_k (w_ik * delta_ok)  (delta_ok is the delta from the upper node)
	 *  @param index index pointing to the weight associated with the neuron in the lower layer
	 *  @return coef vector for each neuron in the layer
	 */
	public abstract void computeCoefs();

	/**
	 * return the coef value for error calculation
	 * @param index k in (t_k - y_k) for output layer and in Sum_k (w_jk * delta_ok)
	 * @return the coef value
	 */
	
	public abstract double getCoef(int index);
	
	/**
	 * return all the coefs
	 * @return coef array (double)
	 */
	
	public abstract double[] getCoefs();
	
	/**
	 * calculate deltas based on backpropagation process
	 * output layer: delta_ok = (t_k - y_k) y_k (1 - y_k) (this is done in the layer itself)
	 * hidden layer: delta_h_j = a_j (1 - a_j) Sum_k w_jk * delta_ok (the summation part should be provided by the upper layer, passed by getCoefs() method)
	 */
	public abstract void computeDeltas();

}

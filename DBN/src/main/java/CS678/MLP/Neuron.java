/* Neuron abstract class
 * the difference between hidden/output layers are error calculation
 */

package CS678.MLP;

import java.util.Arrays;
import java.util.Random;

public class Neuron {
/*	 <Forward phase>
 * 	 the following calculation is the same for any neurons in both hidden and output layers
 *   keep doing activation until the process reaches the output layer
 *   calculation:
 *   (1) get the net value
 *   (2) compute activation with sigmoid function
 *   
 *   <Backward phase>
 *   (1) calculate delta (error)
 *   (2) update weights based on the error, eta, and activation or input values
 *   it seems these calculations should be handled in layer because it requires iterations over both 
 *   delta calculated in the upper layer and the activations (or target). 
*/	

	final static boolean printout = false; // print out info.
	
	double[] weights; // weight vector for input vector
	double[] tempWeights; // temporary weight vector
	double[] input; // input vector
	double eta; // learning rate
	double net; // dot product of inputs and weights
	double output; // activation with sigmoid function
	double delta; // error
	double alpha; // momentum coefficient
	Random rand; // random number generator
	
	// constructor
	public Neuron(){}
	
	public Neuron(double eta){
		this();
		this.eta = eta;
	}
	
	public Neuron(double eta, Random rand){
		this(eta);
		this.rand = rand;
	}
	
	public Neuron(double eta, Random rand, double alpha){
		this(eta, rand);
		this.alpha = alpha;
	}
	
	/**
	 * instantiate weight and golde weight vectors from the layer
	 * @param size # of weights 
	 */
	public void instantiateWeights(int size){
		this.weights = new double[size];
		setInitialWeights(this.weights);
		this.tempWeights = Arrays.copyOf(this.weights, this.weights.length);
//		if(printout){
//			System.out.print("     ");
//			for(double weight : weights){
//				System.out.printf("%6.3f ", weight);
//			}
//			System.out.println();
//		}
	}
	
	// compute net value with dot product (Sigma (w_i x_i))
	// pre-condition: both features and weights have the same number of elements
	// post-condition: return the dot product (double)
	public double computeNet(double[] features, double[] weights) throws Exception{
		
		if(features.length != weights.length)
			throw(new Exception("(Feature Length: " + features.length + " Weight Length: " + weights.length + "   The feature and weight vectors should have the equal length."));
		
		double product = 0.0;
		
		for(int i = 0; i < features.length; i++){
			product += features[i] * weights[i];
		}
			
		return product;
	}
	
	// compute the activation using sigmoid function
	// 1/(1 + e ^ (-beta * net)) (but beta is always 1.0 in this project.
	private double computeOutput(double net, double beta){
		
		double activation = 0.0;
		
		activation = 1.0 / (1.0 + Math.exp(-1.0 * beta * net));
		
		//if(printout){
		//	System.out.println("Activation: " + activation);
		//}
		
		return activation;
	}
	
	// set learning rate
	protected void setEta(double eta){
		this.eta = eta;
	}
	
	// set error (delta)
	protected void setDelta(double delta){
		this.delta = delta;
	}
	
	/**
	 * set initial weights based on the Gaussian distribution
	 * @param weights weight vector to be initialized
	 */
	private void setInitialWeights(double[] weights){
		
		for(int i = 0; i < weights.length; i++){
			double value;

			do{
				value = rand.nextGaussian();
			}while(value == 0.0 || value > 0.1 || value < -0.1); // limit the range to avoid weight decay
			
			weights[i] = value;
		}
		
	}
	
	// set random number generator
	protected void setRand(Random rand){
		this.rand = rand;
	}
	
	/**
	 * set input vector for backward process
	 * @param input feature vector (input + bias)
	 */
	protected void setInput(double[] input){

		// the bias input is treated in NeuralNet class
		
		this.input = input;//new double[input.length];
		
//		for(int i = 0; i < input.length; i++){
//			this.input[i] = input[i]; 
//		}
		
	}

	
	// compute the error value. the update is a coefficient to be multiplied by f'(net)
	// for output, update = (t_k - y_k)
	// for hidden, update = Sum_k w_jk * delta_ok
	// update should be passed down from the upper layer
	public double computeDelta(double update){
		
		double delta = this.output * (1.0 - this.output ) * update;
		return delta;
		
	}
	
	// update weights with multiple of eta, delta, and the corresponding input
	// pre-condition: weights and input have the same length
	// post-condition: set the new weight vector with the update formula
	public void updateWeights(double[] weights, double delta, double[] input) throws Exception{

//		this.tempWeights = Arrays.copyOf(weights, weights.length);
		
		if(weights.length != input.length)
			throw(new Exception("The weight and input vectors should have the same lengths."));
		
//		if(printout){
//			System.out.print("     ");			
//		}
		for(int j = 0; j < weights.length; j++){
			updateWeights(weights, delta, input, j);
		}
//		if(printout){
//			System.out.println();
//		}
	}
	
	// update a particular weight with the update formula
	// Do I need to separate this process from the method above??
	private void updateWeights(double[] weights, double delta, double[] input, int index) {

		if(this.alpha > 0){
			double deltaWeight = this.eta * delta * input[index];
			weights[index] += deltaWeight + this.alpha * deltaWeight;			
		}
		else{
			weights[index] += this.eta * delta * input[index];
		}
//		if(printout){
//			System.out.printf("%6.3f ", weights[index]);
//		}
		
	}

	// return the activation value to the upper node 
	public double getOutput() throws Exception{
		this.net = computeNet(this.input, this.weights); // set the net value
		return this.computeOutput(this.net, 1.0);
	}
	
	// return error (delta)
	public double getDelta(){
		return this.delta;
	}

	/**
	 * undo weight update done by backpropagation
	 */
//	public void undoWeightUpdate() {
//		
//		this.weights = Arrays.copyOf(this.tempWeights, this.tempWeights.length);
//		
//	}
		
}

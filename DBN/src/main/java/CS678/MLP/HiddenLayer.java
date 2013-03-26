package CS678.MLP;

import java.util.Random;

public class HiddenLayer extends Layer{

	double[] upperCoefs; // place-holder for coefs (from output layer, (t_k - y_k); otherwise, Sum_k w_jk * delta_ok)
	double[] coefs; // place-holder for Sum_k w_jk * delta_ok (j = # of neurons in this layer)
	//int numNeuronsInLowerLayer; // # of neurons in the lower layer
	
	public HiddenLayer(int numNeurons){
		super(numNeurons);
	}

	public HiddenLayer(int numNeurons, double eta){
		super(numNeurons, eta);
	}
	
	public HiddenLayer(int numNeurons, double eta, Random rand){
		super(numNeurons, eta, rand);
	}
	
	public HiddenLayer(int numNeurons, double eta, Random rand, double alpha){
		super(numNeurons, eta, rand, alpha);
	}
	
	/**
	 * set # of neurons in the lower layer
	 */
	//public void setNumNeuronsInLowerLayer(){
	//	this.numNeuronsInLowerLayer = this.input.length;
	//}
	
	/**
	 * set the coefs provided by the upper layer for delta_hj calculation
	 * delta_hj = a_j (1 - a_j) * UpperCoef[j]
	 * @param coefs coefficients sent from the upper layer
	 */
	public void setUpperCoefs(double[] coefs){
		this.upperCoefs = new double[coefs.length];
		for(int i = 0; i < coefs.length; i++){
			this.upperCoefs[i] = coefs[i];
		}
	}

	/**
	 * calculate the coefficient of delta_hj for lower hidden layer
	 * Sum_k (w_jk * delta_ok)  (delta_ok is the delta of this layer)
	 * k ranges in the number of input ( = the number of neurons in the lower layer)
	 */
	@Override
	public void computeCoefs() {
		
		double[] coefs = new double[this.input.length];
		
		for(int j = 0; j < coefs.length; j++){ // this j corresponds to j in w_jk and coef index
			coefs[j] = 0.0; // initialization for summation
			for(int k = 0; k < this.neurons.length; k++){ // summation part
				coefs[j] += neurons[k].weights[j] * this.upperCoefs[k]; // w_jk * delta_ok 
			}
		}
		
	}

	@Override
	public double getCoef(int index) {
		return this.coefs[index]; // Sum_k w_jk * delta_ok (j = index)
	}
	
	public double[] getCoefs(){
		return this.coefs; // return all k elements of Sum_k w_jk * delta_ok
	}
	
	/**
	 * set coefs (Sum_k w_jk * delta_ok) in the variable coefs
	 * pre-condition: coefs and # of neurons in the lower layer have the same length
	 * post-condition: coefs are stored in this.coefs
	 * @param coefs coefficient vector sent from the upper layer
	 */
	public void setCoefs(double[] coefs) throws Exception{
		
		if(coefs.length != this.input.length){
			throw(new Exception("The number of neurons is different from that of coefficients."));
		}
		
		this.coefs = new double[coefs.length];
		
		for(int i = 0; i < coefs.length; i++){
			this.coefs[i] = coefs[i];
		}
		
	}

	@Override
	/**
	 * compute deltas
	 * j ranges in the number of neurons in this layer ( = # inputs for the upper layer)
	 */
	public void computeDeltas() {
		
		this.deltas = new double[this.getNumNeurons()-1];
		
		if(printout){
			System.out.println("delta: " + this.deltas.length + "  output: " + output.length + 
					"   upperCoefs: " + upperCoefs.length);
		}
		
		for(int j = 0; j < this.deltas.length; j++){
			this.deltas[j] = this.output[j] * (1 - this.output[j]) * upperCoefs[j];
		}
		
	}
	
}

package CS678.MLP;

import java.util.Random;

public class OutputLayer extends Layer{
	
	double[] diff; // t_k - y_k (target - output) vector
	double[] coefs; // the vector of all j of Sum_k w_jk * delta_ok
	double[] target; // target value vector for output layer
	
	public OutputLayer(int numNeurons, double[] target){
		super(numNeurons);
	}
	
	public OutputLayer(int numNeurons, double eta){
		super(numNeurons, eta);
	}
	
	public OutputLayer(int numNeurons, double eta, Random rand){
		super(numNeurons, eta, rand);
	}
	
	public OutputLayer(int numNeurons, double eta, Random rand, double alpha){
		super(numNeurons, eta, rand, alpha);
	}
	
	/**
	 * set the target value 
	 * target value vector (t_k)
	 * method just for output layer
	 * @param target target value vector
	 */
	public void setTarget(double[] target){
		this.target = target;
	}

	/**
	 * return the vector of (t_k - y_k) for MSE calculation
	 * @return diff vector
	 */

	public double[] getDiff(){
		return this.diff;
	}
	
	/**
	 * calculate (t_k - y_k). This can be used for MSE calculation
	 * also calculate all j of Sum_k w_jk * delta_ok for lower layer
	 * j is the index of neurons in the lower hidden layer and ranges in the number of neurons
	 * in the lower hidden layer.
	 * output layer need to do these two processes (should I separate the process into two??) 
	 */
	public void computeCoefs() {

		this.diff = new double[this.target.length]; // instantiate diff vector
		
		// calculate (t_k - y_k)
		for(int i = 0; i < this.target.length; i++){
			this.diff[i] = this.target[i] - this.output[i]; // t_k - y_k
			if(printout){
				System.out.printf("diff: %6.3f - %6.3f = %6.3f\n", this.target[i], this.output[i], this.diff[i]);
			}
		}
		
		computeDeltas();

		// calculate all Sum_k w_jk * delta_ok
		this.coefs = new double[this.input.length-1]; // coefs for all j in Sum_k w_jk * delta_ok, w/o bias
		for(int j = 0; j < coefs.length; j++){
			this.coefs[j] = 0.0;
			for(int k = 0; k < this.deltas.length; k++){ // summation over k 
				coefs[j]+= neurons[k].weights[j] * deltas[k]; // w_jk * delta_ok 
			}
		}
		
	}

	/**
	 * get a coef at a particular position j = index
	 * @param index j in Sum_k w_jk * delta_ok
	 */	
	@Override
	public double getCoef(int index) {
		return this.coefs[index]; // Sum_k w_jk * delta_ok (j = index)
	}
	
	/**
	 * get all the coefs Sum_k w_jk * delta_ok as an array
	 * @return coefs array
	 */
	@Override
	public double[] getCoefs(){
		
		return this.coefs;
	}

	/**
	 * calculate deltas (delta_ok) with (t_k - y_k) * y_k (1 - y_k) 
	 * (this is done with variables in its own layer only)
	 */
	@Override
	public void computeDeltas() {
		
		this.deltas = new double[this.getNumNeurons()]; // 
		
		if(printout){
			System.out.println("delta: " + this.deltas.length + "  diff: " + diff.length + "   output: " 
					+ output.length + "   neurons: " + neurons.length);
			System.out.print("Delta_ok: ");
		}
		for(int k = 0; k < deltas.length; k++){
			this.deltas[k] = this.diff[k] * this.output[k] * (1 - this.output[k]);
			if(printout){
				System.out.printf("%6.3f (%6.3f * %6.3f * (1 - %6.3f))", deltas[k], 
						this.diff[k], this.output[k], this.output[k]);
			}
		}
		if(printout)
			System.out.println();
		
	}
	
}

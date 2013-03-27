package CS678.DBN;

import java.io.Serializable;

import cs678.tools.Matrix;

public class DBN implements Serializable {
	
	private static final long serialVersionUID = 1L;

	static final int intrimSamplingSize = 60000;
	
	private RBM[] rbms;
	private int layerCount;
	private int maxSampleSize;
	private Matrix trainingData, testData;
	private int[] numHiddenNodes;
	private double[] criteria;
	private double[] thresholds;
	
	public DBN(){
		
	}
	
	public DBN(int layerCount){
		this();
		this.layerCount = layerCount;
		rbms = new RBM[this.layerCount];
	}

	public DBN(int layerCount, Matrix trainingData){
		this(layerCount);
		this.trainingData = trainingData;
	}
	
	public DBN(int layerCount, Matrix trainingData, int[] numHiddenNodes){
		this(layerCount, trainingData);
		this.numHiddenNodes = numHiddenNodes;
	}
	
	public DBN(int layerCount, Matrix trainingData, int[] numHiddenNodes, double[] criteria){
		this(layerCount, trainingData, numHiddenNodes);
		this.criteria = criteria;		
	}
	
	public DBN(int layerCount, Matrix trainingData, Matrix testData, int[] numHiddenNodes, 
			double[] criteria){
		this(layerCount, trainingData, numHiddenNodes, criteria);
		this.testData = testData;
	}
	
	public DBN(int layerCount, Matrix trainingData, Matrix testData, int[] numHiddenNodes, 
			double[] criteria, double[] thresholds){
		this(layerCount, trainingData, testData, numHiddenNodes, criteria);
		this.thresholds = thresholds;
	}
	
	public DBN(int layerCount, Matrix trainingData, Matrix testData, int[] numHiddenNodes, 
			double[] criteria, double[] thresholds, int maxSampleSize){
		this(layerCount, trainingData, testData, numHiddenNodes, criteria, thresholds);
		this.maxSampleSize = maxSampleSize;
	}
	
	public void train() throws Exception{
		
		Matrix data = new Matrix();
		Matrix testData = this.testData;
		for(int i = 0; i < this.layerCount; i++){
			if(i == 0){
				data = this.trainingData;				
			}
			if(i == this.layerCount - 1)
				this.rbms[i] = new RBM(data, this.numHiddenNodes[i], data.cols()-1, 
						(i+1), this.criteria[i], this.thresholds[i], maxSampleSize);
			else
				this.rbms[i] = new RBM(data, this.numHiddenNodes[i], data.cols()-1, 
						(i+1), this.criteria[i], this.thresholds[i], intrimSamplingSize);
				
			this.rbms[i].update();
			testData = this.rbms[i].convertTestSet(testData);
			data = this.rbms[i].nextInputs();
		}
		
	}
	
}

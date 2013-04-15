// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
package CS678.MLP;

import cs678.tools.Matrix;

public abstract class SupervisedLearner {

	private String fileName;
	private String evalMethod;
	private String learnerName;
	private boolean useValidation;
	
	private Matrix testFeatures;
	private Matrix testLabels;
	
	// Before you call this method, you need to divide your data
	// into a feature matrix and a label matrix.
	public abstract void train(Matrix features, Matrix labels) throws Exception;

	// A feature vector goes in. A label vector comes out. (Some supervised
	// learning algorithms only support one-dimensional label vectors. Some
	// support multi-dimensional label vectors.)
	public abstract void predict(double[] features, double[] labels) throws Exception;

	// The model must be trained before you call this method. If the label is nominal,
	// it returns the predictive accuracy. If the label is continuous, it returns
	// the root mean squared error (RMSE). If confusion is non-NULL, and the
	// output label is nominal, then confusion will hold stats for a confusion matrix.
	public double measureAccuracy(Matrix features, Matrix labels, Matrix confusion) throws Exception
	{
		if(features.rows() != labels.rows())
			throw(new Exception("Expected the features and labels to have the same number of rows"));
		if(labels.cols() != 1)
			throw(new Exception("Sorry, this method currently only supports one-dimensional labels"));
		if(features.rows() == 0)
			throw(new Exception("Expected at least one row"));

		int labelValues = labels.valueCount(0);
		System.out.println("label values: " + labelValues);
		if(labelValues == 0) // If the label is continuous...
		{
			// The label is continuous, so measure root mean squared error
			double[] pred = new double[1];
			double sse = 0.0;
			for(int i = 0; i < features.rows(); i++)
			{
				double[] feat = features.row(i);
				double[] targ = labels.row(i);
				pred[0] = 0.0; // make sure the prediction is not biassed by a previous prediction
				predict(feat, pred);
				double delta = targ[0] - pred[0];
				sse += (delta * delta);
			}
			return Math.sqrt(sse / features.rows());
		}
		else
		{
			// The label is nominal, so measure predictive accuracy
			if(confusion != null)
			{
				confusion.setSize(labelValues, labelValues);
				for(int i = 0; i < labelValues; i++)
					confusion.setAttrName(i, labels.attrValue(0, i));
			}
			int correctCount = 0;
			double[] prediction = new double[1];
			for(int i = 0; i < features.rows(); i++)
			{
				double[] feat = features.row(i);
				int targ = (int)labels.get(i, 0);
				if(targ >= labelValues)
					throw new Exception("The label is out of range");
				predict(feat, prediction);
				int pred = (int)prediction[0];
				if(confusion != null)
					confusion.set(targ, pred, confusion.get(targ, pred) + 1);

				//System.out.println("Output: " + prediction[0] + " Target: " + targ);

				if(pred == targ)
					correctCount++;
			}
			return (double)correctCount / features.rows();
		}
	}
	
	public void setTestData(Matrix features, Matrix labels){
		this.testFeatures = features;
		this.testLabels = labels;
	}
	
	public Matrix getTestFeatures(){
		return this.testFeatures;
	}
	
	public Matrix getTestLabels(){
		return this.testLabels;
	}

	// I added this method for decision tree (graphviz)
	public void setFileName(String name){
		this.fileName = name;
	}
	
	// I added this method for decision tree (graphviz)
	public String getFileName(){
		return this.fileName;
	}

	// I added this method for decision tree (graphviz)
	public void setEvaluationMethod(String evalMethod) {
		this.evalMethod = evalMethod;
	}
	
	// I added this method for decision tree (graphviz)
	public String getEvaluationMethod(){
		return this.evalMethod;
	}
	
	// I added this method for decision tree
	public void setIfValudationIsUsed(boolean useValidation){
		this.useValidation = useValidation;
	}
	
	// I added this method for decision tree
	public boolean useValidation(){
		return this.useValidation;
	}
	
	public void setLearnerName(String learnerName){
		this.learnerName = learnerName;
	}
	
	public String getLearnerName(){
		return this.learnerName;
	}
}

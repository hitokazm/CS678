package CS678.DBN;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cs678.tools.Matrix;

public class Main {

	public static void main(String[] args) throws Exception{
		
//		Matrix trainingData = new Matrix();
//		trainingData.loadArff("data/train.arff");
//		trainingData.normalize();
//		data.print();
		
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream("data/train.matrix"));
		Matrix trainingData = (Matrix) ois.readObject();
		ois.close();
		
		Matrix testData = new Matrix();
		testData.loadArff("data/test.arff");
		
		//ois = new ObjectInputStream(new FileInputStream("data/testset1.matrix"));
		//Matrix testData = (Matrix) ois.readObject();
		//ois.close();
		
		int[] numHiddenNodes = {400, 200, 60};
		double[] criteria = {1E-3, 1E-3, 1E-3};
		double[] thresholds = {1, 1, 1};
		int maxSampleSize = 100000;
		
		DBN dbn = new DBN(thresholds.length, trainingData, testData, numHiddenNodes, criteria, 
				thresholds, maxSampleSize);
		
		dbn.train();
		
//		//RBM rbm = new RBM(data, 2, 1, 1, 1E-5);
//		RBM rbm = new RBM(data, 1000, 784, 1, 5E-3);
//		
//		rbm.update();
//		
//		Matrix next = rbm.nextInputs();
//		
//		next.print();
		
//		ObjectInputStream ois = new ObjectInputStream(new FileInputStream("data/rbm2.matrix"));
//		Matrix data = (Matrix) ois.readObject();
//		ois.close();
//		
//		RBM rbm = new RBM(data, 1000, 1000, 3, 5E-3);
//		
//		rbm.update();
//		
//		Matrix next = rbm.nextInputs();
		
//		Matrix test = new Matrix(0, 11);
//		test.setOutputClass("class", 9);
//		double[] datum = {1,1,1,1,1,1,1,1,1,1,3};
//		test.addRow(datum);
//		test.print();
		
		//System.out.println(data.export());
		
		//ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("data/train.matrix"));
		//oos.writeObject(data);
//		
//		
//		data2.print();

	}

}

package cs678.bptt;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import cs678.tools.MLSystemManager;
import cs678.tools.Matrix;

public class Main 
{
	
	protected final static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
	protected final static Level logLevel = Level.OFF;
	
	private final static String filename = "activity";
	
    public static void main( String[] args ) throws Exception
    {
//		MLSystemManager ml = new MLSystemManager();
//		Matrix matrix = new Matrix();
//		matrix.loadArff("data/activity.arff");
//		matrix.normalize();
//		Matrix small = new Matrix(matrix, 0, 3, matrix.rows(), 3);
//		small.print();
    	
    	logger.setLevel(logLevel);
    
    	int k = 2;
    	
		Matrix data = new Matrix();
		if(filename.equals("activity"))
			//data.loadArff("data/activity-5cols.arff");
			data.loadArff("data/stocks.arff");
		else
			data.loadArff("data/fake.arff");
		
		Matrix features = new Matrix(data, 0, 0, data.rows(), data.cols() - 1);
		
		if(filename.equals("activity")){
			//features = new Matrix(features, 0, 3, features.rows(), 3);
			features.normalize();
		}
		
		Matrix labels = new Matrix(data, 0, data.cols() - 1, data.rows(), 1);
		
		//features.normalize();
		
		BPTT learner = new BPTT(k);
		
		learner.train(features, labels);


    	
    }
}

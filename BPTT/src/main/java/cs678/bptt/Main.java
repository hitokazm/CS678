package cs678.bptt;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import cs678.tools.MLSystemManager;
import cs678.tools.Matrix;

public class Main 
{
	
	protected final static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
	protected final static Level logLevel = Level.INFO;
	
    public static void main( String[] args ) throws Exception
    {
//		MLSystemManager ml = new MLSystemManager();
//		Matrix matrix = new Matrix();
//		matrix.loadArff("data/activity.arff");
//		matrix.normalize();
//		Matrix small = new Matrix(matrix, 0, 3, matrix.rows(), 3);
//		small.print();
    	
    	logger.setLevel(logLevel);
    	
    	Layer layer = new OutputLayer(2, 0.1, new Random(), 0.5, 10);
    	Layer layer2 = new HiddenLayer(2, 0.1, new Random(), 0.9, 2, 10);
    }
}

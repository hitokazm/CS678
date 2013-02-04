package cs678.tools;

public class ArgParser {

	String projectName;
	String arff;
	String learner;
	String evaluation;
	String evalExtra;
	boolean verbose;
	boolean normalize;

	//You can add more options for specific learning models if you wish
	public ArgParser(String[] argv) {
		try{

		 	for (int i = 0; i < argv.length; i++) {

		 		if (argv[i].equals("-V"))
		 		{
		 			verbose = true;
		 		}
		 		else if (argv[i].equals("-N"))
		 		{
		 			normalize = true;
		 		}
					else if (argv[i].equals("-A"))
					{
						arff = argv[++i];
					}
					else if (argv[i].equals("-L"))
					{
						learner = argv[++i];
					}
					else if (argv[i].equals("-E"))
					{
						evaluation = argv[++i];
						if (argv[i].equals("static"))
						{
							//expecting a test set name
							evalExtra = argv[++i];
						}
						else if (argv[i].equals("random"))
						{
							//expecting a double representing the percentage for testing
							//Note stratification is NOT done
							evalExtra = argv[++i];
						}
						else if (argv[i].equals("cross"))
						{
							//expecting the number of folds
							evalExtra = argv[++i];
						}
						else if (!argv[i].equals("training"))
						{
							System.out.println("Invalid Evaluation Method: " + argv[i]);
							System.exit(0);
						}
					}
					else
					{
						System.out.println("Invalid parameter: " + argv[i]);
						System.exit(0);
					}
		  	}
	 
			}
			catch (Exception e) {
				System.out.println("Usage:");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
				System.out.println("OPTIONS:");
				System.out.println("-V Print the confusion matrix and learner accuracy on individual class values\n");
				
				System.out.println("Possible evaluation methods are:");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTesting]");
			  	System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
				System.exit(0);
			}
			
			if (arff == null || learner == null || evaluation == null)
			{
				System.out.println("Usage:");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
				System.out.println("OPTIONS:");
				System.out.println("-V Print the confusion matrix and learner accuracy on individual class values");
				System.out.println("-N Use normalized data");
				System.out.println();
				System.out.println("Possible evaluation methods are:");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
				System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTesting]");
			  	System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
				System.exit(0);
			}
		}
 
	//The getter methods
	public String getARFF(){ return arff; }	
	public String getLearner(){ return learner; }	 
	public String getEvaluation(){ return evaluation; }	
	public String getEvalParameter() { return evalExtra; }
	public boolean getVerbose() { return verbose; } 
	public boolean getNormalize() { return normalize; }
	
}

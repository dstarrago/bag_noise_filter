/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S�nchez (luciano@uniovi.es)
    J. Alcal�-Fdez (jalcala@decsai.ugr.es)
    S. Garc�a (sglopez@ujaen.es)
    A. Fern�ndez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

/**
 * <p>
 * @author Written by Jose A. Saez Munoz, research group SCI2S (Soft Computing and Intelligent Information Systems).
 * DECSAI (DEpartment of Computer Science and Artificial Intelligence), University of Granada - Spain.
 * Date: 06/01/10
 * @version 1.0
 * @since JDK1.6
 * </p>
 */

package keel.Algorithms.Preprocess.NoiseFilters.ClassificationFilter;

import java.util.Arrays;
import java.util.Random;
import java.util.Vector;
import keel.Dataset.Instance;


/**
 * <p>
 * The Classification Filter begins with n equal-sized disjoint subsets of the training set E (done with n-fold cross validation)
 * and the empty output set A of detected noisy examples. The main loop is repeated for each training subset Ei. Ey is formed
 * which includes all examples from E except those in Ei. Set Ey is used as the input for an arbitrary inductive learning
 * algorithm that induces a hypothesis (a classifier) Hy. Those examples from Ei for which the hypothesis Hy does not give the
 * correct classification are added to A as potentially noisy examples.
 * Reference: 1999-Gamberger-ICML
 * </p>
 */
public class ClassificationFilter {
	
	static protected String tempFolder = System.getProperty("java.io.tmpdir") + "/";
	private Instance[] instancesTrain;	// all the instances of the training set
	private Vector[] partitions;		// indexes of the instances in each partition
	private boolean[] correctlyLabeled;	// indicates if the instance is correctly labeled
	private int[] classExFinal;
	private PartitionScheme partSch;	// partition scheme used
	public static boolean[] cleanExENN;
        private final String timeStamp;
        private final Random random = new Random();
	

//*******************************************************************************************************************************

	/**
	 * <p>
	 * It initializes the partitions from training set
	 * </p>
	 */
	public ClassificationFilter(String newDataset, boolean[] cleanEx){
		
        timeStamp = String.valueOf(System.currentTimeMillis()) + "-" + String.valueOf(random.nextInt());
        cleanExENN = new boolean[cleanEx.length];
        for(int i = 0 ; i < cleanEx.length ; ++i)
        	cleanExENN[i] = cleanEx[i];
		
		Parameters.trainInputFile = newDataset;

		// create instances
		partSch = new PartitionScheme();			// create the partitions
		instancesTrain = partSch.getInstances();	// get all the instances of training set
		partitions = partSch.getPartitions();
		
		partSch.createPartitionFiles(timeStamp);
		
		correctlyLabeled = new boolean[Parameters.numInstances];
		Arrays.fill(correctlyLabeled, true);
		
		classExFinal = new int[Parameters.numInstances];
		Arrays.fill(classExFinal, -1);
	}
	
//*******************************************************************************************************************************

  /**
   * <p>
   * It initializes the partitions from training set
   * </p>
   * @param paramName parameter name
   * @return true if the parameter is real, false otherwise
   */
  public Vector run(){
    for(int partTest = 0 ; partTest < Parameters.numPartitions ; ++partTest){
      C45 c45 = null;
      try {
        String trainFN = String.format("%strain-%s-%d.dat", tempFolder, timeStamp, partTest + 1);
        String testFN = String.format("%stest-%s-%d.dat", tempFolder, timeStamp, partTest + 1);
        c45 = new C45(trainFN,testFN);
      } catch (Exception e) {
        e.printStackTrace();
      }
      int[] pre = c45.getPredictions();
      for(int i = 0 ; i < partitions[partTest].size() ; ++i){
        correctlyLabeled[(Integer)partitions[partTest].get(i)] = (pre[i] == instancesTrain[(Integer)partitions[partTest].get(i)].getOutputNominalValuesInt(0));
        classExFinal[(Integer)partitions[partTest].get(i)] = pre[i];
      }
    }
    partSch.deletePartitionFiles(timeStamp);
    int numnoisyex = 0;
    for(int j = 0 ; j < Parameters.numInstances ; ++j)
      if(correctlyLabeled[j] ==  false)
        numnoisyex++;
    for(int j = 0 ; j < Parameters.numInstances ; ++j)
      correctlyLabeled[j] = !correctlyLabeled[j];
    Vector finalreturn = new Vector();
    finalreturn.add(correctlyLabeled);
    finalreturn.add(classExFinal);
    finalreturn.add(numnoisyex);
    return finalreturn;
  }
	
}
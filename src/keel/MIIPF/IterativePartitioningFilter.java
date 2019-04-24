/** *********************************************************************
 *
 * This file is part of KEEL-software, the Data Mining tool for regression,
 * classification, clustering, pattern mining and so on.
 *
 * Copyright (C) 2004-2010
 *
 * F. Herrera (herrera@decsai.ugr.es)
 * L. S�nchez (luciano@uniovi.es)
 * J. Alcal�-Fdez (jalcala@decsai.ugr.es)
 * S. Garc�a (sglopez@ujaen.es)
 * A. Fern�ndez (alberto.fernandez@ujaen.es)
 * J. Luengo (julianlm@decsai.ugr.es)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/
 *
 ********************************************************************* */
/**
 * <p>
 * @author Written by Jose A. Saez Munoz, research group SCI2S (Soft Computing
 * and Intelligent Information Systems). DECSAI (DEpartment of Computer Science
 * and Artificial Intelligence), University of Granada - Spain. Date: 06/01/10
 * @version 1.0
 * @since JDK1.6
 * </p>
 */
package keel.MIIPF;

import keel.Algorithms.Preprocess.NoiseFilters.IterativePartitioningFilter.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;
import java.util.Vector;
import keel.Algorithms.Genetic_Rule_Learning.Globals.FileManagement;
import org.core.Fichero;
import keel.Dataset.Attribute;
import keel.Dataset.Attributes;
import keel.Dataset.Instance;
import keel.Dataset.InstanceSet;

/**
 * <p>
 * The Ensemble Filter... Reference: 1999-Brodley-JAIR
 * </p>
 */
public class IterativePartitioningFilter {

  static protected String tempFolder = System.getProperty("java.io.tmpdir") + "/";
  private Instance[] instancesTrain;	// all the instances of the training set
  private boolean[][] correctlyLabeled;	// indicates if the instance is correctly labeled
  private PartitionScheme partSch;	// partition scheme used

  private Vector noisyInstances;		// indexes of the noisy instances from training set
  private Vector ClassNoisyIns;

  private int numFilters;
  private Vector NoisyIndexesByIteration;
  private Vector NoisyClassByIteration;

  public static boolean[] cleanExENN;
  private String timeStamp;
  private Random random = new Random();

//*******************************************************************************************************************************
  /**
   * <p>
   * It initializes the partitions from training set
   * </p>
   */
  public IterativePartitioningFilter(String newDataset, boolean[] cleanEx) {

    timeStamp = String.valueOf(System.currentTimeMillis()) + "-" + String.valueOf(random.nextInt());
    Parameters.trainInputFile = newDataset;

    numFilters = Parameters.numPartitions;

    InstanceSet is = new InstanceSet();
    Attributes.clearAll();
    try {
      is.readSet(Parameters.trainInputFile, true);
    } catch (Exception e) {
      System.exit(1);
    }

    instancesTrain = is.getInstances();
    Parameters.numClasses = Attributes.getOutputAttribute(0).getNumNominalValues();
    Parameters.numAttributes = Attributes.getInputAttributes().length;
    Parameters.numInstances = instancesTrain.length;

    NoisyIndexesByIteration = new Vector();
    NoisyClassByIteration = new Vector();

    cleanExENN = new boolean[cleanEx.length];
    for (int i = 0; i < cleanEx.length; ++i) {
      cleanExENN[i] = cleanEx[i];
    }
  }

//*******************************************************************************************************************************
  /**
   * <p>
   * It initializes the partitions from training set
   * </p>
   *
   * @param paramName parameter name
   * @return true if the parameter is real, false otherwise
   */
  public Vector run() {

    boolean stop = false;
    int countToStop = 0;
    int iter = 0;
    Instance[] insAux;

    partSch = new PartitionScheme(Parameters.trainInputFile, Parameters.numPartitions);
    insAux = partSch.getInstances();
    partSch.createPartitionFiles(timeStamp);

    noisyInstances = new Vector();
    ClassNoisyIns = new Vector();

    String paramFN = String.format("%sIPF_train-%s_0.txt", tempFolder, timeStamp);
    createDatasetTrain(Parameters.trainInputFile, paramFN);

    while (!stop) {

      correctlyLabeled = new boolean[numFilters][insAux.length];
      for (int i = 0; i < numFilters; ++i) {
        for (int j = 0; j < insAux.length; ++j) {
          correctlyLabeled[i][j] = true;
        }
      }

      int[][] pre = new int[Parameters.numPartitions][];
      String paramFNx = String.format("%sIPF_train-%s_%d.txt", tempFolder, timeStamp, iter);
      for (int k = 0; k < Parameters.numPartitions; ++k) {

        // to train C45 classifier
        C45 c45 = null;
        
        String trainFN = String.format("%strain-%s-%d.dat", tempFolder, timeStamp, k + 1);
        try {
          c45 = new C45(trainFN, paramFNx);
        } catch (Exception e) {
          e.printStackTrace();
        }

        pre[k] = c45.getPredictions();
        for (int i = 0; i < insAux.length; ++i) {
          correctlyLabeled[k][i] = (pre[k][i] == insAux[i].getOutputNominalValuesInt(0));
        }
      }

      noisyInstances = new Vector();
      ClassNoisyIns = new Vector();

      /*if(Parameters.filterType.equals("consensus")){
				
				boolean ruido;
				
				for(int j = 0 ; j < insAux.length ; ++j){
					ruido = true;
					for(int i = 0 ; i < numFilters && ruido ; ++i)
						if(correctlyLabeled[i][j] ==  true){
							ruido = false;
						}
					
					if(ruido && correctlyLabeled[partSch.getPartitionOfInstance(j)][j] == false)
						noisyInstances.add(j);
				}
			}*/
      if (Parameters.filterType.equals("majority")) {

        for (int j = 0; j < insAux.length; ++j) {
          int cont = 0;
          for (int i = 0; i < numFilters; ++i) {
            if (correctlyLabeled[i][j] == false) {
              cont++;
            }
          }

          if (cont > (double) numFilters / 2 && correctlyLabeled[partSch.getPartitionOfInstance(j)][j] == false) {
            noisyInstances.add(j);

            //meter una clase para el ejemplo j!!----------
            int[] votos_aux = new int[Parameters.numClasses];
            Arrays.fill(votos_aux, 0);

            for (int pa = 0; pa < numFilters; ++pa) {
              votos_aux[pre[pa][j]]++;
            }

            int claM = -1;
            int max = -1;
            for (int cls = 0; cls < Parameters.numClasses; ++cls) {

              if (votos_aux[cls] > max) {
                max = votos_aux[cls];
                claM = cls;
              }

              // si iguala el maximo deberia decidir la clase aleatoriamente
              //else if(votos_aux[cls] == max){
              //	
              //}
            }

            ClassNoisyIns.add(claM);
            //---------------------------------------------
          }
        }
      }

      // remove instances from training set and create new partitions
      // elimino las instancias recogidas hasta ahora de la ultima particion
      
      String paramFNnext = String.format("%sIPF_train-%s_%d.txt", tempFolder, timeStamp, iter + 1);
      createDatasetTrain(paramFNx, paramFNnext);

      // delete old partition files and old IPF_train file
      partSch.deletePartitionFiles(timeStamp);
      File fi = new File(paramFNx);
      fi.delete();

      if (noisyInstances.size() < Parameters.numInstances * 0.01) {
        countToStop++;
      } else {
        countToStop = 0;
      }

      if (countToStop == 3) {
        stop = true;
      } else {
        // create the new partition files
        partSch = new PartitionScheme(paramFNnext, Parameters.numPartitions);
        insAux = partSch.getInstances();
        partSch.createPartitionFiles(timeStamp);
      }

      // guardo las instancias con ruido de esta iteracion
      NoisyIndexesByIteration.add(noisyInstances);
      NoisyClassByIteration.add(ClassNoisyIns);

      iter++;
    }

    // la solucion esta... en el fichero IPF_train_(iter)!!!!!
    //createDatasets("IPF_train_"+iter+".txt", "OUT.dat");
    String paramFNx = String.format("%sIPF_train-%s_%d.txt", tempFolder, timeStamp, iter);
    File fi = new File(paramFNx);
    fi.delete();

    // ---------------------------------------- vector con ejemplos con ruido
    // contar el numero de noisy examples
    int sizenoisytotal = 0;
    for (int t1 = 0; t1 < NoisyIndexesByIteration.size(); ++t1) {
      sizenoisytotal += ((Vector) NoisyIndexesByIteration.get(t1)).size();
    }

    boolean[] globalnoise = new boolean[Parameters.numInstances];
    Arrays.fill(globalnoise, false);

    int[] classglobal = new int[Parameters.numInstances];
    Arrays.fill(classglobal, -1);

    for (int t1 = 0; t1 < NoisyIndexesByIteration.size(); ++t1) {

      boolean[] thisIteration = new boolean[Parameters.numInstances];
      Arrays.fill(thisIteration, false);

      int[] thisIterationCLASS = new int[Parameters.numInstances];
      Arrays.fill(thisIterationCLASS, -1);

      for (int t2 = 0; t2 < ((Vector) NoisyIndexesByIteration.get(t1)).size(); ++t2) {

        boolean seguiru = true;
        int indACT = 0;
        int indiceBus = (Integer) ((Vector) NoisyIndexesByIteration.get(t1)).get(t2);
        int classBus = (Integer) ((Vector) NoisyClassByIteration.get(t1)).get(t2);

        for (int kk = 0; kk < Parameters.numInstances && seguiru; ++kk) {
          if (!globalnoise[kk]) {
            if (indiceBus == indACT) {
              thisIteration[kk] = true;
              thisIterationCLASS[kk] = classBus;
              seguiru = false;
            } else {
              indACT++;
            }
          }
        }
      }

      // actualizo global
      for (int kk = 0; kk < Parameters.numInstances; ++kk) {
        if (thisIteration[kk]) {
          globalnoise[kk] = true;
          classglobal[kk] = thisIterationCLASS[kk];
        }
      }

    }

    /*
		int[] noisyEx = new int[sizenoisytotal];
		int[] noisyCl = new int[sizenoisytotal];
		
		int contnoisy = 0;
		for(int t1 = 0 ; t1 < Parameters.numInstances ; ++t1){
			if(globalnoise[t1]){
				noisyEx[contnoisy] = t1; 
				noisyCl[contnoisy] = classglobal[t1];
				contnoisy++;
			}
		}
     */
    Vector res_final = new Vector();
    res_final.add(globalnoise);
    res_final.add(classglobal);
    res_final.add(sizenoisytotal);

    return res_final;
  }

//*******************************************************************************************************************************
  /**
   * <p>
   * It applies the changes to remove the noise
   * </p>
   */
  public void createDatasets(String trainIN, String trainOUT) {

    // to create the train file-----------------------------------------
    try {
      String s;
      File Archi1 = new File(trainIN);
      File Archi2 = new File(trainOUT);
      BufferedReader in;
      in = new BufferedReader(new FileReader(Archi1));
      PrintWriter out = new PrintWriter(new FileWriter(Archi2));

      while ((s = in.readLine()) != null) {
        out.println(s);
      }

      in.close();
      out.close();
    } catch (Exception e) {
      e.printStackTrace();
    }

  }

//*******************************************************************************************************************************
  /**
   * <p>
   * It apllies the changes to remove the noise
   * </p>
   */
  public void createDatasetTrain(String trainIN, String trainOUT) {

    // to check if the noisyInstances vector is ordered
    if (noisyInstances.size() > 0) {
      int menor = (Integer) noisyInstances.get(0);
      boolean correcto = true;
      for (int i = 1; i < noisyInstances.size() && correcto; ++i) {
        if ((Integer) noisyInstances.get(i) <= menor) {
          correcto = false;
        } else {
          menor = (Integer) noisyInstances.get(i);
        }
      }

      if (!correcto) {
        System.out.println("\n\nERROR: The noisy vector is not ordered!");
        System.exit(-1);
      }
    }
    // create the files...
    InstanceSet is = new InstanceSet();
    Instance[] instances = null;
    try {
      is.readSet(trainIN, false);
      instances = is.getInstances();
    } catch (Exception e1) {
      e1.printStackTrace();
    }
    int numAtt = Attributes.getInputNumAttributes();

    // create an array with the non-noisy instances
    Vector validInstances = new Vector();
    int cont = 0;

    if (noisyInstances.size() == 0) {
      for (int i = 0; i < instances.length; ++i) {
        validInstances.add(i);
      }
    } else {
      for (int i = 0; i < instances.length; ++i) {
        if (cont < noisyInstances.size() && (Integer) noisyInstances.get(cont) == i) {
          cont++;
        } else {
          validInstances.add(i);
        }
      }
    }

    // to create the train file-----------------------------------------
    String header = "";
    header = "@relation " + Attributes.getRelationName() + "\n";
    header += Attributes.getInputAttributesHeader();
    header += Attributes.getOutputAttributesHeader();
    header += Attributes.getInputHeader() + "\n";
    header += Attributes.getOutputHeader() + "\n";
    header += "@data\n";

    Attribute[] att = Attributes.getInputAttributes();

    try {

      Fichero.escribeFichero(trainOUT, header);

      for (int k = 0; k < validInstances.size(); k++) {

        int i = (Integer) validInstances.get(k);

        boolean[] missing = instances[i].getInputMissingValues();
        String newInstance = "";

        for (int j = 0; j < numAtt; j++) {

          if (missing[j]) {
            newInstance += "?";
          } else {
            if (att[j].getType() == Attribute.REAL) {
              newInstance += instances[i].getInputRealValues(j);
            }
            if (att[j].getType() == Attribute.INTEGER) {
              newInstance += (int) instances[i].getInputRealValues(j);
            }
            if (att[j].getType() == Attribute.NOMINAL) {
              newInstance += instances[i].getInputNominalValues(j);
            }
          }

          newInstance += ", ";
        }

        String className = instances[i].getOutputNominalValues(0);
        newInstance += className + "\n";

        Fichero.AnadirtoFichero(trainOUT, newInstance);

      }

    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }

  }

}
